"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import ipdb as pdb
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from LiLY.modules.components.beta import BetaVAE_MLP
from LiLY.modules.metrics.correlation import compute_mcc
from LiLY.modules.components.tc import Discriminator, permute_dims
from LiLY.modules.components.transforms import ComponentWiseSpline
from LiLY.modules.components.mlp import (Inference, 
                             NLayerLeakyMLP)
from LiLY.modules.components.transition import (MBDTransitionPrior, NPTransitionPrior)


class SRNNSyntheticNS(pl.LightningModule):

    def __init__(
        self, 
        input_dim,
        length,
        z_dim, 
        lag,
        nclass,
        hidden_dim=128,
        trans_prior='L',
        infer_mode='R',
        bound=5,
        count_bins=8,
        order='linear',
        lr=1e-4,
        beta=0.0025,
        gamma=0.0075,
        sigma=1e-6,
        bias=False,
        use_warm_start=False,
        spline_pth=None,
        decoder_dist='gaussian',
        correlation='Pearson'):
        '''Bi-directional inference network'''
        super().__init__()
        # Transition prior must be L (Linear), PNL (Post-nonlinear) or IN (Interaction)
        assert trans_prior in ('L', 'PNL','IN', 'NP')
        # Inference mode must be R (Recurrent) or F (Factorized)
        assert infer_mode in ('R', 'F')

        self.z_dim = z_dim
        self.input_dim = input_dim
        self.lr = lr
        self.lag = lag
        self.length = length
        self.nclass = nclass
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.l1 = 0.0001
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.infer_mode = infer_mode

        # Recurrent/Factorized inference
        if infer_mode == 'R':
            self.enc = NLayerLeakyMLP(in_features=input_dim, 
                                      out_features=z_dim, 
                                      num_layers=3, 
                                      hidden_dim=hidden_dim) 

            self.dec = NLayerLeakyMLP(in_features=z_dim, 
                                      out_features=input_dim, 
                                      num_layers=3, 
                                      hidden_dim=hidden_dim) 

            # Bi-directional hidden state rnn
            self.rnn = nn.GRU(input_size=z_dim, 
                              hidden_size=hidden_dim, 
                              num_layers=1, 
                              batch_first=True, 
                              bidirectional=True)
            
            # Inference net
            self.net = Inference(lag=lag,
                                 z_dim=z_dim, 
                                 hidden_dim=hidden_dim, 
                                 num_layers=2)

        elif infer_mode == 'F':
            self.net = BetaVAE_MLP(input_dim=input_dim, 
                                   z_dim=z_dim, 
                                   hidden_dim=hidden_dim)

        # Initialize transition prior
        if trans_prior == 'L':
            self.transition_prior = MBDTransitionPrior(lags=lag, 
                                                       latent_size=z_dim, 
                                                       bias=bias)
        elif trans_prior == 'PNL':
            self.transition_prior = PNLTransitionPrior(lags=lag, 
                                                       latent_size=z_dim, 
                                                       num_layers=3, 
                                                       hidden_dim=hidden_dim)
        elif trans_prior == 'IN':
            self.transition_prior = INTransitionPrior()
        
        elif trans_prior == 'NP':
            self.transition_prior = NPTransitionPrior(lags=lag, 
                                                      latent_size=z_dim, 
                                                      num_layers=3, 
                                                      hidden_dim=hidden_dim)
        
        # Spline flow model to learn the noise distribution
        self.spline_list = []
        for i in range(self.nclass):
            spline = ComponentWiseSpline(input_dim=z_dim,
                                         bound=bound,
                                         count_bins=count_bins,
                                         order=order)

            if use_warm_start:
                spline.load_state_dict(torch.load(spline_pth, 
                                                  map_location=torch.device('cpu')))

                print("Load pretrained spline flow", flush=True)
            self.spline_list.append(spline)
        self.spline_list = nn.ModuleList(self.spline_list)
        # self.dictionary = {0:self.spline_list[0], 1:self.spline_list[1], 2:self.spline_list[2]}

        # FactorVAE
        self.discriminator = Discriminator(z_dim = z_dim*self.length)

        # Register skeleton logits
        self.logits = nn.Parameter(torch.randn(self.z_dim, self.z_dim*self.lag)+4)

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def inference(self, ft, random_sampling=True):
        ## bidirectional lstm/gru 
        # input: (batch, seq_len, z_dim)
        # output: (batch, seq_len, z_dim)
        output, h_n = self.rnn(ft)
        batch_size, length, _ = output.shape
        # beta, hidden = self.gru(ft, hidden)
        ## sequential sampling & reparametrization
        ## transition: p(zt|z_tau)
        zs, mus, logvars = [], [], []
        for tau in range(self.lag):
            zs.append(torch.zeros((batch_size, self.z_dim), device=output.device))

        for t in range(length):
            mid = torch.cat(zs[-self.lag:], dim=1)
            inputs = torch.cat([mid, output[:,t,:]], dim=1)    
            distributions = self.net(inputs)
            mu = distributions[:, :self.z_dim]
            logvar = distributions[:, self.z_dim:]
            zt = self.reparameterize(mu, logvar, random_sampling)
            zs.append(zt)
            mus.append(mu)
            logvars.append(logvar)

        zs = torch.squeeze(torch.stack(zs, dim=1))
        # Strip the first L zero-initialized zt 
        zs = zs[:,self.lag:]
        mus = torch.squeeze(torch.stack(mus, dim=1))
        logvars = torch.squeeze(torch.stack(logvars, dim=1))
        return zs, mus, logvars
    
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def reconstruction_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'gaussian':
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'sigmoid_gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        return recon_loss

    def forward(self, batch):
        x = batch['xt']
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        if self.infer_mode == 'R':
            ft = self.enc(x_flat)
            ft = ft.view(batch_size, length, -1)
            zs, mus, logvars = self.inference(ft, random_sampling=True)
        elif self.infer_mode == 'F':
            _, mus, logvars, zs = self.net(x_flat)
        return zs, mus, logvars       

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, ct = batch['s1']['xt'], batch['s1']['ct']
        batch_size, length, _ = x.shape; ct = torch.squeeze(ct).to(torch.int64)
        x_flat = x.view(-1, self.input_dim)
        # Adjacent matrix
        adj_matrix = F.sigmoid(self.logits)
        # adj_matrix = F.gumbel_softmax(self.logits, tau=0.2, hard=True)[:,:,0]
        # Inference
        if self.infer_mode == 'R':
            ft = self.enc(x_flat)
            ft = ft.view(batch_size, length, -1)
            zs, mus, logvars = self.inference(ft)
            zs_flat = zs.contiguous().view(-1, self.z_dim)
            x_recon = self.dec(zs_flat)
        elif self.infer_mode == 'F':
            x_recon, mus, logvars, zs = self.net(x_flat)
        
        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)
        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag]), torch.ones_like(logvars[:,:self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag]),dim=-1),dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:,:self.lag],dim=-1),dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:,self.lag:]
        residuals, logabsdet = self.transition_prior(zs, adj_matrix)
        # residuals, logabsdet = self.transition_prior(zs)
        sum_log_abs_det_jacobians = 0
        one_hot = F.one_hot(ct, num_classes=self.nclass)
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
        # Nonstationary branch
        es = [ ]
        logabsdet = [ ]
        for c in range(self.nclass):
            es_c, logabsdet_c = self.spline_list[c](residuals.contiguous().view(-1, self.z_dim))
            es.append(es_c)
            logabsdet.append(logabsdet_c)
        es = torch.stack(es, axis=1)
        logabsdet = torch.stack(logabsdet, axis=1)
        mask = one_hot.unsqueeze(1).repeat(1,self.length,1).reshape(-1, self.nclass)
        es = (es * mask.unsqueeze(-1)).sum(1)
        logabsdet = (logabsdet * mask).sum(1)
        es = es.reshape(batch_size, length-self.lag, self.z_dim)
        logabsdet = torch.sum(logabsdet.reshape(batch_size,length-self.lag), dim=1)

        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
        log_pz_laplace = torch.sum(self.base_dist.log_prob(es), dim=1) + sum_log_abs_det_jacobians
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
        kld_laplace = kld_laplace.mean()

        l1_loss = torch.norm(adj_matrix, 1)
        # VAE training
        if optimizer_idx == 0:
            for p in self.discriminator.parameters():
                p.requires_grad = False
            D_z = self.discriminator(residuals.contiguous().view(batch_size, -1))
            tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
            loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + self.sigma * tc_loss + self.l1*l1_loss
            # loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + self.sigma * tc_loss
            self.log("train_elbo_loss", loss)
            self.log("train_recon_loss", recon_loss)
            self.log("train_kld_normal", kld_normal)
            self.log("train_kld_laplace", kld_laplace)
            self.log("train_l1_loss", l1_loss)
            self.log("v_tc_loss", tc_loss)
            return loss

        # Discriminator training
        if optimizer_idx == 1:
            for p in self.discriminator.parameters():
                p.requires_grad = True
            residuals = residuals.detach()
            D_z = self.discriminator(residuals.contiguous().view(batch_size, -1))
            # Permute the other data batch
            ones = torch.ones(batch_size, dtype=torch.long).to(batch['s2']['xt'].device)
            zeros = torch.zeros(batch_size, dtype=torch.long).to(batch['s2']['xt'].device)
            zs_perm, _, _ = self.forward(batch['s2'])
            zs_perm = zs_perm.reshape(batch_size, length, self.z_dim)
            residuals_perm, _ = self.transition_prior(zs_perm, adj_matrix)
            # residuals_perm, _ = self.transition_prior(zs_perm)
            residuals_perm = permute_dims(residuals_perm.contiguous().view(batch_size, -1)).detach()
            D_z_pperm = self.discriminator(residuals_perm)
            D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))            
            self.log("d_tc_loss", D_tc_loss)
            return D_tc_loss
    
    def validation_step(self, batch, batch_idx):
        x, ct = batch['s1']['xt'], batch['s1']['ct']
        batch_size, length, _ = x.shape; ct = torch.squeeze(ct).to(torch.int64)
        x_flat = x.view(-1, self.input_dim)
        # Adjacent matrix
        adj_matrix = F.sigmoid(self.logits)
        # adj_matrix = F.gumbel_softmax(self.logits, tau=0.2, hard=True)[:,:,0]
        # Inference
        if self.infer_mode == 'R':
            ft = self.enc(x_flat)
            ft = ft.view(batch_size, length, -1)
            zs, mus, logvars = self.inference(ft)
            zs_flat = zs.contiguous().view(-1, self.z_dim)
            x_recon = self.dec(zs_flat)
        elif self.infer_mode == 'F':
            x_recon, mus, logvars, zs = self.net(x_flat)
        
        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)
        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag]), torch.ones_like(logvars[:,:self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag]),dim=-1),dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:,:self.lag],dim=-1),dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:,self.lag:]
        residuals, logabsdet = self.transition_prior(zs, adj_matrix)
        # residuals, logabsdet = self.transition_prior(zs)
        sum_log_abs_det_jacobians = 0
        one_hot = F.one_hot(ct, num_classes=self.nclass)
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
        # Nonstationary branch
        es = [ ]
        logabsdet = [ ]
        for c in range(self.nclass):
            es_c, logabsdet_c = self.spline_list[c](residuals.contiguous().view(-1, self.z_dim))
            es.append(es_c)
            logabsdet.append(logabsdet_c)
        es = torch.stack(es, axis=1)
        logabsdet = torch.stack(logabsdet, axis=1)
        mask = one_hot.unsqueeze(1).repeat(1,self.length,1).reshape(-1, self.nclass)
        es = (es * mask.unsqueeze(-1)).sum(1)
        logabsdet = (logabsdet * mask).sum(1)
        es = es.reshape(batch_size, length-self.lag, self.z_dim)
        logabsdet = torch.sum(logabsdet.reshape(batch_size,length-self.lag), dim=1)
        
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
        log_pz_laplace = torch.sum(self.base_dist.log_prob(es), dim=1) + sum_log_abs_det_jacobians
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
        kld_laplace = kld_laplace.mean()
        l1_loss = torch.norm(adj_matrix, 1)
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + self.l1 * l1_loss
        # loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace
        # Compute Mean Correlation Coefficient (MCC)
        # zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()
        # zt_true = batch['s1']["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        # mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        # self.log("val_mcc", mcc) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_normal", kld_normal)
        self.log("val_kld_laplace", kld_laplace)

        return loss
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    def reconstruct(self, batch):
        zs, mus, logvars = self.forward(batch)
        zs_flat = zs.contiguous().view(-1, self.z_dim)
        x_recon = self.dec(zs_flat)
        x_recon = x_recon.view(batch_size, self.length, self.input_dim)       
        return x_recon

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        opt_d = torch.optim.SGD(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=self.lr/2)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [opt_v, opt_d], []

    # def on_after_backward(self):
    #     # Hack trick to skip nan/inf gradient
    #     valid_gradients = True
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
    #             if not valid_gradients:
    #                 break

    #     if not valid_gradients:
    #         print('detected inf or nan values in gradients. not updating model parameters')
    #         self.zero_grad()
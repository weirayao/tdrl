"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from .components.beta import BetaVAE_MLP
from .components.transition import (MBDTransitionPrior, 
                                    NPChangeTransitionPrior)
from .components.mlp import MLPEncoder, MLPDecoder, Inference
from .metrics.correlation import compute_mcc

import ipdb as pdb

class TimeVaryingProcess(pl.LightningModule):

    def __init__(
        self, 
        input_dim,
        length,
        z_dim, 
        lag,
        nclass,
        hidden_dim=128,
        embedding_dim=8,
        trans_prior='NP',
        lr=1e-4,
        infer_mode='F',
        beta=0.0025,
        gamma=0.0075,
        decoder_dist='gaussian',
        correlation='Pearson'):
        '''Nonlinear ICA for time-varing causal processes'''
        super().__init__()
        # Transition prior must be L (Linear), NP (Nonparametric)
        assert trans_prior in ('L', 'NP')
        self.z_dim = z_dim
        self.lag = lag
        self.input_dim = input_dim
        self.lr = lr
        self.lag = lag
        self.length = length
        self.beta = beta
        self.gamma = gamma
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.infer_mode = infer_mode
        # Domain embeddings (dynamics)
        self.embed_func = nn.Embedding(nclass, embedding_dim)
        # Recurrent/Factorized inference
        if infer_mode == 'R':
            self.enc = MLPEncoder(latent_size=z_dim, 
                                  num_layers=3, 
                                  hidden_dim=hidden_dim)

            self.dec = MLPDecoder(latent_size=z_dim, 
                                  num_layers=2,
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
                                                       bias=False)
        elif trans_prior == 'NP':
            self.transition_prior = NPChangeTransitionPrior(lags=lag, 
                                                            latent_size=z_dim,
                                                            embedding_dim=embedding_dim,
                                                            num_layers=3, 
                                                            hidden_dim=hidden_dim)
                                                            
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

    @property
    def base_dist(self):
        # Noise density function
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
            zs.append(torch.ones((batch_size, self.z_dim), device=output.device))

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
        x, y, c = batch['xt'], batch['yt'], batch['ct']
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        if self.infer_mode == 'R':
            ft = self.enc(x_flat)
            ft = ft.view(batch_size, length, -1)
            zs, mus, logvars = self.inference(ft, random_sampling=True)
        elif self.infer_mode == 'F':
            _, mus, logvars, zs = self.net(x_flat)
        return zs, mus, logvars       

    def training_step(self, batch, batch_idx):
        x, y, c = batch['xt'], batch['yt'], batch['ct']
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        sum_log_abs_det_jacobians = 0
        x_flat = x.view(-1, self.input_dim)
        embeddings = self.embed_func(c)
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
        residuals, logabsdet = self.transition_prior(zs, embeddings)
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
        log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + sum_log_abs_det_jacobians
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
        kld_laplace = kld_laplace.mean()

        # VAE training
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", kld_normal)
        self.log("train_kld_laplace", kld_laplace)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, c = batch['xt'], batch['yt'], batch['ct']
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape
        sum_log_abs_det_jacobians = 0
        x_flat = x.view(-1, self.input_dim)
        embeddings = self.embed_func(c)
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
        residuals, logabsdet = self.transition_prior(zs, embeddings)
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + logabsdet
        log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + sum_log_abs_det_jacobians
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
        kld_laplace = kld_laplace.mean()

        # VAE training
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace

        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = batch["yt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc) 
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

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []
"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from .components.beta import BetaVAE_CNN
from .components.transition import (MBDTransitionPrior, 
                                    NPChangeTransitionPrior)
from .components.mlp import MLPEncoder, MLPDecoder, Inference
from .components.flow import ComponentWiseCondSpline
from .metrics.correlation import compute_mcc

import ipdb as pdb

class ModularShifts(pl.LightningModule):

    def __init__(
        self, 
        input_dim,
        length,
        obs_dim,
        dyn_dim, 
        lag,
        nclass,
        hidden_dim=128,
        dyn_embedding_dim=3,
        obs_embedding_dim=0,
        trans_prior='NP',
        lr=1e-4,
        infer_mode='F',
        beta=0.0025,
        gamma=0.0075,
        sigma=0.0025,
        decoder_dist='gaussian',
        correlation='Pearson'):
        '''Nonlinear ICA for general causal processes with modualar distribution shifts'''
        super().__init__()
        # Transition prior must be L (Linear), NP (Nonparametric)
        assert trans_prior in ('L', 'NP')
        self.obs_dim = obs_dim
        self.dyn_dim = dyn_dim
        self.obs_embedding_dim = obs_embedding_dim
        self.dyn_embedding_dim = dyn_embedding_dim
        self.z_dim = obs_dim + dyn_dim
        self.lag = lag
        self.input_dim = input_dim
        self.lr = lr
        self.lag = lag
        self.length = length
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.infer_mode = infer_mode
        # Domain embeddings (dynamics)
        if dyn_embedding_dim > 0:
            self.dyn_embed_func = nn.Embedding(nclass, dyn_embedding_dim)
        if obs_embedding_dim > 0:
            self.obs_embed_func = nn.Embedding(nclass, obs_embedding_dim)

        # Factorized inference
        self.net = BetaVAE_CNN(nc=1, 
                               z_dim=self.z_dim,
                               hidden_dim=hidden_dim)
        transition_priors = [ ]
        # Initialize transition prior
        for i in range(2):
            if trans_prior == 'L':
                transition_prior = MBDTransitionPrior(lags=lag, 
                                                      latent_size=self.dyn_dim, 
                                                      bias=False)
            elif trans_prior == 'NP':
                transition_prior = NPChangeTransitionPrior(lags=lag, 
                                                           latent_size=self.dyn_dim,
                                                           embedding_dim=dyn_embedding_dim,
                                                           num_layers=4, 
                                                           hidden_dim=hidden_dim)
            transition_priors.append(transition_prior)
        self.transition_priors = nn.ModuleList(transition_priors)

        # base distribution for calculation of log prob under the model
        if self.dyn_dim > 0:
            self.register_buffer('dyn_base_dist_mean', torch.zeros(self.dyn_dim))
            self.register_buffer('dyn_base_dist_var', torch.eye(self.dyn_dim))
        if self.obs_dim > 0:
            self.register_buffer('obs_base_dist_mean', torch.zeros(self.obs_dim))
            self.register_buffer('obs_base_dist_var', torch.eye(self.obs_dim))

    @property
    def dyn_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.dyn_base_dist_mean, self.dyn_base_dist_var)

    @property
    def obs_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.obs_base_dist_mean, self.obs_base_dist_var)
    
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
        _, mus, logvars, zs = self.net(x_flat)
        return zs, mus, logvars       

    def training_step(self, batch, batch_idx):
        x, y, a, c = batch['xt'], batch['yt'], batch['at'], batch['ct']
        c = torch.squeeze(c).to(torch.int64)
        a = torch.squeeze(a).to(torch.int64)
        batch_size, length, nc, h, w = x.shape
        x_flat = x.view(-1, nc, h, w)
        if self.dyn_dim > 0:
            dyn_embeddings = self.dyn_embed_func(c)
        if self.obs_dim > 0:
            obs_embeddings = self.obs_embed_func(c)
            obs_embeddings = obs_embeddings.reshape(batch_size,1,self.obs_embedding_dim).repeat(1,length,1)
        # Inference
        x_recon, mus, logvars, zs = self.net(x_flat)
        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, nc, h, w)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        ### Dynamics parts ###
        # Past KLD <=> N(0,1) #
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag,:self.dyn_dim]), torch.ones_like(logvars[:,:self.lag, :self.dyn_dim]))
        log_pz_past = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag,:self.dyn_dim]),dim=-1),dim=-1)
        log_qz_past = torch.sum(torch.sum(log_qz[:,:self.lag,:self.dyn_dim],dim=-1),dim=-1)
        past_kld_dyn = log_qz_past - log_pz_past
        past_kld_dyn = past_kld_dyn.mean()
        # Future KLD #
        log_qz_future = log_qz[:,self.lag:]
        residuals = [ ]
        logabsdet = [ ]
        # Two action branches
        for a_idx in range(2):
            residuals_action, logabsdet_action = self.transition_priors[a_idx](zs[:,:,:self.dyn_dim], dyn_embeddings)
            residuals.append(residuals_action)
            logabsdet.append(logabsdet_action)
        mask = F.one_hot(a, num_classes=2)
        residuals = torch.stack(residuals, axis=1)
        logabsdet = torch.stack(logabsdet, axis=1)
        residuals = (residuals * mask[:,:,None,None]).sum(1)
        logabsdet = (logabsdet * mask).sum(1)
        log_pz_future = torch.sum(self.dyn_base_dist.log_prob(residuals), dim=1) + logabsdet
        future_kld_dyn = (torch.sum(torch.sum(log_qz_future,dim=-1),dim=-1) - log_pz_future) / (length-self.lag)
        future_kld_dyn = future_kld_dyn.mean()

        ### Observation parts ###
        if self.obs_dim > 0:
            p_dist_obs = D.Normal(obs_embeddings[:,:,0].reshape(batch_size, length, 1), 
                                torch.exp(obs_embeddings[:,:,1].reshape(batch_size, length, 1) / 2) )
            log_pz_obs = torch.sum(torch.sum(p_dist_obs.log_prob(zs[:,:,self.dyn_dim:]), dim=1),dim=-1)
            log_qz_obs = torch.sum(torch.sum(log_qz[:,:self.lag,self.dyn_dim:],dim=-1),dim=-1)
            kld_obs = log_qz_obs - log_pz_obs
            kld_obs = kld_obs.mean()
        else:
            kld_obs = 0      

        # VAE training
        loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", past_kld_dyn)
        self.log("train_kld_dynamics", future_kld_dyn)
        self.log("train_kld_observation", kld_obs)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, a, c = batch['xt'], batch['yt'], batch['at'], batch['ct']
        c = torch.squeeze(c).to(torch.int64)
        a = torch.squeeze(a).to(torch.int64)
        batch_size, length, nc, h, w = x.shape
        x_flat = x.view(-1, nc, h, w)
        if self.dyn_dim > 0:
            dyn_embeddings = self.dyn_embed_func(c)
        if self.obs_dim > 0:
            obs_embeddings = self.obs_embed_func(c)
            obs_embeddings = obs_embeddings.reshape(batch_size,1,self.obs_embedding_dim).repeat(1,length,1)
        # Inference
        x_recon, mus, logvars, zs = self.net(x_flat)
        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, nc, h, w)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        # VAE ELBO loss: recon_loss + kld_loss
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon[:,self.lag:], self.decoder_dist))/(length-self.lag)
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        ### Dynamics parts ###
        # Past KLD <=> N(0,1) #
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag,:self.dyn_dim]), torch.ones_like(logvars[:,:self.lag, :self.dyn_dim]))
        log_pz_past = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag,:self.dyn_dim]),dim=-1),dim=-1)
        log_qz_past = torch.sum(torch.sum(log_qz[:,:self.lag,:self.dyn_dim],dim=-1),dim=-1)
        past_kld_dyn = log_qz_past - log_pz_past
        past_kld_dyn = past_kld_dyn.mean()
        # Future KLD #
        log_qz_future = log_qz[:,self.lag:]
        residuals = [ ]
        logabsdet = [ ]
        # Two action branches
        for a_idx in range(2):
            residuals_action, logabsdet_action = self.transition_priors[a_idx](zs[:,:,:self.dyn_dim], dyn_embeddings)
            residuals.append(residuals_action)
            logabsdet.append(logabsdet_action)
        mask = F.one_hot(a, num_classes=2)
        residuals = torch.stack(residuals, axis=1)
        logabsdet = torch.stack(logabsdet, axis=1)
        residuals = (residuals * mask[:,:,None,None]).sum(1)
        logabsdet = (logabsdet * mask).sum(1)
        log_pz_future = torch.sum(self.dyn_base_dist.log_prob(residuals), dim=1) + logabsdet
        future_kld_dyn = (torch.sum(torch.sum(log_qz_future,dim=-1),dim=-1) - log_pz_future) / (length-self.lag)
        future_kld_dyn = future_kld_dyn.mean()

        ### Observation parts ###
        if self.obs_dim > 0:
            p_dist_obs = D.Normal(obs_embeddings[:,:,0].reshape(batch_size, length, 1), 
                                torch.exp(obs_embeddings[:,:,1].reshape(batch_size, length, 1) / 2) )
            log_pz_obs = torch.sum(torch.sum(p_dist_obs.log_prob(zs[:,:,self.dyn_dim:]), dim=1),dim=-1)
            log_qz_obs = torch.sum(torch.sum(log_qz[:,:self.lag,self.dyn_dim:],dim=-1),dim=-1)
            kld_obs = log_qz_obs - log_pz_obs
            kld_obs = kld_obs.mean()
        else:
            kld_obs = 0      

        # VAE training
        loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs

        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus[:,0].view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = batch["yt"][:,0].view(-1, 2).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_normal", past_kld_dyn)
        self.log("val_kld_dynamics", future_kld_dyn)
        self.log("val_kld_observation", kld_obs)

        return loss
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []
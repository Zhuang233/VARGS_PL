#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl 

import sys
import os 
from pathlib import Path
import numpy as np 
import math

from einops import rearrange, reduce

from models.archs.gs_decoder import * 
from models.archs.encoders.conv_pointnet import ConvPointnet
from utils import evaluate


class GsModel(pl.LightningModule):

    def __init__(self, specs):
        super().__init__()
        
        self.specs = specs
        model_specs = self.specs["GSModelSpecs"]
        self.hidden_dim = model_specs["hidden_dim"]
        self.latent_dim = model_specs["latent_dim"]
        self.skip_connection = model_specs.get("skip_connection", True)
        self.tanh_act = model_specs.get("tanh_act", False)
        self.pn_hidden = model_specs.get("pn_hidden_dim", self.latent_dim)

        self.pointnet = ConvPointnet(c_dim=self.latent_dim, dim=59, hidden_dim=self.pn_hidden, plane_resolution=64)
        
        self.model = GSDecoder(latent_size=self.latent_dim, hidden_dim=self.hidden_dim, skip_connection=self.skip_connection, tanh_act=self.tanh_act)

        self.occ_model = OccDecoder(latent_size=self.latent_dim, hidden_dim=self.hidden_dim, skip_connection=self.skip_connection, tanh_act=self.tanh_act)

        self.color_model = ColorDecoder(latent_size=self.latent_dim, hidden_dim=self.hidden_dim, skip_connection=self.skip_connection, tanh_act=self.tanh_act)
        
        self.occ_model.train()
        self.color_model.train()

            
    def forward(self, pc, gs):
        shape_features = self.pointnet(pc, gs)

        return self.model(gs, shape_features).squeeze()

    def forward_with_plane_features(self, plane_features, gs):
        gs = gs[:,:,:3]
        point_features = self.pointnet.forward_with_plane_features(plane_features, gs) # point_features: B, N, D
        pred_color = self.color_model( torch.cat((gs, point_features),dim=-1))
        pred_gs = self.model( torch.cat((gs, point_features),dim=-1))
        return pred_color, pred_gs # [B, num_points] 
    

    def forward_with_plane_features_occ(self, plane_features, gs):
        point_features = self.pointnet.forward_with_plane_features(plane_features, gs) # point_features: B, N, D
        pred_occ = self.occ_model( torch.cat((gs, point_features),dim=-1) )  
        return pred_occ # [B, num_points] 
    
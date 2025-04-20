import torch
import torch.utils.data 
from torch.nn import functional as F
from lightning import LightningModule
import time
# add paths in model/__init__.py for new models
from .betavae import BetaVAE
from .gaussian_model import GsModel
from einops import reduce
from omegaconf import OmegaConf

class DiffGSAutoEncoder(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = OmegaConf.create(config)

        self.gs_model = GsModel(config=config) 

        feature_dim = self.config.GSModel.latent_dim
        modulation_dim = feature_dim*3
        latent_std = self.config.latent_std # default 0.25
        hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
        self.vae_model = BetaVAE(in_channels=feature_dim*3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)

    def training_step(self, batch, idx):
        taxonomy_id, model_id, data, centroid, scale_factor, scale_c, scale_m, occ, occ_xyz = batch
        # occ_xyz = data[]
        # occ = x['occ']
        # gt = x['gt_gaussian']
        # gs = x['gaussians']
        # gaussian_xyz = x['gaussian_xyz']
        gs=data

        plane_features = self.gs_model.pointnet.get_plane_features(gs)
        original_features = torch.cat(plane_features, dim=1)
        out = self.vae_model(original_features)
        reconstructed_plane_feature, latent = out[0], out[-1]

        pred_color, pred_gs = self.gs_model.forward_with_plane_features(reconstructed_plane_feature, gaussian_xyz)
        pred_occ = self.gs_model.forward_with_plane_features_occ(reconstructed_plane_feature, occ_xyz)
        
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.config["kld_weight"] )
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        color_loss = F.l1_loss(pred_color.squeeze()[:,:,0:48], gt.squeeze()[:,:,0:48], reduction='none')
        color_loss = reduce(color_loss, 'b ... -> b (...)', 'mean').mean()

        scale_loss = F.l1_loss(pred_gs.squeeze()[:,:,0:3], gt.squeeze()[:,:,49:52], reduction='none')
        scale_loss = reduce(scale_loss, 'b ... -> b (...)', 'mean').mean()
        rotation_loss = F.l1_loss(pred_gs.squeeze()[:,:,3:7], gt.squeeze()[:,:,52:56], reduction='none')
        rotation_loss = reduce(rotation_loss, 'b ... -> b (...)', 'mean').mean()

        occ_loss = F.l1_loss(pred_occ.squeeze(), occ.squeeze(), reduction='none')
        occ_loss = reduce(occ_loss, 'b ... -> b (...)', 'mean').mean()

        loss = color_loss + vae_loss + occ_loss + scale_loss + rotation_loss

        loss_dict =  {"loss": loss, "color": color_loss, "vae": vae_loss, "occ": occ_loss, "scale": scale_loss, "rotation": rotation_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss

        

    def configure_optimizers(self):
        params_list = [{ 'params': self.parameters(), 'lr':self.config['sdf_lr'] }]
        return torch.optim.Adam(params_list)
    
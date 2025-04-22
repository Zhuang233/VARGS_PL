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
from utils import pointcloud
from utils import convert

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
        taxonomy_id, model_id, gs, centroid, scale_factor, scale_c, scale_m, occ, occ_xyz = batch
        gt = gs[...,3:]
        # occ_xyz = data[]
        # occ = x['occ']
        # gt = x['gt_gaussian']
        # gs = x['gaussians']
        # gaussian_xyz = x['gaussian_xyz']
        gaussian_xyz = gs[...,:3]

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

        color_loss = F.l1_loss(pred_color.squeeze()[:,:,0:3], gt.squeeze()[:,:,8:11], reduction='none')
        color_loss = reduce(color_loss, 'b ... -> b (...)', 'mean').mean()

        scale_loss = F.l1_loss(pred_gs.squeeze()[:,:,0:3], gt.squeeze()[:,:,1:4], reduction='none')
        scale_loss = reduce(scale_loss, 'b ... -> b (...)', 'mean').mean()
        rotation_loss = F.l1_loss(pred_gs.squeeze()[:,:,3:7], gt.squeeze()[:,:,4:8], reduction='none')
        rotation_loss = reduce(rotation_loss, 'b ... -> b (...)', 'mean').mean()

        occ_loss = F.l1_loss(pred_occ.squeeze(), occ.squeeze(), reduction='none')
        occ_loss = reduce(occ_loss, 'b ... -> b (...)', 'mean').mean()

        loss = color_loss + vae_loss + occ_loss + scale_loss + rotation_loss

        loss_dict =  {"loss": loss, "color": color_loss, "vae": vae_loss, "occ": occ_loss, "scale": scale_loss, "rotation": rotation_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss
    
    def validation_step(self, batch, idx):
        pass
        # taxonomy_id, model_id, gs, centroid, scale_factor, scale_c, scale_m, occ, occ_xyz = batch
        # plane_features = self.gs_model.pointnet.get_plane_features(gs)
        # original_features = torch.cat(plane_features, dim=1)
        # plane_features = self.vae_model(original_features)

        # for i in range(len(plane_features)):
        #     plane_feature = plane_features[i].unsqueeze(0)
        #     with torch.no_grad():
        #         print('create points fast')
        #         new_pc = pointcloud.create_pc_fast(self.gs_model, plane_feature, N=1024, max_batch=2**20, from_plane_features=True)
        #     new_pc_optimizer = pointcloud.pc_optimizer(self.gs_model, plane_feature.detach(), new_pc.clone().detach().cuda())            
        #     with torch.no_grad():
        #         new_pc = torch.cat([new_pc, new_pc_optimizer], dim=1)
        #         new_pc = new_pc.reshape(1, -1, 3).float()
        #         pred_color, pred_gs = self.gs_model.forward_with_plane_features(plane_feature, new_pc)
        #         gaussian = torch.zeros(new_pc.shape[1], 59).cpu()
        #         gaussian[:,:3] = new_pc[0]
        #         gaussian[:,3:6] = pred_color[0]
        #         gaussian[:,6] = 2.9444
        #         gaussian[:,7:10] = 0.9 * torch.log(pred_gs[0,:,0:3])
        #         gaussian[:,10:14] = pred_gs[0,:,3:7]
        #         convert.convert(gaussian.detach().cpu().numpy(), f"./generate/gaussian_{idx}.ply")
        #         idx = idx + 1

        # reconstructed_plane_feature, latent = out[0]
        # gaussian_xyz = gs[...,:3]
        # pred_color, pred_gs = self.gs_model.forward_with_plane_features(reconstructed_plane_feature, gaussian_xyz)
        # pred_occ = self.gs_model.forward_with_plane_features_occ(reconstructed_plane_feature, occ_xyz)

        # pointcloud.create_pc_fast(self.gs_model, )


        

    def configure_optimizers(self):
        params_list = [{ 'params': self.parameters(), 'lr':self.config['sdf_lr'] }]
        return torch.optim.Adam(params_list)
    
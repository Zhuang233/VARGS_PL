import torch.nn as nn
import torch
from math import pi
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import OmegaConf
from chamferdist import ChamferDistance
from utils.logger import get_logger
from lightning.pytorch.utilities import grad_norm
from .dvae import DiscreteVAE
import math
from collections import OrderedDict

## diffgs
# from autoencoder import BetaVAE
# from conv_pointnet import ConvPointnet


def fourier_encode(x, max_freqs, num_bands=4):
    """
    Fourier encode the input with independent frequency ranges for each dimension.

    Args:
        x: Input tensor of shape [batch_size, n, d], where `n` is the number of points
           and `d` is the number of dimensions.
        max_freqs: List or tensor of length `d`, specifying the maximum frequency for each dimension.
        num_bands: Number of frequency bands to use per dimension.

    Returns:
        Tensor of shape [batch_size, n, d * (2 * num_bands).
    """
    assert x.ndim == 3, "Input tensor x must have shape [batch_size, n, d]."
    assert len(max_freqs) == x.shape[2], "max_freqs must have the same length as the number of dimensions in x."

    # Extract batch size and input shape
    batch_size, n, d = x.shape

    # Prepare frequency scales for each dimension
    device, dtype = x.device, x.dtype
    max_freqs = torch.tensor(max_freqs, device=device, dtype=dtype)
    scales = torch.stack([torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype) for max_freq in max_freqs], dim=0)
    scales = scales.view(1, 1, d, num_bands)  # Shape: [1, 1, d, num_bands]

    # Expand input for frequency encoding
    x = x.unsqueeze(-1)  # Shape: [batch_size, n, d, 1]

    # Compute Fourier features
    x = x * scales * pi  # Shape: [batch_size, n, d, num_bands]
    fourier_features = torch.cat([x.sin(), x.cos()], dim=-1)  # Shape: [batch_size, n, d, 2 * num_bands]

    # Flatten features across dimensions
    fourier_features = fourier_features.view(batch_size, n, -1)  # Shape: [batch_size, n, d * (2 * num_bands)]

    return fourier_features

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, num_queries):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim))
        self.attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=1, batch_first=True)

    def forward(self, features):
        # Query features through cross-attention
        queries = self.queries.unsqueeze(0).expand(features.size(0), -1, -1)  # Batch the queries
        attended, _ = self.attention(queries, features, features)
        return attended

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=input_dim, nhead=1) for _ in range(num_layers)])

    def forward(self, features):
        # Pass through self-attention layers
        for layer in self.layers:
            features = layer(features)

        return features

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # 定义 5 层的 MLP，使用 ReLU 激活函数
        self.fc1 = nn.Linear(96, 43)    
        self.fc2 = nn.Linear(43, 24)   
        self.fc3 = nn.Linear(24, 12)
        self.fc4 = nn.Linear(12,1)  
        self.s = nn.Sigmoid()
        
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播过程
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x) 
        return self.s(x)
    
class MLP_test(nn.Module):
    def __init__(self):
        super(MLP_test, self).__init__()
        
        # 定义 5 层的 MLP，使用 ReLU 激活函数
        self.fc1 = nn.Linear(768, 768)    
        self.fc2 = nn.Linear(768, 768)   
        self.fc3 = nn.Linear(768, 384)
        self.fc4 = nn.Linear(384,48)  
        self.fc5 = nn.Linear(48,24)  
        self.fc6 = nn.Linear(24,3)  
        self.s = nn.Sigmoid()
        
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播过程
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x) 
        return self.s(x) *2 -1
    
class Simple_Net(nn.Module):
    def __init__(self, num_points=1024, latent_dim=128):
        super(Simple_Net, self).__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim

        # Encoder: 3D点 (1024,3) -> 全局特征向量 (128,)
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)  # 128 维全局特征
        )

        # Decoder: 全局特征 (128,) -> 重建点云 (1024,3)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3)  # 1024 个点，每个点 3 维
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Encoder: 对每个点进行 MLP 变换
        x = self.encoder(x)  # (B, 1024, 128)
        x = torch.max(x, dim=1)[0]  # Max Pooling，得到全局特征 (B, 128)

        # Decoder: 生成点云
        x = self.decoder(x)  # (B, 1024*3)
        x = x.view(batch_size, self.num_points, 3)  # 变回 (B, 1024, 3)
        return x

class TriPlaneFeatureExtractor:
    def __init__(self):
        pass

    @staticmethod
    def interpolate(feature, points):
        """
        对输入的 3D 点进行三平面特征插值。
        :param feature: 形状为 (bs, 3, c, h, w) 的三平面特征。
        :param points: 形状为 (bs, num, 3) 的采样点，值域应为 [0, 1]。
        :return: 形状为 (bs, num, c) 的特征向量。
        """
        bs, _, c, h, w = feature.shape
        num = points.shape[1]

        # 提取三个平面的特征
        xy_plane, xz_plane, yz_plane = feature[:, 0], feature[:, 1], feature[:, 2]  # (bs, c, h, w)

        x, y, z = points.split(1, dim=-1)  # (bs, num, 1)

        # 对每个平面进行双线性插值
        xy_features = F.grid_sample(
            xy_plane, TriPlaneFeatureExtractor._normalize_grid(x, y),
            mode='bilinear', align_corners=True
        )  # (bs, c, num, 1)
        xz_features = F.grid_sample(
            xz_plane, TriPlaneFeatureExtractor._normalize_grid(x, z),
            mode='bilinear', align_corners=True
        )  # (bs, c, num, 1)
        yz_features = F.grid_sample(
            yz_plane, TriPlaneFeatureExtractor._normalize_grid(y, z),
            mode='bilinear', align_corners=True
        )  # (bs, c, num, 1)

        # 将三个平面的特征相加
        out_feature = (xy_features + xz_features + yz_features).squeeze(-1)  # (bs, c, num)
        return out_feature.permute(0, 2, 1)  # 调整为 (bs, num, c)

    @staticmethod
    def _normalize_grid(coord1, coord2):
        """
        规范化网格坐标，将 [0, resolution-1] 映射到 [-1, 1]。
        """
        grid = torch.stack((coord1, coord2), dim=-1)  # (bs, num, 1, 2)
        grid = 2 * grid - 1  # 将 [0, 1] 范围的坐标直接映射到 [-1, 1]
        return grid  # (bs, num, 1, 2)


class VAT(LightningModule):
    def __init__(self, config):
        super(VAT, self).__init__()
        self.config = OmegaConf.create(config)
        # self.linear = nn.Linear(in_features= self.config.feature_num_per_point*(self.config.fourier_encode.num_bands*2 + 1), out_features=768)
        # self.linear = nn.Linear(in_features= self.config.feature_num_per_point, out_features=768 * 2)
        # self.cross_attention = CrossAttentionLayer(query_dim = 768, num_queries = 3072)
        # self.tokens = nn.Parameter(torch.randn(1024, 768))
        # self.self_attention = SelfAttention(input_dim = 768, num_layers = 4)
        # self.linear_mu = nn.Linear(in_features=768, out_features=16)
        # self.linear_sigma = nn.Linear(in_features=768, out_features=16)

        # self.linear2 = nn.Linear(in_features=16, out_features=768)
        # self.cross_attention2 = CrossAttentionLayer(query_dim = 768, num_queries = 3072)
        # self.self_attention2 = SelfAttention(input_dim = 768, num_layers = 12)

        # Mipmap convolutional layers (Upsampling)
        # self.conv_r = nn.ConvTranspose2d(768*3, 384*3, kernel_size=4, stride=2, padding=1)  # From 32x32 to 64x64
        # self.conv_r2 = nn.ConvTranspose2d(384*3, 192*3, kernel_size=4, stride=2, padding=1)  # From 64x64 to 128x128
        # self.conv_r4 = nn.ConvTranspose2d(192*3, 96*3, kernel_size=4, stride=2, padding=1)  # From 128x128 to 256x256

        # self.TriPlaneFeatureExtractor = TriPlaneFeatureExtractor()
        # self.mlp = MLP()
        self.mlp_test = MLP_test()
        self.ChamferDistance = ChamferDistance()
        # self.test_net = Simple_Net()
        # state_dict = torch.load(self.config.dvae_ckpt)
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.base_model.items():
        #     # 去除 'module.' 前缀
        #     name = k.replace('module.', '') if k.startswith('module.') else k
        #     new_state_dict[name] = v
        # self.dvae.load_state_dict(new_state_dict)


        ## diffgs_test
        # feature_dim = 256
        # modulation_dim = feature_dim*3
        # hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
        # latent_std = 0.25
        # self.vae_test = BetaVAE(in_channels=feature_dim*3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)

        # self.pointnet = ConvPointnet(c_dim=256, dim=59, hidden_dim=128, plane_resolution=64)

        ## debug 参数
        # self.logger_txt = None # ⚠️ logger 只有在 trainer 初始化后才存在！
        # self.example_input_array = torch.Tensor(3, config.npoints, config.feature_num_per_point)

    def training_step(self, batch, batch_idx):
        taxonomy_id, model_id, data, centroid, scale_factor, scale_c, scale_m = batch
        points = data  # b n 3
        # diffgs
        # plane_features = self.pointnet.get_plane_features(data)
        # original_features = torch.cat(plane_features, dim=1)
        # out = self.vae_test(original_features)
        # reconstructed_plane_feature, latent = out[0], out[-1]
        points_rebuild = self(points)
        loss = self.ChamferDistance(points, points_rebuild, bidirectional=True) / self.config.npoints
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        taxonomy_id, model_id, data, centroid, scale_factor, scale_c, scale_m = batch
        points = data
        points_rebuild = self(points)
        val_loss = self.ChamferDistance(points, points_rebuild, bidirectional=True) / self.config.npoints
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return points_rebuild, taxonomy_id, model_id 
    
    def test_step(self, *args, **kwargs):
        pass
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
    
    def forward(self, x):
        # x_re = self.mlp(x)
        # return x_re
        bs = x.shape[0]
        # x shape: [batch_size, npoints, 14]
        # max_freqs = [
        #     15.0, 15.0, 15.0,  # pos
        #     10.0,              # opacity
        #     10.0, 10.0, 10.0,  # scale
        #     20.0, 20.0, 20.0, 20.0,  # rotation
        #     8.0, 8.0, 8.0      # sh
        # ]

        max_freqs = [1120,1120,1120]
        # 傅里叶编码，拓展特征
        x_fourier_features = fourier_encode(x, max_freqs, num_bands=self.config.fourier_encode.num_bands)
        # x = torch.cat([x, x_fourier_features], dim=-1)
        x = x_fourier_features

        # x shape: [batch_size, npoints, 14*64+14]
        # x = self.linear(x)
        # x shape: [batch_size, npoints, 768]
        # x = self.cross_attention(x)
        # x shape: [batch_size, 3072, 768]

        # Concatenate tokens with features
        # tokens = self.tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        # x = torch.cat([tokens, x], dim=1)

        # x = self.self_attention(x)
        # x = x[:, :self.tokens.size(0), :]

        x_re = self.mlp_test(x)
        # #1*96*32 -> 1*1024*3
        x_re = x_re.reshape(bs,self.config.npoints,3)
        return x_re
    
        # x shape: [batch_size, 1024, 768]
        mu = self.linear_mu(x)
        sigma = self.linear_sigma(x)
        # x shape: [batch_size, 1024, 16]

        sampled_latents = self.sample_from_gaussian(mu, sigma)
        # sampled_latents shape: [batch_size, 1024, 8]

        # TODO: VVQ 




        x = self.linear2(sampled_latents)
        # x shape: [batch_size, 1024, 768]
        x = self.cross_attention2(x)
        # x shape: [batch_size, 3072, 768]
        x = self.self_attention2(x)
        # x shape: [batch_size, 3*32*32, 768]
        trifeat_num = x.shape[-1]*3
        Triplane_feature = x.view(-1, 3, 32, 32, 768).permute(0, 1, 4, 2, 3).reshape(-1, trifeat_num, 32, 32)

        # Create mipmaps by upsampling the triplane features
        mip_r = self.conv_r(Triplane_feature)  # From 32x32 to 64x64
        mip_r2 = self.conv_r2(mip_r)  # From 64x64 to 128x128
        mip_r4 = self.conv_r4(mip_r2)  # From 128x128 to 256x256

        # mipmaps to GS 
        # 在cube均匀采样点上进行三线性插值，得到mipr4的3D特征,经过mlp得到最终的GS特征
        #mip_r4(bs, feature_num*3, 256, 256)

        # 采样随机半径（开立方确保均匀性）
        radii = torch.pow(torch.rand(bs, 20000, 1), 1/3)
        # 采样随机方位角 phi (0, 2pi) 和仰角 theta (0, pi)
        theta = torch.acos(2 * torch.rand(bs, 20000, 1) - 1)  # 0 到 pi
        phi = 2 * torch.pi * torch.rand(bs, 20000, 1)  # 0 到 2pi

        # 转换为笛卡尔坐标
        x_sample = radii * torch.sin(theta) * torch.cos(phi)
        y_sample = radii * torch.sin(theta) * torch.sin(phi)
        z_sample = radii * torch.cos(theta)

        # # 拼接得到最终采样点
        sample_points = torch.cat([x_sample, y_sample, z_sample], dim=-1).to(x.device)

        sample_points_feature = self.TriPlaneFeatureExtractor.interpolate(mip_r4.view(bs, 3, -1, 256, 256) , sample_points)
        # sample_points_feature(bs, 20000, feature_num)
        Occupancy = self.mlp(sample_points_feature)
        # sample_points_feature(bs, 20000, 1)

        Rebuild_Gaussian = self.get_top_k_points_by_confidence(Rebuild_Gaussian=sample_points, confidence=Occupancy , k=self.config.npoints)

        return Rebuild_Gaussian, mu, sigma

    def loss(self, Rebuild_Gaussian, target):
        """
        计算两个高斯点集之间的损失。损失包括两部分：
        1. XYZ的倒角距离（欧氏距离）
        2. 其他高斯参数的L1损失与最近点的匹配
        
        Rebuild_Gaussian, target: 张量，形状为 (bs, point_num, gs_para)
            gs_para 的长度为 14，分别包括位置 (xyz 3)，不透明度 (1)，尺度 (scale 3)，旋转 (rotation 4)，sh 参数 (sh 3)
        """
        # 提取参数
        xyz1 = Rebuild_Gaussian[:, :, :3]  # (bs, point_num, 3)
        # opacity1 = Rebuild_Gaussian[:, :, 3:4]  # (bs, point_num, 1)
        # scale1 = Rebuild_Gaussian[:, :, 4:7]  # (bs, point_num, 3)
        # rotation1 = Rebuild_Gaussian[:, :, 7:11]  # (bs, point_num, 4)
        # sh1 = Rebuild_Gaussian[:, :, 11:]  # (bs, point_num, 3)
        
        xyz2 = target[:, :, :3]  # (bs, point_num, 3)
        # opacity2 = target[:, :, 3:4]  # (bs, point_num, 1)
        # scale2 = target[:, :, 4:7]  # (bs, point_num, 3)
        # rotation2 = target[:, :, 7:11]  # (bs, point_num, 4)
        # sh2 = target[:, :, 11:]  # (bs, point_num, 3)
        
        # 1. 计算 XYZ 的倒角距离（欧氏距离）
        dist = torch.norm(xyz1.unsqueeze(2) - xyz2.unsqueeze(1), p=2, dim=-1)  # (bs, point_num1, point_num2, 3)
        # 找到最近点的距离
        min_dist, min_indices = dist.min(dim=-1)  # (bs, point_num1), 索引为最近点的索引
        # print(min_dist)
        
        # 2. 计算其他参数的 L1 损失
        # 使用找到的最近点的索引，计算 opacity, scale, rotation, sh 参数的 L1 损失
        # opacity_loss = F.l1_loss(opacity1, opacity2.gather(1, min_indices.unsqueeze(-1).expand(-1, -1, 1)), reduction='none').sum(-1)
        # scale_loss = F.l1_loss(scale1, scale2.gather(1, min_indices.unsqueeze(-1).expand(-1, -1, 3)), reduction='none').sum(-1)
        # rotation_loss = F.l1_loss(rotation1, rotation2.gather(1, min_indices.unsqueeze(-1).expand(-1, -1, 4)), reduction='none').sum(-1)
        # sh_loss = F.l1_loss(sh1, sh2.gather(1, min_indices.unsqueeze(-1).expand(-1, -1, 3)), reduction='none').sum(-1)

        # 3. 计算总的损失
        # print(min_dist.sum(), opacity_loss.sum(), scale_loss.sum(), rotation_loss.sum(), sh_loss.sum())
        return min_dist.sum() /4096
        total_loss = min_dist.sum() + opacity_loss.sum() + scale_loss.sum() + rotation_loss.sum() + sh_loss.sum()
        
        return total_loss

    def kl_divergence_loss(self, mu, logvar):
        """
        计算 KL 散度损失，使 q(z|x) 逼近标准正态分布 N(0, I)
        Args:
            mu:  VAE 编码器的均值输出 (batch_size, latent_dim)
            logvar:  编码器输出的对数方差 log(σ^2) (batch_size, latent_dim)
        Returns:
            KL 散度损失标量值
        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

    def sample_from_gaussian(self, means, log_var):
        """
        从给定的均值和方差中采样出高斯隐向量
        
        Args:
            vvq_features: 输入特征，大小为 [batch_size, num_tokens, 16]
            latent_dim: 隐空间的维度，默认为 8
        
        Returns:
            sampled_latents: 从高斯分布中采样的隐向量，大小为 [batch_size, num_tokens, latent_dim]
        """

        # 对数方差转换为标准方差
        std_dev = torch.exp(0.5 * log_var)  # 因为方差是 log_var 的指数，因此 std_dev = exp(0.5 * log_var)
        
        # 从标准正态分布中采样
        epsilon = torch.randn_like(means)  # 生成与均值相同形状的标准正态分布噪声

        # 计算隐向量
        sampled_latents = means + std_dev * epsilon  # 使用 reparameterization trick 采样

        return sampled_latents

    def get_top_k_points_by_confidence(self, Rebuild_Gaussian, confidence, k=4096):
        """
        根据置信度获取前 k 个点。
        
        Rebuild_Gaussian: Tensor，形状为 (bs, point_num, 3)
        confidence: 置信度 Tensor，形状为 (bs, point_num, 1)
        k: 选择的点的数量，默认为 4096
        
        返回：选择的前 k 个点的张量，形状为 (bs, k, 3)
        """
        bs, point_num, _ = Rebuild_Gaussian.shape
        
        # 由于 confidence 形状为 (bs, point_num, 1)，去掉最后一个维度以进行排序
        confidence = confidence.squeeze(-1)  # 变成 (bs, point_num)
        
        # 获取前 k 个最大置信度的索引
        topk_indices = torch.topk(confidence, k=min(k, point_num), dim=-1, largest=True, sorted=True).indices  # (bs, k)
        
        # 根据索引选取对应的点
        topk_points = torch.gather(Rebuild_Gaussian, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, 3))
        
        return topk_points
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models.feature_extraction import create_feature_extractor
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import grid_res

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
    
    def forward(self, x):
        # 处理只有一个特征图的情况
        if len(x) == 1 and len(self.inner_blocks) == 1:
            return self.layer_blocks[0](self.inner_blocks[0](x[0]))
            
        # 原始FPN处理多尺度特征
        results = []
        for i, (inner_block, layer_block) in enumerate(zip(self.inner_blocks, self.layer_blocks)):
            if i == 0:
                results.append(layer_block(inner_block(x[i])))
            else:
                results.append(layer_block(inner_block(x[i]) + F.interpolate(results[-1], size=x[i].shape[2:], mode='nearest')))
        return results[-1]

class EfficientLOFTRFeatureExtractor(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # 加载EfficientNet作为backbone
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        
        # 简化的特征金字塔网络，只处理一个特征图
        self.fpn = FPN(
            in_channels_list=[1280],
            out_channels=256
        )
        
        # 加载预训练权重
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
    
    def load_pretrained_weights(self, pretrained_path):
        state_dict = torch.load(pretrained_path)
        backbone_state_dict = {k: v for k, v in state_dict.items() 
                             if 'backbone' in k or 'fpn' in k}
        self.load_state_dict(backbone_state_dict, strict=False)
    
    def forward(self, x):
        # 提取多尺度特征
        features = self.backbone.extract_features(x)
        # 将单一特征图包装为列表以适应FPN接口
        features_list = [features]
        features = self.fpn(features_list)
        return features

class FeatureMatcher(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(d_model*2, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
        # 保存最后的特征以用于损失计算
        self.last_feature1 = None
        self.last_feature2 = None
    
    def forward(self, feat1, feat2):
        # 特征匹配
        b, c, h, w = feat1.shape
        feat1 = feat1.flatten(2).permute(2, 0, 1)  # [h*w, b, c]
        feat2 = feat2.flatten(2).permute(2, 0, 1)  # [h*w, b, c]
        
        # Transformer特征匹配
        matched_feat1 = self.transformer(feat1)
        matched_feat2 = self.transformer(feat2)
        
        # 恢复空间维度
        matched_feat1 = matched_feat1.permute(1, 2, 0).view(b, c, h, w)
        matched_feat2 = matched_feat2.permute(1, 2, 0).view(b, c, h, w)
        
        # 保存特征以用于损失计算
        self.last_feature1 = matched_feat1
        self.last_feature2 = matched_feat2
        
        # 特征融合
        correlation = self.fusion(torch.cat([matched_feat1, matched_feat2], dim=1))
        return correlation

class ValidPointDiscriminator(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 使用自适应池化来确保固定大小的输出
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # 更新全连接层的输入尺寸
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 获取输入特征图的尺寸
        b, c, h, w = x.shape
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 应用自适应池化，确保固定尺寸
        x = self.adaptive_pool(x)
        
        # 展平特征
        x = x.view(b, -1)
        
        # 应用全连接层
        x = self.fc(x)
        return x

def compute_continuity_loss(features, valid_mask):
    """
    计算特征图的连续性损失
    Args:
        features: 特征图 [B, C, H, W]
        valid_mask: 有效点掩码 [B, 1] 或 [B]
    Returns:
        continuity_loss: 连续性损失
    """
    # 确保valid_mask是4D张量
    if valid_mask.dim() == 2:
        b, c = valid_mask.shape
        if c == 1:
            # 从 [B, 1] 扩展到 [B, 1, H, W]
            valid_mask = valid_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, features.size(2), features.size(3))
        else:
            # 如果不是 [B, 1] 形状，先转为 [B, 1] 再扩展
            valid_mask = valid_mask.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, features.size(2), features.size(3))
    elif valid_mask.dim() == 1:
        # 从 [B] 扩展到 [B, 1, H, W]
        valid_mask = valid_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, features.size(2), features.size(3))
    
    # 计算水平和垂直方向的梯度
    h_grad = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])
    v_grad = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])
    
    # 应用有效点掩码
    h_grad = h_grad * valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
    v_grad = v_grad * valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
    
    # 计算连续性损失
    continuity_loss = (h_grad.mean() + v_grad.mean()) / 2.0
    
    return continuity_loss

class ImprovedWarpNetwork(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        
        # 使用EfficientLOFTR的特征提取器
        self.feature_extractor = EfficientLOFTRFeatureExtractor(pretrained_path)
        
        # 特征匹配模块
        self.feature_matcher = FeatureMatcher()
        
        # 有效点判别网络
        self.valid_point_discriminator = ValidPointDiscriminator()
        
        # 保持原有的回归网络结构
        self.regressNet1_part1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 添加自适应池化确保输出固定大小
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 更新第一个线性层的输入尺寸为32*4*4=512
        self.regressNet1_part2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=8, bias=True)
        )
        
        self.regressNet2_part1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 添加自适应池化确保输出固定大小
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 更新第一个线性层的输入尺寸为32*4*4=512
        self.regressNet2_part2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)
        )
    
    def forward(self, input1_tensor, input2_tensor):
        batch_size, _, img_h, img_w = input1_tensor.size()
        
        # 使用EfficientLOFTR的特征提取
        feature_1 = self.feature_extractor(input1_tensor)
        feature_2 = self.feature_extractor(input2_tensor)
        
        # 特征匹配
        correlation = self.feature_matcher(feature_1, feature_2)
        
        # 有效点判别
        valid_scores = self.valid_point_discriminator(correlation)
        
        # 创建有效点掩码，不使用阈值判断
        valid_mask = valid_scores  # 现在是 [batch_size, 1] 的张量
        
        # 计算连续性损失
        continuity_loss = compute_continuity_loss(correlation, valid_mask)
        
        # 第一阶段：全局单应性估计
        temp_1 = self.regressNet1_part1(correlation)
        temp_1 = temp_1.view(temp_1.size()[0], -1)
        offset_1 = self.regressNet1_part2(temp_1)
        H_motion_1 = offset_1.reshape(-1, 4, 2)
        
        # 计算单应性矩阵
        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
        if torch.cuda.is_available():
            src_p = src_p.cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion_1
        H = torch_DLT.tensor_DLT(src_p/8, dst_p/8)
        
        # 第二阶段：局部变形估计
        temp_2 = self.regressNet2_part1(correlation)
        temp_2 = temp_2.view(temp_2.size()[0], -1)
        offset_2 = self.regressNet2_part2(temp_2)
        
        return offset_1, offset_2, valid_scores, continuity_loss 
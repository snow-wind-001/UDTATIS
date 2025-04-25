import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    感知损失，使用预训练的VGG模型特征计算感知相似度
    """
    def __init__(self, device='cuda'):
        super(PerceptualLoss, self).__init__()
        self.device = device
        # 加载预训练的VGG模型并提取特征层
        vgg = models.vgg19(pretrained=True).to(device)
        vgg.eval()
        
        # 冻结VGG参数
        for param in vgg.parameters():
            param.requires_grad = False
            
        # 使用特定层作为特征提取器
        self.blocks = nn.ModuleList([
            nn.Sequential(*list(vgg.features)[:4]),  # conv1_2
            nn.Sequential(*list(vgg.features)[4:9]),  # conv2_2
            nn.Sequential(*list(vgg.features)[9:18]),  # conv3_4
            nn.Sequential(*list(vgg.features)[18:27]),  # conv4_4
        ]).to(device)
        
        # 特征图权重
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4]
        
    def forward(self, x, y):
        """
        计算两个图像之间的感知损失
        Args:
            x: 输入图像
            y: 目标图像
        """
        # 确保图像在设备上
        x = x.to(self.device)
        y = y.to(self.device)
        
        # 两图像必须是相同尺寸
        assert x.shape == y.shape, f"Input and target shapes don't match: {x.shape} vs {y.shape}"
        
        # 初始化损失值
        loss = 0.0
        
        # 依次通过VGG的各个块提取特征
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            
            # 计算特征图的均方误差并加权
            loss += self.weights[i] * F.mse_loss(x, y)
        
        return loss


# 简化版本，用于在没有预训练模型时使用
class SimplePerceptualLoss(nn.Module):
    """
    简化的感知损失实现，适用于无法使用预训练模型的情况
    """
    def __init__(self, device='cuda'):
        super(SimplePerceptualLoss, self).__init__()
        self.device = device
        
        # 简单的卷积层替代VGG特征提取
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1).to(device)
        
        # 权重初始化
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
            
        # 冻结参数，只用于特征提取
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        """
        计算简化的感知损失
        """
        # 确保图像在设备上
        x = x.to(self.device)
        y = y.to(self.device)
        
        # 分别提取特征
        feat_x1 = F.relu(self.conv1(x))
        feat_y1 = F.relu(self.conv1(y))
        
        feat_x2 = F.relu(self.conv2(F.avg_pool2d(feat_x1, 2)))
        feat_y2 = F.relu(self.conv2(F.avg_pool2d(feat_y1, 2)))
        
        feat_x3 = F.relu(self.conv3(F.avg_pool2d(feat_x2, 2)))
        feat_y3 = F.relu(self.conv3(F.avg_pool2d(feat_y2, 2)))
        
        # 计算不同层次的损失
        loss1 = F.mse_loss(feat_x1, feat_y1)
        loss2 = F.mse_loss(feat_x2, feat_y2)
        loss3 = F.mse_loss(feat_x3, feat_y3)
        
        # 加权求和
        return 0.5 * loss1 + 0.3 * loss2 + 0.2 * loss3 
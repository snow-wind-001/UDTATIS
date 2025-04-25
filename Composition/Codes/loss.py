import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



# def get_vgg19_FeatureMap(vgg_model, input_255, layer_index):

#     vgg_mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape((1,3,1,1))
#     if torch.cuda.is_available():
#         vgg_mean = vgg_mean.cuda()
#     vgg_input = input_255-vgg_mean
#     #x = vgg_model.features[0](vgg_input)
#     #FeatureMap_list.append(x)


#     for i in range(0,layer_index+1):
#         if i == 0:
#             x = vgg_model.features[0](vgg_input)
#         else:
#             x = vgg_model.features[i](x)

#     return x



def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))


def boundary_extraction(mask):

    ones = torch.ones_like(mask)
    zeros = torch.zeros_like(mask)
    #define kernel
    in_channel = 1
    out_channel = 1
    kernel = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]
    kernel = torch.FloatTensor(kernel).expand(out_channel,in_channel,3,3)
    if torch.cuda.is_available():
        kernel = kernel.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    weight = nn.Parameter(data=kernel, requires_grad=False)

    #dilation
    x = F.conv2d(1-mask,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)

    return x*mask

def cal_boundary_term(inpu1_tesnor, inpu2_tesnor, mask1_tesnor, mask2_tesnor, stitched_image):
    boundary_mask1 = mask1_tesnor * boundary_extraction(mask2_tesnor)
    boundary_mask2 = mask2_tesnor * boundary_extraction(mask1_tesnor)

    loss1 = l_num_loss(inpu1_tesnor*boundary_mask1, stitched_image*boundary_mask1, 1)
    loss2 = l_num_loss(inpu2_tesnor*boundary_mask2, stitched_image*boundary_mask2, 1)

    return loss1+loss2, boundary_mask1


def cal_smooth_term_stitch(stitched_image, learned_mask1):


    delta = 1
    dh_mask = torch.abs(learned_mask1[:,:,0:-1*delta,:] - learned_mask1[:,:,delta:,:])
    dw_mask = torch.abs(learned_mask1[:,:,:,0:-1*delta] - learned_mask1[:,:,:,delta:])
    dh_diff_img = torch.abs(stitched_image[:,:,0:-1*delta,:] - stitched_image[:,:,delta:,:])
    dw_diff_img = torch.abs(stitched_image[:,:,:,0:-1*delta] - stitched_image[:,:,:,delta:])

    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss



def cal_smooth_term_diff(img1, img2, learned_mask, mask_overlap):
    """计算差分平滑损失"""
    # 计算图像差异
    diff = torch.abs(img1 - img2)
    diff_dx, diff_dy = gradient(diff)
    
    # 计算掩码梯度
    mask_dx, mask_dy = gradient(learned_mask)
    
    # 计算梯度平滑损失
    weight_x = torch.exp(-torch.mean(torch.abs(diff_dx), 1, keepdim=True))
    weight_y = torch.exp(-torch.mean(torch.abs(diff_dy), 1, keepdim=True))
    
    # 确保尺寸一致
    # mask_dx的尺寸为 [B, C, H-1, W]
    # mask_dy的尺寸为 [B, C, H, W-1]
    # 调整mask_overlap的尺寸以匹配
    mask_overlap_x = F.interpolate(mask_overlap, size=(mask_dx.shape[2], mask_dx.shape[3]), 
                                  mode='bilinear', align_corners=True)
    mask_overlap_y = F.interpolate(mask_overlap, size=(mask_dy.shape[2], mask_dy.shape[3]), 
                                  mode='bilinear', align_corners=True)
    
    # 只在重叠区域计算损失
    mask_dx = mask_dx * mask_overlap_x
    mask_dy = mask_dy * mask_overlap_y
    
    smooth_loss = torch.mean(weight_x * torch.abs(mask_dx)) + torch.mean(weight_y * torch.abs(mask_dy))
    
    return smooth_loss

    # dh_zeros = torch.zeros_like(dh_pixel)
    # dw_zeros = torch.zeros_like(dw_pixel)
    # if torch.cuda.is_available():
    #     dh_zeros = dh_zeros.cuda()
    #     dw_zeros = dw_zeros.cuda()


    # loss = l_num_loss(dh_pixel, dh_zeros, 1) + l_num_loss(dw_pixel, dw_zeros, 1)


    # return  loss, dh_pixel

class VGGPerceptualLoss(nn.Module):
    """使用VGG19提取特征的感知损失"""
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        blocks = []
        blocks.append(vgg[:4])    # relu1_2
        blocks.append(vgg[4:9])   # relu2_2
        blocks.append(vgg[9:18])  # relu3_4
        blocks.append(vgg[18:27]) # relu4_4
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        
        self.blocks = nn.ModuleList(blocks)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize
        
    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        # 将输入从[-1,1]归一化到[0,1]
        input = (input + 1) / 2
        target = (target + 1) / 2
        
        # 归一化到ImageNet均值和标准差
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
            
        return loss / len(self.blocks)

def gradient(pred):
    """计算图像梯度"""
    D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy

def cal_smooth_term_stitch(img, mask, weight=1.0):
    """计算拼接平滑损失"""
    # 计算图像梯度
    img_dx, img_dy = gradient(img)
    
    # 计算掩码梯度
    mask_dx, mask_dy = gradient(mask)
    
    # 计算梯度平滑损失
    weight_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True))
    weight_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True))
    
    smooth_loss = torch.mean(weight_x * torch.abs(mask_dx)) + torch.mean(weight_y * torch.abs(mask_dy))
    
    return smooth_loss * weight

def cal_boundary_term(im1, im2, mask1, mask2, stitched_image):
    """计算边界一致性损失"""
    # 创建边界掩码
    boundary_mask = mask1 * mask2
    
    # 在重叠区域计算两张图像和拼接结果的差异
    im1_diff = torch.abs(im1 - stitched_image) * boundary_mask
    im2_diff = torch.abs(im2 - stitched_image) * boundary_mask
    
    # 计算边界损失
    boundary_loss = torch.sum(im1_diff + im2_diff) / (torch.sum(boundary_mask) + 1e-6)
    
    return boundary_loss, boundary_mask

def cal_perceptual_loss(stitched_image, warp1, warp2, learned_mask1, learned_mask2):
    """计算感知损失"""
    # 初始化VGG感知损失
    perceptual_loss = VGGPerceptualLoss().to(stitched_image.device)
    
    # 使用学习的掩码对输入图像加权
    weighted_warp1 = warp1 * learned_mask1
    weighted_warp2 = warp2 * learned_mask2
    combined_image = weighted_warp1 + weighted_warp2
    
    # 计算拼接图像和加权组合图像之间的感知损失
    loss = perceptual_loss(stitched_image, combined_image)
    
    return loss

class SSIM(nn.Module):
    """结构相似性指数损失"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _gaussian(self, window_size, sigma):
        # 创建窗口索引张量
        coords = torch.arange(window_size, dtype=torch.float)
        # 计算中心点
        center = window_size // 2
        # 创建高斯权重
        x = coords - center
        x = x.pow(2)
        x = x / (2 * sigma**2)
        # 应用 exp 函数
        gauss = torch.exp(-x)
        # 归一化
        return gauss / gauss.sum()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

def cal_ssim_loss(stitched_image, warp1, warp2, learned_mask1, learned_mask2):
    """计算SSIM损失"""
    ssim_module = SSIM().to(stitched_image.device)
    
    # 使用学习的掩码对输入图像加权
    weighted_warp1 = warp1 * learned_mask1
    weighted_warp2 = warp2 * learned_mask2
    combined_image = weighted_warp1 + weighted_warp2
    
    # 计算SSIM损失
    loss = ssim_module(stitched_image, combined_image)
    
    return loss

def cal_color_consistency_loss(stitched_image, warp1, warp2, mask1, mask2):
    """计算颜色一致性损失"""
    # 只在非重叠区域计算颜色一致性
    non_overlap1 = mask1 * (1 - mask2)
    non_overlap2 = mask2 * (1 - mask1)
    
    # 计算非重叠区域的颜色一致性
    color_loss1 = torch.sum(torch.abs(stitched_image - warp1) * non_overlap1) / (torch.sum(non_overlap1) + 1e-6)
    color_loss2 = torch.sum(torch.abs(stitched_image - warp2) * non_overlap2) / (torch.sum(non_overlap2) + 1e-6)
    
    color_loss = color_loss1 + color_loss2
    
    return color_loss

class MultiScaleLoss(nn.Module):
    """多尺度损失，在不同分辨率上计算L1损失"""
    def __init__(self, scales=[1, 0.5, 0.25]):
        super(MultiScaleLoss, self).__init__()
        self.scales = scales
        
    def forward(self, stitched_image, target_image):
        loss = 0.0
        
        for scale in self.scales:
            if scale == 1:
                # 原始分辨率
                scaled_loss = F.l1_loss(stitched_image, target_image)
            else:
                # 下采样
                height = int(stitched_image.shape[2] * scale)
                width = int(stitched_image.shape[3] * scale)
                
                if height < 8 or width < 8:  # 避免尺寸过小
                    continue
                    
                down_stitched = F.interpolate(
                    stitched_image, 
                    size=(height, width), 
                    mode='bilinear', 
                    align_corners=False
                )
                down_target = F.interpolate(
                    target_image, 
                    size=(height, width), 
                    mode='bilinear', 
                    align_corners=False
                )
                scaled_loss = F.l1_loss(down_stitched, down_target)
            
            loss += scaled_loss
            
        return loss / len(self.scales)
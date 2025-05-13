import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math



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


# 添加辅助函数，转换为灰度图像
def convert_to_grayscale(image_tensor):
    """
    将彩色图片转换为灰白图片
    
    Args:
        image_tensor: 形状为[B, C, H, W]的图像张量
        
    Returns:
        灰白图像张量，保持原始形状但内容为灰度
    """
    # 检查是否为彩色图片（通道数大于1）
    if image_tensor.shape[1] > 1:
        # 使用标准RGB到灰度的转换公式
        gray = 0.299 * image_tensor[:, 0:1] + 0.587 * image_tensor[:, 1:2] + 0.114 * image_tensor[:, 2:3]
        # 复制灰度值到所有通道，保持原始通道数
        grayscale_image = gray.repeat(1, image_tensor.shape[1], 1, 1)
        return grayscale_image
    else:
        # 已经是单通道，直接返回
        return image_tensor

# 添加rgb_to_gray函数 (用于计算梯度一致性损失)
def rgb_to_gray(image_tensor):
    """
    将RGB图像转换为灰度图像 - 只保留一个通道
    
    Args:
        image_tensor: 形状为[B, C, H, W]的图像张量
        
    Returns:
        灰度图像张量 [B, 1, H, W]
    """
    # 检查是否为彩色图片（通道数大于1）
    if image_tensor.shape[1] > 1:
        # 使用标准RGB到灰度的转换公式
        gray = 0.299 * image_tensor[:, 0:1] + 0.587 * image_tensor[:, 1:2] + 0.114 * image_tensor[:, 2:3]
        return gray
    else:
        # 已经是单通道，直接返回
        return image_tensor

# 修改边界提取函数，使用灰度图像
def boundary_extraction(mask):
    """
    提取掩码的边界区域
    
    Args:
        mask: 输入掩码张量
        
    Returns:
        边界区域的张量
    """
    # 转换为灰度处理，确保单通道
    if mask.shape[1] > 1:
        mask = 0.299 * mask[:, 0:1] + 0.587 * mask[:, 1:2] + 0.114 * mask[:, 2:3]
    
    # 创建边界提取卷积核
    kernel = torch.tensor([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
    
    # 应用卷积来提取边界
    edge = F.conv2d(mask, kernel, padding=1)
    edge = torch.abs(edge)
    
    # 归一化边界强度
    if edge.max() > 0:
        edge = edge / edge.max()
    
    return edge

# 修改边界损失计算函数，使用灰度图像
def cal_boundary_term(im1, im2, mask1, mask2, stitched_image, valid_mask=None):
    """
    计算边界损失，确保边界区域平滑过渡
    
    Args:
        im1, im2: 输入图像
        mask1, mask2: 输入掩码
        stitched_image: 拼接图像
        valid_mask: 有效区域掩码
        
    Returns:
        边界损失与边界掩码
    """
    # 转换为灰度图像进行边界检测
    im1_gray = convert_to_grayscale(im1)
    im2_gray = convert_to_grayscale(im2)
    stitched_gray = convert_to_grayscale(stitched_image)
    
    # 提取边界
    boundary1 = boundary_extraction(mask1)
    boundary2 = boundary_extraction(mask2)
    
    # 合并边界区域
    boundary = boundary1 + boundary2
    boundary = torch.clamp(boundary, 0, 1)
    
    # 创建有效区域掩码
    if valid_mask is None:
        valid_mask = torch.ones_like(mask1)
    
    # 计算边界区域的图像差异
    loss = torch.mean(torch.abs(stitched_gray - im1_gray) * boundary * valid_mask) + \
           torch.mean(torch.abs(stitched_gray - im2_gray) * boundary * valid_mask)
    
    return loss / 2.0, boundary

# 修改平滑损失函数，使用灰度图像，并彻底解决维度不匹配问题
def cal_smooth_term_stitch(img, mask, weight=1.0):
    """计算平滑项损失，确保维度一致性和错误处理
    
    Args:
        img: 图像张量 [B,C,H,W]
        mask: 掩码张量 [B,1,H,W]
        weight: 损失权重
        
    Returns:
        平滑损失值
    """
    try:
        # 确保输入有效
        if img is None or mask is None:
            print("输入为空，无法计算平滑损失")
            return torch.tensor(0.0, device=img.device if img is not None else 'cpu')
            
        # 确保尺寸一致
        if img.shape[2:] != mask.shape[2:]:
            # 使用img的尺寸作为标准
            mask = F.interpolate(mask, size=img.shape[2:], mode='bilinear', align_corners=False)
            
        # 确保通道数一致
        if mask.shape[1] != 1 and mask.shape[1] != img.shape[1]:
            # 将mask转为单通道
            if mask.shape[1] > 1:
                mask = mask.mean(dim=1, keepdim=True)
        
        # 将图像转换为灰度
        gray = convert_to_grayscale(img)
        
        # 强制转换为偶数尺寸，防止梯度计算时维度不一致
        h, w = gray.shape[2], gray.shape[3]
        if h % 2 != 0 or w % 2 != 0:
            new_h = (h // 2) * 2
            new_w = (w // 2) * 2
            if new_h != h or new_w != w:
                gray = F.interpolate(gray, size=(new_h, new_w), mode='bilinear', align_corners=False)
                mask = F.interpolate(mask, size=(new_h, new_w), mode='bilinear', align_corners=False)
                
        # 确保掩码的维度与灰度图完全一致 (仅在宽高上)
        if gray.shape[2:] != mask.shape[2:]:
            mask = F.interpolate(mask, size=gray.shape[2:], mode='bilinear', align_corners=False)
        
        # 计算图像梯度 - 分别获取x和y方向的梯度
        grad_x, grad_y = gradient(gray, direction='both')
        
        # 确保掩码只有一个通道
        if mask.shape[1] > 1:
            mask = mask.mean(dim=1, keepdim=True)
            
        # 如果掩码通道数仍与梯度不一致，调整掩码通道数
        if mask.shape[1] != grad_x.shape[1]:
            mask = mask.repeat(1, grad_x.shape[1], 1, 1)
        
        # 计算图像梯度与掩码的加权和
        smooth_loss_x = torch.mean(torch.abs(grad_x) * mask)
        smooth_loss_y = torch.mean(torch.abs(grad_y) * mask)
        
        # 返回总平滑损失
        smooth_loss = smooth_loss_x + smooth_loss_y
        return smooth_loss * weight
        
    except Exception as e:
        print(f"计算平滑损失时出错: {e}")
        # 打印详细的输入形状信息帮助调试
        if img is not None and mask is not None:
            print(f"输入形状: img={img.shape}, mask={mask.shape}")
        else:
            print("输入为None")
        # 返回零张量作为替代
        return torch.tensor(0.0, device=img.device if img is not None else 'cpu', requires_grad=True)

# 修复感知损失中的维度不匹配问题
def cal_perceptual_loss(stitched_image, warp1, warp2, learned_mask1, learned_mask2, weight=1.0):
    """计算感知损失，确保维度一致并处理错误情况 - 增强版本，增加数值稳定性
    
    Args:
        stitched_image: 拼接后的图像
        warp1, warp2: 需要拼接的两个图像
        learned_mask1, learned_mask2: 学习得到的两个掩码
        weight: 损失权重系数
        
    Returns:
        perceptual_loss: 计算的感知损失
    """
    try:
        # 提升数值稳定性：检查并修复NaN/Inf
        stitched_image = torch.nan_to_num(stitched_image, nan=0.0, posinf=1.0, neginf=-1.0)
        warp1 = torch.nan_to_num(warp1, nan=0.0, posinf=1.0, neginf=-1.0)
        warp2 = torch.nan_to_num(warp2, nan=0.0, posinf=1.0, neginf=-1.0)
        learned_mask1 = torch.nan_to_num(learned_mask1, nan=0.5, posinf=1.0, neginf=0.0)
        learned_mask2 = torch.nan_to_num(learned_mask2, nan=0.5, posinf=1.0, neginf=0.0)
        
        # 确保掩码在[0,1]范围内
        learned_mask1 = torch.clamp(learned_mask1, 0.0, 1.0)
        learned_mask2 = torch.clamp(learned_mask2, 0.0, 1.0)
        
        # 确保所有输入都有相同的空间尺寸
        target_size = stitched_image.shape[2:]
        
        # 调整输入图像尺寸
        if warp1.shape[2:] != target_size:
            warp1 = F.interpolate(warp1, size=target_size, mode='bilinear', align_corners=False)
        if warp2.shape[2:] != target_size:
            warp2 = F.interpolate(warp2, size=target_size, mode='bilinear', align_corners=False)
        
        # 调整掩码尺寸
        if learned_mask1.shape[2:] != target_size:
            learned_mask1 = F.interpolate(learned_mask1, size=target_size, mode='bilinear', align_corners=False)
        if learned_mask2.shape[2:] != target_size:
            learned_mask2 = F.interpolate(learned_mask2, size=target_size, mode='bilinear', align_corners=False)
        
        # 确保掩码通道数为1
        if learned_mask1.shape[1] > 1:
            learned_mask1 = learned_mask1[:, :1]
        if learned_mask2.shape[1] > 1:
            learned_mask2 = learned_mask2[:, :1]
            
        # 如果掩码和图像通道数不一致，复制掩码通道
        if stitched_image.shape[1] > 1 and learned_mask1.shape[1] == 1:
            learned_mask1 = learned_mask1.repeat(1, stitched_image.shape[1], 1, 1)
        if stitched_image.shape[1] > 1 and learned_mask2.shape[1] == 1:
            learned_mask2 = learned_mask2.repeat(1, stitched_image.shape[1], 1, 1)
            
        # 创建或重用VGG感知损失模块
        if not hasattr(cal_perceptual_loss, 'vgg_loss'):
            cal_perceptual_loss.vgg_loss = VGGPerceptualLoss().to(stitched_image.device)
            
        # 计算区域1的感知损失 - 仅在掩码>0.5的区域
        region1_mask = (learned_mask1 > 0.5).float()
        if region1_mask.sum() > 10:  # 确保区域足够大
            region1_stitched = stitched_image * region1_mask
            region1_warp = warp1 * region1_mask
            loss1 = cal_perceptual_loss.vgg_loss(region1_stitched, region1_warp)
        else:
            loss1 = torch.tensor(0.0, device=stitched_image.device)
            
        # 计算区域2的感知损失 - 仅在掩码<0.5的区域
        region2_mask = (learned_mask2 > 0.5).float()
        if region2_mask.sum() > 10:  # 确保区域足够大
            region2_stitched = stitched_image * region2_mask
            region2_warp = warp2 * region2_mask
            loss2 = cal_perceptual_loss.vgg_loss(region2_stitched, region2_warp)
        else:
            loss2 = torch.tensor(0.0, device=stitched_image.device)
            
        # 计算总体感知损失 - 对整个图像
        loss_whole = cal_perceptual_loss.vgg_loss(stitched_image, 
                                              warp1 * learned_mask1 + warp2 * learned_mask2)
        
        # 组合损失，应用权重
        perceptual_loss = (loss1 + loss2 + loss_whole) * weight / 3.0
        
        # 最终检查损失值的有效性
        if torch.isnan(perceptual_loss) or torch.isinf(perceptual_loss):
            print("警告: 感知损失包含NaN或Inf，返回小的非零值")
            return torch.tensor(1e-5, device=stitched_image.device, requires_grad=True)
            
        return perceptual_loss
        
    except Exception as e:
        print(f"计算感知损失时出错: {e}")
        # 返回一个需要梯度的小张量而不是零，避免梯度消失
        return torch.tensor(1e-5, device=stitched_image.device, requires_grad=True)

# 修改梯度一致性损失，使用灰度图像
def calculate_gradient_consistency_loss(stitched_image, img1, img2, mask):
    """计算梯度一致性损失，确保维度一致并处理错误"""
    try:
        # 确保所有输入具有相同的空间尺寸
        target_size = stitched_image.shape[2:]
        if img1.shape[2:] != target_size:
            img1 = F.interpolate(img1, size=target_size, mode='bilinear', align_corners=False)
        if img2.shape[2:] != target_size:
            img2 = F.interpolate(img2, size=target_size, mode='bilinear', align_corners=False)
        if mask.shape[2:] != target_size:
            mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
        
        # 确保尺寸是偶数，以防止梯度计算时的舍入问题
        h, w = target_size
        if h % 2 != 0 or w % 2 != 0:
            h = (h // 2) * 2
            w = (w // 2) * 2
            stitched_image = F.interpolate(stitched_image, size=(h, w), mode='bilinear', align_corners=False)
            img1 = F.interpolate(img1, size=(h, w), mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=(h, w), mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        
        # 转换为灰度图
        if stitched_image.shape[1] > 1:
            gray_stitched = rgb_to_gray(stitched_image)
        else:
            gray_stitched = stitched_image
            
        if img1.shape[1] > 1:
            gray_img1 = rgb_to_gray(img1)
        else:
            gray_img1 = img1
            
        if img2.shape[1] > 1:
            gray_img2 = rgb_to_gray(img2)
        else:
            gray_img2 = img2
        
        # 确保掩码是单通道
        if mask.shape[1] > 1:
            mask = mask.mean(dim=1, keepdim=True)
        
        # 计算梯度 - 确保相同的卷积核和通道数
        grad_x_stitched = gradient(gray_stitched, 'x')
        grad_y_stitched = gradient(gray_stitched, 'y')
        grad_x_img1 = gradient(gray_img1, 'x')
        grad_y_img1 = gradient(gray_img1, 'y')
        grad_x_img2 = gradient(gray_img2, 'x')
        grad_y_img2 = gradient(gray_img2, 'y')
        
        # 再次检查梯度的尺寸是否一致
        gradient_shape = grad_x_stitched.shape[2:]
        if grad_x_img1.shape[2:] != gradient_shape:
            grad_x_img1 = F.interpolate(grad_x_img1, size=gradient_shape, mode='bilinear', align_corners=False)
            grad_y_img1 = F.interpolate(grad_y_img1, size=gradient_shape, mode='bilinear', align_corners=False)
        if grad_x_img2.shape[2:] != gradient_shape:
            grad_x_img2 = F.interpolate(grad_x_img2, size=gradient_shape, mode='bilinear', align_corners=False)
            grad_y_img2 = F.interpolate(grad_y_img2, size=gradient_shape, mode='bilinear', align_corners=False)
        if mask.shape[2:] != gradient_shape:
            mask = F.interpolate(mask, size=gradient_shape, mode='bilinear', align_corners=False)
            
        # 归一化梯度，避免数值太大或太小
        def norm_gradient(grad):
            return grad / (torch.abs(grad).mean() + 1e-7)
            
        grad_x_stitched = norm_gradient(grad_x_stitched)
        grad_y_stitched = norm_gradient(grad_y_stitched)
        grad_x_img1 = norm_gradient(grad_x_img1)
        grad_y_img1 = norm_gradient(grad_y_img1)
        grad_x_img2 = norm_gradient(grad_x_img2)
        grad_y_img2 = norm_gradient(grad_y_img2)
        
        # 创建权重图 - 使用遮罩确定哪些区域应该与img1/img2一致
        weight1 = mask
        weight2 = 1 - mask
        
        # 添加平滑过渡区域
        transition_area = (mask > 0.1) & (mask < 0.9)
        
        # 确保不除以零
        eps = 1e-7
        weight1_sum = weight1.sum() + eps
        weight2_sum = weight2.sum() + eps
        
        # 在过渡区域加强权重
        enhanced_weight = transition_area.float() * 5.0
        
        # 加权梯度差异
        x_diff1 = torch.abs(grad_x_stitched - grad_x_img1) * (weight1 + enhanced_weight)
        y_diff1 = torch.abs(grad_y_stitched - grad_y_img1) * (weight1 + enhanced_weight)
        
        x_diff2 = torch.abs(grad_x_stitched - grad_x_img2) * (weight2 + enhanced_weight)
        y_diff2 = torch.abs(grad_y_stitched - grad_y_img2) * (weight2 + enhanced_weight)
        
        # 计算总损失 - 添加数值稳定性检查
        loss_parts = []
        if weight1_sum > eps:
            loss_parts.append(x_diff1.sum() / weight1_sum)
            loss_parts.append(y_diff1.sum() / weight1_sum)
        if weight2_sum > eps:
            loss_parts.append(x_diff2.sum() / weight2_sum)
            loss_parts.append(y_diff2.sum() / weight2_sum)
            
        if len(loss_parts) > 0:
            loss = sum(loss_parts) / len(loss_parts)
        else:
            loss = torch.tensor(0.0, device=stitched_image.device)
            
        # 检查并处理NaN/Inf结果
        if torch.isnan(loss) or torch.isinf(loss):
            print("警告: 梯度一致性损失包含NaN或Inf，使用零替代")
            loss = torch.tensor(0.0, device=stitched_image.device)
        
        return loss
        
    except Exception as e:
        print(f"计算梯度一致性损失时出错: {e}")
        print(f"输入形状: stitched={stitched_image.shape}, img1={img1.shape}, img2={img2.shape}, mask={mask.shape}")
        # 返回零张量作为备选
        return torch.tensor(0.0, device=stitched_image.device, requires_grad=True)

def cal_smooth_term_diff(img1, img2, learned_mask, mask_overlap):
    """计算差异图像平滑损失 - 增强版本，增加数值稳定性
    
    Args:
        img1, img2: 输入图像
        learned_mask: 学习到的掩码，用于混合两个图像
        mask_overlap: 两个输入掩码重叠区域
        
    Returns:
        smooth_loss: 计算的平滑损失
    """
    try:
        # 提升数值稳定性：检查并修复NaN/Inf
        img1 = torch.nan_to_num(img1, nan=0.0, posinf=1.0, neginf=-1.0)
        img2 = torch.nan_to_num(img2, nan=0.0, posinf=1.0, neginf=-1.0)
        learned_mask = torch.nan_to_num(learned_mask, nan=0.5, posinf=1.0, neginf=0.0)
        mask_overlap = torch.nan_to_num(mask_overlap, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 确保掩码在[0,1]范围内
        learned_mask = torch.clamp(learned_mask, 0.0, 1.0)
        mask_overlap = torch.clamp(mask_overlap, 0.0, 1.0)
        
        # 确保维度一致
        if img1.shape[2:] != img2.shape[2:]:
            img2 = F.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
        
        if img1.shape[2:] != learned_mask.shape[2:]:
            learned_mask = F.interpolate(learned_mask, size=img1.shape[2:], mode='bilinear', align_corners=False)
            
        if img1.shape[2:] != mask_overlap.shape[2:]:
            mask_overlap = F.interpolate(mask_overlap, size=img1.shape[2:], mode='bilinear', align_corners=False)
            
        # 计算图像差异
        diff = img1 - img2
        
        # 检查差异是否包含无效值，如果有则做裁剪
        if torch.isnan(diff).any() or torch.isinf(diff).any():
            diff = torch.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=-1.0)
            diff = torch.clamp(diff, -2.0, 2.0)  # 限制差异范围
            
        # 计算x和y方向的梯度
        diff_dx, diff_dy = gradient(diff, direction='both')
        
        # 验证梯度是否包含无效值
        if torch.isnan(diff_dx).any() or torch.isinf(diff_dx).any():
            diff_dx = torch.nan_to_num(diff_dx, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(diff_dy).any() or torch.isinf(diff_dy).any():
            diff_dy = torch.nan_to_num(diff_dy, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算x方向的掩码梯度（确保纬度匹配）
        mask_dx, mask_dy = gradient(learned_mask, direction='both')
        
        # 确保梯度维度一致
        if diff_dx.shape != mask_dx.shape:
            if diff_dx.shape[2:] != mask_dx.shape[2:]:
                mask_dx = F.interpolate(mask_dx, size=diff_dx.shape[2:], mode='bilinear', align_corners=False)
            if mask_dx.shape[1] != diff_dx.shape[1]:
                mask_dx = mask_dx.repeat(1, diff_dx.shape[1] // mask_dx.shape[1], 1, 1)
        
        if diff_dy.shape != mask_dy.shape:
            if diff_dy.shape[2:] != mask_dy.shape[2:]:
                mask_dy = F.interpolate(mask_dy, size=diff_dy.shape[2:], mode='bilinear', align_corners=False)
            if mask_dy.shape[1] != diff_dy.shape[1]:
                mask_dy = mask_dy.repeat(1, diff_dy.shape[1] // mask_dy.shape[1], 1, 1)
        
        # 计算重叠区域的平滑损失
        overlap_sum = torch.sum(mask_overlap) + 1e-8
        total_sum = torch.sum(torch.ones_like(mask_overlap)) + 1e-8
        overlap_ratio = overlap_sum / total_sum
        loss_weight = torch.clamp(overlap_ratio * 10, 0.1, 1.0)
        
        # 安全取绝对值和加权处理
        abs_diff_dx = torch.abs(diff_dx)
        abs_diff_dy = torch.abs(diff_dy)
        
        # 处理可能的极端值
        abs_diff_dx = torch.clamp(abs_diff_dx, 0.0, 10.0)
        abs_diff_dy = torch.clamp(abs_diff_dy, 0.0, 10.0)
        
        # 加权平滑损失，确保重叠区域平滑
        sum_overlap_x = torch.sum(abs_diff_dx * mask_dx * mask_overlap) 
        sum_overlap_y = torch.sum(abs_diff_dy * mask_dy * mask_overlap)
        
        # 使用小的epsilon避免除零
        eps = 1e-8
        smooth_loss_x = sum_overlap_x / (overlap_sum + eps)
        smooth_loss_y = sum_overlap_y / (overlap_sum + eps)
        
        # 组合x和y方向的损失，应用权重
        smooth_loss = (smooth_loss_x + smooth_loss_y) * loss_weight
        
        # 最终检查损失值的有效性
        if torch.isnan(smooth_loss) or torch.isinf(smooth_loss):
            print("警告: 差异平滑损失包含NaN或Inf，返回小的非零值")
            return torch.tensor(1e-5, device=img1.device, requires_grad=True)
            
        return smooth_loss
    except Exception as e:
        print(f"计算差异平滑损失时出错: {e}")
        print(f"输入形状: img1={img1.shape}, img2={img2.shape}, learned_mask={learned_mask.shape}, mask_overlap={mask_overlap.shape}")
        # 返回零张量作为备选
        return torch.tensor(1e-5, device=img1.device, requires_grad=True)

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
        # 使用torchvision新的权重接口
        try:
            # 尝试使用新的权重API - 只加载特征部分
            from torchvision.models import VGG19_Weights
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except (ImportError, AttributeError):
            # 后备方案：使用旧API，但明确指定只要features部分
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            vgg = models.vgg19(pretrained=True)
            vgg = vgg.features  # 只保留特征提取部分
            
        blocks = []
        blocks.append(vgg[:4])    # relu1_2
        blocks.append(vgg[4:9])   # relu2_2
        blocks.append(vgg[9:18])  # relu3_4
        blocks.append(vgg[18:27]) # relu4_4
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                p.detach_()  # 确保完全分离计算图
        
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

def gradient(x, direction='both'):
    """计算图像梯度，支持x、y方向或两个方向

    Args:
        x: 输入图像张量 [B,C,H,W]
        direction: 'x', 'y', 或 'both'
        
    Returns:
        计算后的梯度张量，或者当direction='both'时返回(grad_x, grad_y)的元组
    """
    # 定义Sobel核
    if direction == 'x' or direction == 'both':
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        # 确保只有单通道输入
        if x.shape[1] > 1:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x

        # 添加反射填充避免边缘伪影
        pad = nn.ReflectionPad2d(1)
        x_pad = pad(x_gray)
        
        # 使用1x1卷积核进行卷积
        grad_x = F.conv2d(x_pad, sobel_x.repeat(1, 1, 1, 1), groups=1)
        
        if direction == 'x':
            return grad_x
    
    if direction == 'y' or direction == 'both':
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        # 确保只有单通道输入
        if x.shape[1] > 1:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x
            
        # 添加反射填充避免边缘伪影
        pad = nn.ReflectionPad2d(1)
        x_pad = pad(x_gray)
        
        # 使用1x1卷积核进行卷积
        grad_y = F.conv2d(x_pad, sobel_y.repeat(1, 1, 1, 1), groups=1)
        
        if direction == 'y':
            return grad_y
    
    # 如果计算两个方向，则返回单独的x和y梯度，而不是合并它们
    if direction == 'both':
        return grad_x, grad_y
    else:
        # 默认返回梯度的平方和开方（保持向后兼容）
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)

def cal_ssim_loss(stitched_image, warp1, warp2, learned_mask1, learned_mask2, weight=0.1):
    """计算SSIM损失，确保维度一致并处理错误情况
    
    Args:
        stitched_image: 拼接后的图像
        warp1, warp2: 需要拼接的两个图像
        learned_mask1, learned_mask2: 学习得到的两个掩码
        weight: 损失权重系数
        
    Returns:
        SSIM损失
    """
    try:
        # 确保所有输入都有相同的空间尺寸
        target_size = stitched_image.shape[2:]
        
        # 调整输入图像尺寸
        if warp1.shape[2:] != target_size:
            warp1 = F.interpolate(warp1, size=target_size, mode='bilinear', align_corners=False)
        if warp2.shape[2:] != target_size:
            warp2 = F.interpolate(warp2, size=target_size, mode='bilinear', align_corners=False)
        
        # 调整掩码尺寸
        if learned_mask1.shape[2:] != target_size:
            learned_mask1 = F.interpolate(learned_mask1, size=target_size, mode='bilinear', align_corners=False)
        if learned_mask2.shape[2:] != target_size:
            learned_mask2 = F.interpolate(learned_mask2, size=target_size, mode='bilinear', align_corners=False)
        
        # 确保掩码通道数为1
        if learned_mask1.shape[1] > 1:
            learned_mask1 = learned_mask1[:, :1]
        if learned_mask2.shape[1] > 1:
            learned_mask2 = learned_mask2[:, :1]
            
        # 如果掩码和图像通道数不一致，复制掩码通道
        if stitched_image.shape[1] > 1 and learned_mask1.shape[1] == 1:
            learned_mask1 = learned_mask1.repeat(1, stitched_image.shape[1], 1, 1)
        if stitched_image.shape[1] > 1 and learned_mask2.shape[1] == 1:
            learned_mask2 = learned_mask2.repeat(1, stitched_image.shape[1], 1, 1)
        
        # 创建SSIM计算模块
        ssim_module = SSIM(window_size=11, size_average=True)
        
        # 计算区域1的SSIM
        # 使用掩码选择区域1中的像素
        region1_stitched = stitched_image * learned_mask1
        region1_warp = warp1 * learned_mask1
        ssim1 = ssim_module(region1_stitched, region1_warp)
        
        # 计算区域2的SSIM
        region2_stitched = stitched_image * learned_mask2
        region2_warp = warp2 * learned_mask2
        ssim2 = ssim_module(region2_stitched, region2_warp)
        
        # 添加小值避免除以零
        eps = 1e-7
        mask1_sum = learned_mask1.sum() + eps
        mask2_sum = learned_mask2.sum() + eps
        
        # 加权SSIM
        ssim_loss = 0.0
        
        # 如果掩码1不为空，计算区域1的SSIM
        if mask1_sum > eps * 2:
            ssim_loss += (1.0 - ssim1) * (mask1_sum / (mask1_sum + mask2_sum))
        
        # 如果掩码2不为空，计算区域2的SSIM
        if mask2_sum > eps * 2:
            ssim_loss += (1.0 - ssim2) * (mask2_sum / (mask1_sum + mask2_sum))
            
        # 检查NaN和Inf
        if torch.isnan(ssim_loss) or torch.isinf(ssim_loss):
            print("警告: SSIM损失计算出现NaN或Inf，返回零损失")
            return torch.tensor(0.0, device=stitched_image.device)
            
        return ssim_loss * weight
    
    except Exception as e:
        print(f"计算SSIM损失时出错: {e}")
        # 返回零张量以确保训练不会中断
        return torch.tensor(0.0, device=stitched_image.device)

def cal_color_consistency_loss(stitched_image, warp1, warp2, mask1, mask2):
    """计算颜色一致性损失"""
    # 确保尺寸匹配
    if warp1.shape[2:] != stitched_image.shape[2:]:
        warp1 = F.interpolate(warp1, size=stitched_image.shape[2:], mode='bilinear', align_corners=False)
    
    if warp2.shape[2:] != stitched_image.shape[2:]:
        warp2 = F.interpolate(warp2, size=stitched_image.shape[2:], mode='bilinear', align_corners=False)
        
    if mask1.shape[2:] != stitched_image.shape[2:]:
        mask1 = F.interpolate(mask1, size=stitched_image.shape[2:], mode='bilinear', align_corners=False)
        
    if mask2.shape[2:] != stitched_image.shape[2:]:
        mask2 = F.interpolate(mask2, size=stitched_image.shape[2:], mode='bilinear', align_corners=False)
    
    # 确保通道数匹配
    if mask1.shape[1] == 1 and warp1.shape[1] > 1:
        mask1 = mask1.repeat(1, warp1.shape[1], 1, 1)
        
    if mask2.shape[1] == 1 and warp2.shape[1] > 1:
        mask2 = mask2.repeat(1, warp2.shape[1], 1, 1)
    
    # 只在非重叠区域计算颜色一致性
    non_overlap1 = mask1 * (1 - mask2)
    non_overlap2 = mask2 * (1 - mask1)
    
    # 计算非重叠区域的颜色一致性
    color_loss1 = torch.sum(torch.abs(stitched_image - warp1) * non_overlap1) / (torch.sum(non_overlap1) + 1e-6)
    color_loss2 = torch.sum(torch.abs(stitched_image - warp2) * non_overlap2) / (torch.sum(non_overlap2) + 1e-6)
    
    color_loss = color_loss1 + color_loss2
    
    return color_loss

def cal_panorama_consistency_loss(panorama_image, warp1, warp2, mask1, mask2, transition_mask):
    """
    计算全景图连续性损失，确保全景图在拼接区域具有良好的一致性和平滑过渡
    
    参数:
        panorama_image: 生成的全景图像
        warp1, warp2: 输入的两个源图像
        mask1, mask2: 输入的两个源图像掩码
        transition_mask: 拼接过渡区域掩码
        
    返回:
        panorama_consistency_loss: 全景图连续性损失
    """
    try:
        # 确保所有输入具有相同的空间尺寸
        h, w = panorama_image.shape[2:]
        if warp1.shape[2:] != (h, w):
            warp1 = F.interpolate(warp1, size=(h, w), mode='bilinear', align_corners=False)
        if warp2.shape[2:] != (h, w):
            warp2 = F.interpolate(warp2, size=(h, w), mode='bilinear', align_corners=False)
        if mask1.shape[2:] != (h, w):
            mask1 = F.interpolate(mask1, size=(h, w), mode='bilinear', align_corners=False)
        if mask2.shape[2:] != (h, w):
            mask2 = F.interpolate(mask2, size=(h, w), mode='bilinear', align_corners=False)
        if transition_mask.shape[2:] != (h, w):
            transition_mask = F.interpolate(transition_mask, size=(h, w), mode='bilinear', align_corners=False)
        
        # 检查并处理NaN/Inf
        panorama_image = torch.nan_to_num(panorama_image, nan=0.0, posinf=1.0, neginf=-1.0)
        warp1 = torch.nan_to_num(warp1, nan=0.0, posinf=1.0, neginf=-1.0)
        warp2 = torch.nan_to_num(warp2, nan=0.0, posinf=1.0, neginf=-1.0)
        mask1 = torch.nan_to_num(mask1, nan=0.5, posinf=1.0, neginf=0.0)
        mask2 = torch.nan_to_num(mask2, nan=0.5, posinf=1.0, neginf=0.0)
        transition_mask = torch.nan_to_num(transition_mask, nan=0.5, posinf=1.0, neginf=0.0)
        
        # 扩展过渡区域范围，以获得更平滑的过渡
        kernel_size = min(31, min(h, w) // 16)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 使用平均池化扩展过渡区域
        if kernel_size >= 3:
            expanded_transition = F.avg_pool2d(
                transition_mask,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            )
        else:
            expanded_transition = transition_mask
            
        # 计算梯度一致性损失 - 确保过渡区域的梯度连续
        gradient_loss = calculate_gradient_consistency_loss(
            panorama_image, 
            warp1, 
            warp2, 
            expanded_transition
        )
        
        # 计算感知损失 - 使用现有的感知损失函数
        # 确保掩码形状正确
        if transition_mask.shape[1] == 1 and panorama_image.shape[1] > 1:
            transition_mask_expanded = transition_mask.repeat(1, panorama_image.shape[1], 1, 1)
        else:
            transition_mask_expanded = transition_mask
            
        if expanded_transition.shape[1] == 1 and panorama_image.shape[1] > 1:
            expanded_transition = expanded_transition.repeat(1, panorama_image.shape[1], 1, 1)
            
        # 创建或重用VGG感知损失模块
        if not hasattr(cal_panorama_consistency_loss, 'vgg_loss'):
            cal_panorama_consistency_loss.vgg_loss = VGGPerceptualLoss().to(panorama_image.device)
            
        # 获取每个源图像对应的区域
        region1 = expanded_transition < 0.5
        region2 = expanded_transition >= 0.5
        
        # 创建源图像参考 - 在各自的区域使用对应的源图像
        reference_image = torch.zeros_like(panorama_image)
        reference_image = torch.where(
            region1.repeat(1, reference_image.shape[1], 1, 1) if region1.shape[1] == 1 else region1,
            warp1,
            reference_image
        )
        reference_image = torch.where(
            region2.repeat(1, reference_image.shape[1], 1, 1) if region2.shape[1] == 1 else region2,
            warp2,
            reference_image
        )
        
        # 过渡区域的感知损失
        perceptual_loss = cal_panorama_consistency_loss.vgg_loss(
            panorama_image * expanded_transition,
            reference_image * expanded_transition
        )
        
        # 计算颜色平滑损失 - 确保过渡区域颜色平滑
        # 创建扩展的过渡区域掩码 - 只关注实际的过渡区域
        transition_area = (expanded_transition > 0.05) & (expanded_transition < 0.95)
        transition_area = transition_area.float()
        
        # 计算过渡区域的颜色梯度
        color_grad_x, color_grad_y = gradient(panorama_image, direction='both')
        
        # 在过渡区域中最小化颜色梯度
        color_smoothness_loss = (
            torch.mean(torch.abs(color_grad_x) * transition_area) +
            torch.mean(torch.abs(color_grad_y) * transition_area)
        )
        
        # 计算结构损失 - 使用SSIM
        ssim_module = SSIM(window_size=11, size_average=True)
        ssim_score = ssim_module(
            panorama_image * expanded_transition,
            reference_image * expanded_transition
        )
        ssim_loss = 1.0 - ssim_score
        
        # 组合所有损失，注意权重调整
        panorama_consistency_loss = (
            0.5 * gradient_loss +     # 梯度一致性损失
            0.3 * perceptual_loss +   # 感知损失
            0.1 * color_smoothness_loss + # 颜色平滑损失
            0.1 * ssim_loss           # 结构相似性损失
        )
        
        # 检查NaN和Inf
        if torch.isnan(panorama_consistency_loss) or torch.isinf(panorama_consistency_loss):
            print("警告: 全景图一致性损失包含NaN或Inf，返回零损失")
            return torch.tensor(0.0, device=panorama_image.device, requires_grad=True)
            
        return panorama_consistency_loss
        
    except Exception as e:
        print(f"计算全景图一致性损失时出错: {e}")
        import traceback
        traceback.print_exc()
        # 返回一个需要梯度的较小张量
        return torch.tensor(1e-5, device=panorama_image.device, requires_grad=True)

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

# SSIM Implementation
class SSIM(nn.Module):
    """结构相似性指数，用于评估图像质量
    
    参考: Z. Wang et al. "Image quality assessment: from error visibility to
          structural similarity"
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size, sigma):
        """创建高斯窗口"""
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def _create_window(self, window_size, channel):
        """创建SSIM使用的窗口"""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        """计算SSIM，增加错误处理和安全检查"""
        try:
            # 检查设备并确保一致
            device = img1.device
            
            # 确保通道维度兼容
            if img1.shape[1] != img2.shape[1]:
                if img1.shape[1] == 1:
                    img1 = img1.repeat(1, img2.shape[1], 1, 1)
                elif img2.shape[1] == 1:
                    img2 = img2.repeat(1, img1.shape[1], 1, 1)
                else:
                    # 不兼容的通道维度，转换为灰度
                    img1 = convert_to_grayscale(img1)
                    img2 = convert_to_grayscale(img2)
                    
            # 确保空间尺寸一致
            if img1.shape[2:] != img2.shape[2:]:
                img2 = F.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
            
            # 检查NaN和Inf值
            if torch.isnan(img1).any() or torch.isinf(img1).any() or torch.isnan(img2).any() or torch.isinf(img2).any():
                print("警告: SSIM输入包含NaN或Inf值，已替换为零")
                img1 = torch.nan_to_num(img1, nan=0.0, posinf=1.0, neginf=0.0)
                img2 = torch.nan_to_num(img2, nan=0.0, posinf=1.0, neginf=0.0)
                
            # 获取批次大小和通道数
            (_, channel, _, _) = img1.size()
            
            # 如果通道数与窗口不同，重新创建窗口
            if channel != self.channel:
                self.channel = channel
                self.window = self._create_window(self.window_size, self.channel)
            
            # 确保窗口在正确的设备上
            window = self.window.to(device)
            
            # 添加小的常数以防除以零
            C1 = 0.01**2
            C2 = 0.03**2
            
            # 检查是否有负值，如果有则归一化到 [0,1]
            if torch.min(img1) < 0 or torch.min(img2) < 0:
                img1 = (img1 - torch.min(img1)) / (torch.max(img1) - torch.min(img1) + 1e-8)
                img2 = (img2 - torch.min(img2)) / (torch.max(img2) - torch.min(img2) + 1e-8)
            
            # 计算均值
            mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            # 计算方差和协方差
            sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
            
            # 防止数值不稳定
            sigma1_sq = torch.clamp(sigma1_sq, min=1e-8)
            sigma2_sq = torch.clamp(sigma2_sq, min=1e-8)
            
            # 计算SSIM
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            # 返回结果
            if self.size_average:
                ssim_value = ssim_map.mean()
            else:
                ssim_value = ssim_map.mean(1).mean(1).mean(1)
                
            # 最终检查
            if torch.isnan(ssim_value) or torch.isinf(ssim_value):
                print("警告: SSIM计算结果是NaN或Inf，返回零代替")
                return torch.tensor(0.0, device=device)
                
            return ssim_value.to(torch.float32)  # 确保返回的是浮点型张量
            
        except Exception as e:
            print(f"计算SSIM损失时出错: {e}")
            # 返回一个零张量
            return torch.tensor(0.0, device=img1.device, dtype=torch.float32, requires_grad=True)
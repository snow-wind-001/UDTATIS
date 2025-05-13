# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_model, Network
from dataset import *
import os
import numpy as np
import skimage
import cv2
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from improved_network import ImprovedWarpNetwork
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import glob

# path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return


def test(args):
    # 设置参数
    batch_size = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置路径
    model_path = args.pretrained_path if hasattr(args, 'pretrained_path') else os.path.join(last_path, 'model')
    result_path = args.result_path if hasattr(args, 'result_path') else os.path.join(last_path, 'results')
    
    # 创建结果目录
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    # 创建数据集
    test_dataset = UDISDataset(root_dir=args.test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = ImprovedWarpNetwork().to(device)
    
    # 加载模型权重
    if os.path.isdir(model_path):
        # 查找最新的checkpoint
        ckpt_list = glob.glob(os.path.join(model_path, "*.pth"))
        ckpt_list.sort()
        if len(ckpt_list) > 0:
            checkpoint = torch.load(ckpt_list[-1])
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                print(f'Loaded model from {ckpt_list[-1]}')
            else:
                model.load_state_dict(checkpoint)
                print(f'Loaded model weights from {ckpt_list[-1]}')
        else:
            print(f'No checkpoint found in {model_path}')
    elif os.path.isfile(model_path):
        # 直接加载指定的模型文件
        checkpoint = torch.load(model_path)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print(f'Loaded model from {model_path}')
        else:
            model.load_state_dict(checkpoint)
            print(f'Loaded model weights from {model_path}')
    else:
        print(f'Model path {model_path} not found')
    
    model.eval()
    
    # 创建网格变量
    from Warp.Codes.grid_res import GRID_H, GRID_W
    grid_h = GRID_H  # 从grid_res.py中导入标准网格大小
    grid_w = GRID_W  # 从grid_res.py中导入标准网格大小
    
    # 评估指标
    total_psnr = 0
    total_ssim = 0
    total_valid_points = 0
    total_continuity_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, (img1, img2, target) in enumerate(tqdm(test_loader)):
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # 前向传播
            offset_1, offset_2, valid_scores, continuity_loss = model(img1, img2)
            
            # 计算单应性矩阵
            batch_size, _, img_h, img_w = img1.size()
            H_motion = offset_1.reshape(-1, 4, 2)
            src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
            if torch.cuda.is_available():
                src_p = src_p.cuda()
            src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
            dst_p = src_p + H_motion
            H = torch_DLT.tensor_DLT(src_p/8, dst_p/8)
            
            # 计算网格变形
            mesh_motion = offset_2.reshape(-1, grid_h+1, grid_w+1, 2)
            
            # 应用变形
            warped_img2 = apply_warp(img2, H, mesh_motion)
            
            # 计算评估指标
            img1_np = img1[0].cpu().numpy().transpose(1, 2, 0)
            warped_img2_np = warped_img2[0].cpu().numpy().transpose(1, 2, 0)
            
            # 计算PSNR和SSIM
            try:
                # 确保图像在正确的数据范围内 (0-1或0-255)
                # 检查图像范围，如果是在[0,1]区间内，设置data_range=1，否则设置为255
                data_range = 1.0 if img1_np.max() <= 1.0 else 255.0
                
                # 计算PSNR
                batch_psnr = psnr(img1_np, warped_img2_np, data_range=data_range)
                
                # 检查图像大小以确保SSIM计算正确
                min_side = min(img1_np.shape[0], img1_np.shape[1])
                if min_side < 7:
                    batch_ssim = 0.0  # 如果图像太小，不计算SSIM
                    print(f"Image too small for SSIM: {img1_np.shape}")
                else:
                    win_size = min(7, min_side - (min_side % 2) + 1)  # 确保win_size是小于图像大小的最大奇数
                    batch_ssim = ssim(img1_np, warped_img2_np, win_size=win_size, channel_axis=2, data_range=data_range)
            except Exception as e:
                print(f"Error computing metrics: {e}")
                batch_psnr = 0.0
                batch_ssim = 0.0
            
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            total_valid_points += valid_scores.mean().item()
            total_continuity_loss += continuity_loss.item()
            num_samples += 1
            
            # 保存结果
            save_results(img1, warped_img2, batch_idx, result_path)
            
            # 打印批次信息
            print(f'Batch [{batch_idx}/{len(test_loader)}], '
                  f'PSNR: {batch_psnr:.2f}, SSIM: {batch_ssim:.4f}, '
                  f'Valid Points: {valid_scores.mean().item():.4f}, '
                  f'Continuity Loss: {continuity_loss.item():.4f}')
    
    # 打印平均评估指标
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_valid_points = total_valid_points / num_samples
    avg_continuity_loss = total_continuity_loss / num_samples
    print(f'Average PSNR: {avg_psnr:.2f}')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Average Valid Points: {avg_valid_points:.4f}')
    print(f'Average Continuity Loss: {avg_continuity_loss:.4f}')
    
    # 将结果写入文件
    with open(os.path.join(result_path, 'results.txt'), 'w') as f:
        f.write(f'Average PSNR: {avg_psnr:.2f}\n')
        f.write(f'Average SSIM: {avg_ssim:.4f}\n')
        f.write(f'Average Valid Points: {avg_valid_points:.4f}\n')
        f.write(f'Average Continuity Loss: {avg_continuity_loss:.4f}\n')

def H2Mesh(H, rigid_mesh):
    batch_size = rigid_mesh.size()[0]
    padded_mesh = torch.cat([rigid_mesh, torch.ones(batch_size, rigid_mesh.size()[1], rigid_mesh.size()[2], 1).cuda()], 3)
    H_padded_mesh = torch.matmul(H.unsqueeze(1).unsqueeze(1), padded_mesh.unsqueeze(4))
    return H_padded_mesh.squeeze(4)[:,:,:,:2] / H_padded_mesh.squeeze(4)[:,:,:,2:3]

def apply_warp(img, H, mesh_motion):
    batch_size, _, img_h, img_w = img.size()
    
    # 应用单应性变换
    warped_img = torch_homo_transform.transformer(img, H, (img.size(2), img.size(3)))
    
    # 获取网格
    from Warp.Codes.grid_res import GRID_H, GRID_W
    grid_h = GRID_H
    grid_w = GRID_W
    
    # 获取刚性网格
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    rigid_mesh_flat = rigid_mesh.reshape(batch_size, -1, 2)
    
    # 计算mesh_motion的预期形状
    expected_size = (grid_h+1)*(grid_w+1)
    
    # 检查mesh_motion的维度，进行必要的适配
    if mesh_motion.dim() == 2:
        # 如果是[batch_size*N]形式，重塑为[batch_size, N, 2]
        total_elements = mesh_motion.size(1)
        if total_elements % 2 == 0:
            elements_per_point = total_elements // 2
            mesh_motion = mesh_motion.reshape(batch_size, elements_per_point, 2)
        else:
            # 如果不是偶数，可能是其他格式，尝试直接将其重塑为正确格式
            mesh_motion = mesh_motion.reshape(batch_size, -1, 2)
    
    # 现在检查点的数量是否匹配
    if mesh_motion.size(1) != expected_size:
        print(f"Warning: mesh_motion has {mesh_motion.size(1)} points but expected {expected_size}")
        # 使用简单的零偏移代替
        mesh_motion = torch.zeros_like(rigid_mesh_flat)
    
    # 应用单应性矩阵到刚性网格
    ones = torch.ones(batch_size, rigid_mesh_flat.size(1), 1).to(rigid_mesh_flat.device)
    rigid_mesh_homo = torch.cat([rigid_mesh_flat, ones], 2)
    transformed_points = torch.bmm(H, rigid_mesh_homo.transpose(1, 2)).transpose(1, 2)
    ini_mesh = transformed_points[:, :, :2] / transformed_points[:, :, 2:3]
    
    # 添加网格运动
    mesh = ini_mesh + mesh_motion
    
    # 归一化网格
    norm_rigid_mesh_flat = rigid_mesh_flat.clone()
    norm_rigid_mesh_flat[:, :, 0] = norm_rigid_mesh_flat[:, :, 0] / (img_w/2) - 1
    norm_rigid_mesh_flat[:, :, 1] = norm_rigid_mesh_flat[:, :, 1] / (img_h/2) - 1
    
    norm_mesh = mesh.clone()
    norm_mesh[:, :, 0] = norm_mesh[:, :, 0] / (img_w/2) - 1
    norm_mesh[:, :, 1] = norm_mesh[:, :, 1] / (img_h/2) - 1
    
    # 应用TPS变换
    mask = torch.ones_like(img)
    if torch.cuda.is_available():
        mask = mask.cuda()
    
    output_tps = torch_tps_transform.transformer(torch.cat((img, mask), 1), norm_mesh, norm_rigid_mesh_flat, (img_h, img_w))
    warped_img = output_tps[:, 0:3, ...]
    
    return warped_img

def get_rigid_mesh(batch_size, h, w):
    # 实现网格生成，根据grid_h和grid_w设置网格大小
    from Warp.Codes.grid_res import GRID_H, GRID_W
    grid_h = GRID_H  # 从grid_res.py中导入标准网格大小
    grid_w = GRID_W  # 从grid_res.py中导入标准网格大小
    
    mesh = torch.zeros(batch_size, grid_h+1, grid_w+1, 2)
    for i in range(grid_h+1):
        for j in range(grid_w+1):
            mesh[:, i, j, 0] = j * (w / grid_w)
            mesh[:, i, j, 1] = i * (h / grid_h)
    
    if torch.cuda.is_available():
        mesh = mesh.cuda()
    
    return mesh

def get_norm_mesh(mesh, h, w):
    # 归一化网格坐标
    norm_mesh = mesh.clone()
    norm_mesh[:, :, :, 0] = norm_mesh[:, :, :, 0] / (w/2) - 1
    norm_mesh[:, :, :, 1] = norm_mesh[:, :, :, 1] / (h/2) - 1
    return norm_mesh

def save_results(img1, warped_img2, batch_idx, result_path):
    # 保存原始图像和变形后的图像
    img1_np = (img1[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    warped_img2_np = (warped_img2[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    
    cv2.imwrite(os.path.join(result_path, f'img1_{batch_idx}.png'), cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(result_path, f'warped_img2_{batch_idx}.png'), cv2.cvtColor(warped_img2_np, cv2.COLOR_RGB2BGR))
    
    # 创建对比图
    comparison = np.hstack((img1_np, warped_img2_np))
    cv2.imwrite(os.path.join(result_path, f'comparison_{batch_idx}.png'), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='/opt/data/private/nl/Data/UDIS-D/testing/')
    parser.add_argument('--result_path', type=str, default=os.path.join(last_path, 'results'))
    parser.add_argument('--pretrained_path', type=str, default=os.path.join(last_path, 'model'))

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)

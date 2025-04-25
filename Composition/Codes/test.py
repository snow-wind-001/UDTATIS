# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
from network import build_model, ImprovedDiffusionComposition
from dataset import *
import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

def visualize_results(input1, input2, stitched, denoised, mask1, mask2, save_path):
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 显示输入图像
    axes[0, 0].imshow(input1)
    axes[0, 0].set_title('Input Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(input2)
    axes[0, 1].set_title('Input Image 2')
    axes[0, 1].axis('off')
    
    # 显示拼接结果
    axes[0, 2].imshow(stitched)
    axes[0, 2].set_title('Stitched Image')
    axes[0, 2].axis('off')
    
    # 显示去噪结果
    axes[1, 0].imshow(denoised)
    axes[1, 0].set_title('Denoised Image')
    axes[1, 0].axis('off')
    
    # 显示掩码
    axes[1, 1].imshow(mask1, cmap='gray')
    axes[1, 1].set_title('Learned Mask 1')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mask2, cmap='gray')
    axes[1, 2].set_title('Learned Mask 2')
    axes[1, 2].axis('off')
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(stitched, denoised, input1, input2):
    # 计算PSNR
    psnr1 = psnr(input1, stitched)
    psnr2 = psnr(input2, stitched)
    psnr_denoised1 = psnr(input1, denoised)
    psnr_denoised2 = psnr(input2, denoised)
    
    # 计算SSIM
    ssim1 = ssim(input1, stitched, multichannel=True)
    ssim2 = ssim(input2, stitched, multichannel=True)
    ssim_denoised1 = ssim(input1, denoised, multichannel=True)
    ssim_denoised2 = ssim(input2, denoised, multichannel=True)
    
    return {
        'psnr': (psnr1 + psnr2) / 2,
        'psnr_denoised': (psnr_denoised1 + psnr_denoised2) / 2,
        'ssim': (ssim1 + ssim2) / 2,
        'ssim_denoised': (ssim_denoised1 + ssim_denoised2) / 2
    }

def test(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 设置路径
    model_path = args.pretrained_path if hasattr(args, 'pretrained_path') else os.path.join(last_path, 'model')
    result_path = args.result_path if hasattr(args, 'result_path') else os.path.join(last_path, 'results')
    
    # 设置扩散参数
    diffusion_params = {
        'num_timesteps': args.num_timesteps if hasattr(args, 'num_timesteps') else 1000,
        'beta_start': args.beta_start if hasattr(args, 'beta_start') else 1e-4,
        'beta_end': args.beta_end if hasattr(args, 'beta_end') else 0.02
    }

    # 创建数据集
    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # 定义网络
    net = ImprovedDiffusionComposition(diffusion_params=diffusion_params)
    if torch.cuda.is_available():
        net = net.cuda()

    # 加载模型权重
    if os.path.isdir(model_path):
        # 查找最新的checkpoint
        ckpt_list = glob.glob(os.path.join(model_path, "*.pth"))
        ckpt_list.sort()
        if len(ckpt_list) > 0:
            checkpoint = torch.load(ckpt_list[-1])
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                net.load_state_dict(checkpoint['model'])
                print(f'Loaded model from {ckpt_list[-1]}')
            else:
                net.load_state_dict(checkpoint)
                print(f'Loaded model weights from {ckpt_list[-1]}')
        else:
            print(f'No checkpoint found in {model_path}')
    elif os.path.isfile(model_path):
        # 直接加载指定的模型文件
        checkpoint = torch.load(model_path)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            net.load_state_dict(checkpoint['model'])
            print(f'Loaded model from {model_path}')
        else:
            net.load_state_dict(checkpoint)
            print(f'Loaded model weights from {model_path}')
    else:
        print(f'Model path {model_path} not found')

    # 创建输出目录
    save_dirs = args.save_dirs if hasattr(args, 'save_dirs') else {
        'learn_mask1': 'learn_mask1',
        'learn_mask2': 'learn_mask2',
        'composition': 'composition',
        'denoised': 'denoised',
        'visualization': 'visualization'
    }
    
    output_dirs = {}
    for key, dir_name in save_dirs.items():
        output_dirs[key] = os.path.join(result_path, dir_name)
        if not os.path.exists(output_dirs[key]):
            os.makedirs(output_dirs[key])

    print("##################start testing#######################")
    net.eval()
    
    # 评估指标
    total_metrics = {
        'psnr': 0,
        'psnr_denoised': 0,
        'ssim': 0,
        'ssim_denoised': 0
    }
    num_samples = 0
    
    with torch.no_grad():
        for i, batch_value in enumerate(tqdm(test_loader)):
            warp1_tensor = batch_value[0].float()
            warp2_tensor = batch_value[1].float()
            mask1_tensor = batch_value[2].float()
            mask2_tensor = batch_value[3].float()

            if torch.cuda.is_available():
                warp1_tensor = warp1_tensor.cuda()
                warp2_tensor = warp2_tensor.cuda()
                mask1_tensor = mask1_tensor.cuda()
                mask2_tensor = mask2_tensor.cuda()

            batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

            stitched_image = batch_out['stitched_image']
            learned_mask1 = batch_out['learned_mask1']
            learned_mask2 = batch_out['learned_mask2']
            denoised = batch_out['denoised']

            # 转换为numpy格式
            stitched_image = ((stitched_image[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            learned_mask1 = (learned_mask1[0]*255).cpu().detach().numpy().transpose(1,2,0)
            learned_mask2 = (learned_mask2[0]*255).cpu().detach().numpy().transpose(1,2,0)
            denoised = ((denoised[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            input1 = ((warp1_tensor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            input2 = ((warp2_tensor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)

            # 保存结果
            cv2.imwrite(os.path.join(output_dirs['learn_mask1'], f"{i+1:06d}.jpg"), learned_mask1)
            cv2.imwrite(os.path.join(output_dirs['learn_mask2'], f"{i+1:06d}.jpg"), learned_mask2)
            cv2.imwrite(os.path.join(output_dirs['composition'], f"{i+1:06d}.jpg"), stitched_image)
            cv2.imwrite(os.path.join(output_dirs['denoised'], f"{i+1:06d}.jpg"), denoised)
            
            # 可视化结果
            visualize_results(
                input1, input2, stitched_image, denoised,
                learned_mask1, learned_mask2,
                os.path.join(output_dirs['visualization'], f"{i+1:06d}.png")
            )

            # 计算评估指标
            metrics = calculate_metrics(stitched_image, denoised, input1, input2)
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            num_samples += 1

            print('Batch [{}/{}] PSNR: {:.2f} SSIM: {:.4f} PSNR(denoised): {:.2f} SSIM(denoised): {:.4f}'.format(
                i+1, len(test_loader), metrics['psnr'], metrics['ssim'],
                metrics['psnr_denoised'], metrics['ssim_denoised']))

    # 打印平均评估指标
    for key in total_metrics:
        total_metrics[key] /= num_samples
    
    print('\nAverage Metrics:')
    print('PSNR: {:.2f}'.format(total_metrics['psnr']))
    print('SSIM: {:.4f}'.format(total_metrics['ssim']))
    print('PSNR(denoised): {:.2f}'.format(total_metrics['psnr_denoised']))
    print('SSIM(denoised): {:.4f}'.format(total_metrics['ssim_denoised']))
    
    # 将结果写入文件
    with open(os.path.join(result_path, 'results.txt'), 'w') as f:
        f.write('Average Metrics:\n')
        f.write('PSNR: {:.2f}\n'.format(total_metrics['psnr']))
        f.write('SSIM: {:.4f}\n'.format(total_metrics['ssim']))
        f.write('PSNR(denoised): {:.2f}\n'.format(total_metrics['psnr_denoised']))
        f.write('SSIM(denoised): {:.4f}\n'.format(total_metrics['ssim_denoised']))

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='/opt/data/private/nl/Data/UDIS-D/testing/')
    parser.add_argument('--result_path', type=str, default=os.path.join(last_path, 'results'))
    parser.add_argument('--pretrained_path', type=str, default=os.path.join(last_path, 'model'))
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_start', type=float, default=1e-4)
    parser.add_argument('--beta_end', type=float, default=0.02)

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
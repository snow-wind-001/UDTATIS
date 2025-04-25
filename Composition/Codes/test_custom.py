#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import your model
from network import ImprovedDiffusionComposition
from dataset import TestDataset

# Define constants
last_path = os.path.dirname(os.path.abspath(__file__))

def build_model(model, warp1, warp2, mask1, mask2):
    """
    Build the model and generate the output dict
    """
    # Prepare inputs
    warped1 = torch.cat([warp1, mask1], dim=1)
    warped2 = torch.cat([warp2, mask2], dim=1)
    
    # Get output from model using correct parameter ordering
    learned_mask1, denoised, stitched_image = model.sample(warp1, warp2, mask1, mask2)
    
    # Calculate learned_mask2 based on the same logic as in network.py
    overlap_region = mask1 * mask2
    learned_mask2 = (mask2 - overlap_region) + overlap_region * (1-learned_mask1)
    
    # Normalize masks if needed
    mask_sum = learned_mask1 + learned_mask2
    mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero
    learned_mask1 = learned_mask1 / mask_sum
    learned_mask2 = learned_mask2 / mask_sum
    
    return {
        'learned_mask1': learned_mask1,
        'learned_mask2': learned_mask2,
        'denoised': denoised,
        'stitched_image': stitched_image
    }

def visualize_results(input1, input2, stitched, denoised, mask1, mask2, save_path):
    """
    Visualize and save the test results
    """
    # Convert to uint8 if needed
    input1 = input1.astype(np.uint8) if input1.dtype != np.uint8 else input1
    input2 = input2.astype(np.uint8) if input2.dtype != np.uint8 else input2
    stitched = stitched.astype(np.uint8) if stitched.dtype != np.uint8 else stitched
    denoised = denoised.astype(np.uint8) if denoised.dtype != np.uint8 else denoised
    
    # Create figure with 3 rows: inputs, masks, and results
    h, w = input1.shape[:2]
    vis_img = np.zeros((h*3, w*2, 3), dtype=np.uint8)
    
    # Row 1: Input images
    vis_img[:h, :w] = input1
    vis_img[:h, w:] = input2
    
    # Row 2: Masks
    mask1_rgb = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR) if mask1.shape[-1] == 1 else mask1
    mask2_rgb = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR) if mask2.shape[-1] == 1 else mask2
    vis_img[h:2*h, :w] = mask1_rgb
    vis_img[h:2*h, w:] = mask2_rgb
    
    # Row 3: Results
    vis_img[2*h:, :w] = stitched
    vis_img[2*h:, w:] = denoised
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_img, "Input 1", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, "Input 2", (w+10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, "Mask 1", (10, h+30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, "Mask 2", (w+10, h+30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, "Stitched", (10, 2*h+30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, "Denoised", (w+10, 2*h+30), font, 0.7, (255, 255, 255), 2)
    
    # Save the visualization
    cv2.imwrite(save_path, vis_img)

def calculate_metrics(stitched, denoised, input1, input2):
    """
    Calculate quality metrics between inputs and results
    """
    # Calculate PSNR
    psnr1 = psnr(input1, stitched)
    psnr2 = psnr(input2, stitched)
    psnr_denoised1 = psnr(input1, denoised)
    psnr_denoised2 = psnr(input2, denoised)
    
    # Calculate SSIM
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
    """
    Test function for the ImprovedDiffusionComposition model
    """
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Setup paths
    model_path = args.pretrained_path if hasattr(args, 'pretrained_path') else os.path.join(last_path, 'model')
    result_path = args.result_path if hasattr(args, 'result_path') else os.path.join(last_path, 'results_custom')
    
    # Setup diffusion parameters
    diffusion_params = {
        'num_timesteps': args.num_timesteps if hasattr(args, 'num_timesteps') else 1000,
        'beta_start': args.beta_start if hasattr(args, 'beta_start') else 1e-4,
        'beta_end': args.beta_end if hasattr(args, 'beta_end') else 0.02
    }

    # Create dataset
    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

    # Define network
    net = ImprovedDiffusionComposition(diffusion_params=diffusion_params)
    if torch.cuda.is_available():
        net = net.cuda()

    # Load model weights
    if os.path.isdir(model_path):
        # Find latest checkpoint
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
        # Load specified model file
        checkpoint = torch.load(model_path)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            net.load_state_dict(checkpoint['model'])
            print(f'Loaded model from {model_path}')
        else:
            net.load_state_dict(checkpoint)
            print(f'Loaded model weights from {model_path}')
    else:
        print(f'Model path {model_path} not found')

    # Create output directories
    save_dirs = {
        'learned_mask1': 'learned_mask1',
        'learned_mask2': 'learned_mask2',
        'composition': 'composition',
        'denoised': 'denoised',
        'visualization': 'visualization'
    }
    
    output_dirs = {}
    for key, dir_name in save_dirs.items():
        output_dirs[key] = os.path.join(result_path, dir_name)
        if not os.path.exists(output_dirs[key]):
            os.makedirs(output_dirs[key])

    print("################## Starting custom test #######################")
    net.eval()
    
    # Evaluation metrics
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

            # Process images with the model - using reduced sampling steps for faster inference
            if args.fast_sampling:
                # 调用sample方法时传递采样步数
                learned_mask1, denoised, stitched_image = net.sample(
                    warp1_tensor, 
                    warp2_tensor, 
                    mask1_tensor, 
                    mask2_tensor,
                    num_steps=args.sampling_steps,
                    use_ddim=True
                )
                
                # 计算learned_mask2
                overlap_region = mask1_tensor * mask2_tensor
                learned_mask2 = (mask2_tensor - overlap_region) + overlap_region * (1-learned_mask1)
                
                # 归一化掩码
                mask_sum = learned_mask1 + learned_mask2
                mask_sum = torch.clamp(mask_sum, min=1.0)
                learned_mask1 = learned_mask1 / mask_sum
                learned_mask2 = learned_mask2 / mask_sum
                
                batch_out = {
                    'learned_mask1': learned_mask1,
                    'learned_mask2': learned_mask2,
                    'denoised': denoised,
                    'stitched_image': stitched_image
                }
            else:
                # Use full sampling process
                batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

            stitched_image = batch_out['stitched_image']
            learned_mask1 = batch_out['learned_mask1']
            learned_mask2 = batch_out['learned_mask2']
            denoised = batch_out['denoised']

            # Convert to numpy format
            stitched_image = ((stitched_image[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            learned_mask1 = (learned_mask1[0]*255).cpu().detach().numpy().transpose(1,2,0)
            learned_mask2 = (learned_mask2[0]*255).cpu().detach().numpy().transpose(1,2,0)
            denoised = ((denoised[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            input1 = ((warp1_tensor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            input2 = ((warp2_tensor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)

            # Save results
            cv2.imwrite(os.path.join(output_dirs['learned_mask1'], f"{i+1:06d}.jpg"), learned_mask1)
            cv2.imwrite(os.path.join(output_dirs['learned_mask2'], f"{i+1:06d}.jpg"), learned_mask2)
            cv2.imwrite(os.path.join(output_dirs['composition'], f"{i+1:06d}.jpg"), stitched_image)
            cv2.imwrite(os.path.join(output_dirs['denoised'], f"{i+1:06d}.jpg"), denoised)
            
            # Visualize results
            visualize_results(
                input1, input2, stitched_image, denoised,
                learned_mask1, learned_mask2,
                os.path.join(output_dirs['visualization'], f"{i+1:06d}.png")
            )

            # Calculate evaluation metrics
            metrics = calculate_metrics(stitched_image, denoised, input1, input2)
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            num_samples += 1

            print(f'Batch [{i+1}/{len(test_loader)}] PSNR: {metrics["psnr"]:.2f} SSIM: {metrics["ssim"]:.4f} PSNR(denoised): {metrics["psnr_denoised"]:.2f} SSIM(denoised): {metrics["ssim_denoised"]:.4f}')

    # Print average evaluation metrics
    for key in total_metrics:
        total_metrics[key] /= num_samples
    
    print('\nAverage Metrics:')
    print(f'PSNR: {total_metrics["psnr"]:.2f}')
    print(f'SSIM: {total_metrics["ssim"]:.4f}')
    print(f'PSNR(denoised): {total_metrics["psnr_denoised"]:.2f}')
    print(f'SSIM(denoised): {total_metrics["ssim_denoised"]:.4f}')
    
    # Write results to file
    with open(os.path.join(result_path, 'results.txt'), 'w') as f:
        f.write('Average Metrics:\n')
        f.write(f'PSNR: {total_metrics["psnr"]:.2f}\n')
        f.write(f'SSIM: {total_metrics["ssim"]:.4f}\n')
        f.write(f'PSNR(denoised): {total_metrics["psnr_denoised"]:.2f}\n')
        f.write(f'SSIM(denoised): {total_metrics["ssim_denoised"]:.4f}\n')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Custom test script for ImprovedDiffusionComposition')

    # Basic parameters
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_path', type=str, default='data/UDIS-D/testing')
    parser.add_argument('--result_path', type=str, default=os.path.join(last_path, 'results_custom'))
    parser.add_argument('--pretrained_path', type=str, default=os.path.join(last_path, 'model'))
    
    # Diffusion model parameters
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_start', type=float, default=1e-4)
    parser.add_argument('--beta_end', type=float, default=0.02)
    
    # Sampling parameters
    parser.add_argument('--fast_sampling', action='store_true', help='Use accelerated sampling')
    parser.add_argument('--sampling_steps', type=int, default=50, help='Number of sampling steps (for fast sampling)')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args) 
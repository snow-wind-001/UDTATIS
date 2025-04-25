import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Warp.Codes.network import build_model, Network
from Warp.Codes.dataset import TrainDataset, UDISDataset
import glob
from Warp.Codes.loss import cal_lp_loss, inter_grid_loss, intra_grid_loss
from Warp.Codes.improved_network import ImprovedWarpNetwork
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import numpy as np
from tqdm import tqdm


# path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

def train(args):
    # 设置参数
    batch_size = args.batch_size
    num_epochs = args.max_epoch
    learning_rate = args.learning_rate if hasattr(args, 'learning_rate') else 1e-4
    
    # 设置路径
    model_save_path = args.model_save_path if hasattr(args, 'model_save_path') else os.path.join(last_path, 'model')
    summary_path = args.summary_path if hasattr(args, 'summary_path') else os.path.join(last_path, 'summary')
    
    # 创建目录
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    
    # 设置tensorboard
    writer = SummaryWriter(log_dir=summary_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集
    train_dataset = UDISDataset(root_dir=args.train_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = ImprovedWarpNetwork().to(device)
    
    # 定义优化器
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': learning_rate},
        {'params': model.feature_matcher.parameters(), 'lr': learning_rate},
        {'params': model.valid_point_discriminator.parameters(), 'lr': learning_rate * 2},
        {'params': model.regressNet1_part1.parameters(), 'lr': learning_rate * 10},
        {'params': model.regressNet1_part2.parameters(), 'lr': learning_rate * 10},
        {'params': model.regressNet2_part1.parameters(), 'lr': learning_rate * 10},
        {'params': model.regressNet2_part2.parameters(), 'lr': learning_rate * 10}
    ])
    
    # 从预训练权重恢复 - 检查现有权重
    start_epoch = 0
    pretrained_path = args.pretrained_path if hasattr(args, 'pretrained_path') else None
    
    # 从预训练权重恢复训练（如果提供了预训练路径）
    if pretrained_path:
        try:
            if os.path.isdir(pretrained_path):
                # 查找目录中最新的checkpoint
                ckpt_list = glob.glob(os.path.join(pretrained_path, "*.pth"))
                if ckpt_list:
                    # 按修改时间排序
                    ckpt_list.sort(key=os.path.getmtime, reverse=True)
                    checkpoint_path = ckpt_list[0]
                    print(f"找到最新的权重文件: {checkpoint_path}")
                else:
                    print(f"目录 {pretrained_path} 中没有找到权重文件，从头开始训练")
                    checkpoint_path = None
            else:
                # 直接使用指定的文件
                checkpoint_path = pretrained_path
                
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        model.load_state_dict(checkpoint['model'])
                        print("使用'model'键加载权重成功")
                    elif 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print("使用'model_state_dict'键加载权重成功")
                    else:
                        # 尝试直接作为state_dict加载
                        try:
                            model.load_state_dict(checkpoint)
                            print("直接加载权重成功")
                        except Exception as e:
                            print(f"权重加载失败: {e}, 从头开始训练")
                    
                    # 如果存在optimizer和epoch信息，也加载它们
                    if 'optimizer' in checkpoint and checkpoint['optimizer']:
                        try:
                            optimizer.load_state_dict(checkpoint['optimizer'])
                            print("加载优化器状态成功")
                        except Exception as e:
                            print(f"优化器状态加载失败: {e}")
                    
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1
                        print(f"从轮次 {start_epoch} 开始续训")
                else:
                    try:
                        model.load_state_dict(checkpoint)
                        print("直接加载权重成功")
                    except Exception as e:
                        print(f"权重加载失败: {e}, 从头开始训练")
                        
                print(f"成功加载预训练权重: {checkpoint_path}")
            else:
                print(f"预训练权重路径不存在: {checkpoint_path}, 从头开始训练")
        except Exception as e:
            print(f"加载预训练权重时出错: {e}")
            print("从头开始训练")
    else:
        print("未提供预训练权重路径，从头开始训练")
        
    # 输出训练起始轮次信息
    print(f"从轮次 {start_epoch} 开始训练，共 {num_epochs} 轮")
    
    # 设置损失权重
    homography_weight = args.homography_weight if hasattr(args, 'homography_weight') else 1.0
    mesh_weight = args.mesh_weight if hasattr(args, 'mesh_weight') else 1.0
    feature_weight = args.feature_weight if hasattr(args, 'feature_weight') else 0.1
    valid_point_weight = args.valid_point_weight if hasattr(args, 'valid_point_weight') else 0.5
    continuity_weight = args.continuity_weight if hasattr(args, 'continuity_weight') else 0.2
    
    # 定义损失函数
    criterion = {
        'homography': nn.MSELoss(),
        'mesh': nn.MSELoss(),
        'feature': nn.CosineSimilarity(),
        'valid_point': nn.BCELoss()
    }
    
    # 训练循环 - 从start_epoch开始
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        total_continuity_loss = 0
        
        for batch_idx, (img1, img2, target) in enumerate(tqdm(train_loader)):
            img1 = img1.to(device)
            img2 = img2.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            
            # 前向传播
            offset_1, offset_2, valid_scores, continuity_loss = model(img1, img2)
            
            # 将offset_1从[batch_size, 8]重塑为[batch_size, 4, 2]以匹配target['homography']的形状
            offset_1_reshaped = offset_1.reshape(-1, 4, 2)
            
            # 将offset_2重塑为[batch_size, (grid_h+1)*(grid_w+1), 2]以匹配target['mesh']
            # 根据grid_res.py，GRID_H=12, GRID_W=12，因此(GRID_H+1)*(GRID_W+1)=169
            offset_2_reshaped = offset_2.reshape(-1, 169, 2)
            
            # 计算损失
            loss_homography = criterion['homography'](offset_1_reshaped, target['homography'])
            loss_mesh = criterion['mesh'](offset_2_reshaped, target['mesh'])
            
            # 修改特征匹配损失计算
            # CosineSimilarity计算余弦相似度，但我们需要的是损失值(越小越好)
            # 因此用1减去余弦相似度的平均值作为损失
            cosine_sim = criterion['feature'](
                model.feature_matcher.last_feature1.flatten(2), 
                model.feature_matcher.last_feature2.flatten(2)
            )
            loss_feature = (1 - cosine_sim.mean())
            
            # 计算有效点判别损失
            batch_size, _, img_h, img_w = img1.size()
            with torch.no_grad():
                H_motion = offset_1_reshaped  # 直接使用已经重塑的张量
                src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
                if torch.cuda.is_available():
                    src_p = src_p.cuda()
                src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
                dst_p = src_p + H_motion
                
                # 计算单应性矩阵H
                H = torch_DLT.tensor_DLT(src_p/8, dst_p/8)
                
                # 使用H直接变换源点，而不是使用transformer函数
                # 首先将源点转为齐次坐标
                ones = torch.ones(batch_size, 4, 1).to(src_p.device)
                src_p_hom = torch.cat([src_p, ones], dim=2)  # [batch_size, 4, 3]
                
                # 使用H变换点
                dst_p_hom = torch.bmm(H, src_p_hom.transpose(1, 2)).transpose(1, 2)  # [batch_size, 4, 3]
                
                # 转回非齐次坐标
                dst_p_transformed = dst_p_hom[:, :, :2] / dst_p_hom[:, :, 2:3]
                
                # 计算误差
                error = torch.norm(dst_p - dst_p_transformed, dim=2)
                valid_labels_raw = (error < 5.0).float()
                
                # 计算每个样本的平均有效点比例，形状为[batch_size, 1]
                valid_labels = valid_labels_raw.mean(dim=1, keepdim=True)
            
            loss_valid = criterion['valid_point'](valid_scores, valid_labels)
            
            # 总损失
            total_batch_loss = (homography_weight * loss_homography + 
                              mesh_weight * loss_mesh + 
                              feature_weight * loss_feature + 
                              valid_point_weight * loss_valid + 
                              continuity_weight * continuity_loss)
            
            # 反向传播
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_continuity_loss += continuity_loss.item()
            
            # 写入tensorboard
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Total', total_batch_loss.item(), step)
            writer.add_scalar('Loss/Homography', loss_homography.item(), step)
            writer.add_scalar('Loss/Mesh', loss_mesh.item(), step)
            writer.add_scalar('Loss/Feature', loss_feature.item(), step)
            writer.add_scalar('Loss/ValidPoint', loss_valid.item(), step)
            writer.add_scalar('Loss/Continuity', continuity_loss.item(), step)
            writer.add_scalar('Metrics/ValidPoints', valid_scores.mean().item(), step)
            
            # 打印训练信息
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {total_batch_loss.item():.4f}, '
                      f'Continuity Loss: {continuity_loss.item():.4f}, '
                      f'Valid Points: {valid_scores.mean().item():.4f}')
        
        # 打印训练信息
        avg_loss = total_loss / len(train_loader)
        avg_continuity_loss = total_continuity_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {avg_loss:.4f}, '
              f'Continuity Loss: {avg_continuity_loss:.4f}')
        
        # 保存模型
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(model_save_path, f'checkpoint_epoch_{epoch+1}.pth'))


if __name__=="__main__":
    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_path', type=str, default='/opt/data/private/nl/Data/UDIS-D/training/')
    parser.add_argument('--model_save_path', type=str, default=os.path.join(last_path, 'model'))
    parser.add_argument('--summary_path', type=str, default=os.path.join(last_path, 'summary'))
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--homography_weight', type=float, default=1.0)
    parser.add_argument('--mesh_weight', type=float, default=1.0)
    parser.add_argument('--feature_weight', type=float, default=0.1)
    parser.add_argument('--valid_point_weight', type=float, default=0.5)
    parser.add_argument('--continuity_weight', type=float, default=0.2)

    # parse the arguments
    args = parser.parse_args()
    print(args)

    # train
    train(args)



import os
import sys
import torch
import torch.nn as nn
import argparse
from PIL import Image
import numpy as np
import glob
import cv2

# 添加项目根目录和Warp目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
warp_path = os.path.join(project_root, "Warp")
warp_codes_path = os.path.join(warp_path, "Codes")
sys.path.insert(0, warp_path)
sys.path.insert(0, warp_codes_path)

# 导入自定义可视化工具
from draw.utils import FeatureVisualizer

# 自定义网格生成函数，以避免依赖性问题
def custom_get_rigid_mesh(batch_size, h, w, grid_h, grid_w, device=None):
    """生成规则网格"""
    mesh = torch.zeros(batch_size, grid_h+1, grid_w+1, 2)
    for i in range(grid_h+1):
        for j in range(grid_w+1):
            mesh[:, i, j, 0] = j * (w / grid_w)
            mesh[:, i, j, 1] = i * (h / grid_h)
    
    if device is not None and torch.cuda.is_available():
        mesh = mesh.cuda()
    
    return mesh

# 尝试直接导入Warp网络模型
try:
    # 导入Warp模型
    from Warp.Codes.improved_network import ImprovedWarpNetwork
    from Warp.Codes.network import Network, get_rigid_mesh, get_norm_mesh
    import Warp.Codes.utils.torch_DLT as torch_DLT
    import Warp.Codes.utils.torch_homo_transform as torch_homo_transform
    import Warp.Codes.utils.torch_tps_transform as torch_tps_transform
    WARP_MODEL_AVAILABLE = True
    print("成功导入原始Warp模型")
except ImportError as e:
    # 尝试导入，使用相对路径
    try:
        # 确保utils目录也在路径中
        sys.path.insert(0, os.path.join(warp_codes_path, "utils"))
        # 使用完整路径导入
        from Warp.Codes.improved_network import ImprovedWarpNetwork
        from Warp.Codes.network import Network, get_rigid_mesh, get_norm_mesh
        import Warp.Codes.utils.torch_DLT as torch_DLT
        import Warp.Codes.utils.torch_homo_transform as torch_homo_transform
        import Warp.Codes.utils.torch_tps_transform as torch_tps_transform
        WARP_MODEL_AVAILABLE = True
        print("成功导入Warp模型（相对路径）")
    except ImportError as e2:
        print(f"无法导入Warp模型: {e2}")
        WARP_MODEL_AVAILABLE = False

# 简化可视化函数
def simple_visualize(img1_batch, img2_batch, output_dir, original_sizes=None, preserve_resolution=False):
    """当无法加载Warp模型时的简单可视化方案
    
    参数:
        img1_batch: 第一批图像
        img2_batch: 第二批图像
        output_dir: 输出目录
        original_sizes: 原始图像尺寸元组，格式为 (img1_sizes, img2_sizes)
        preserve_resolution: 是否保留原始分辨率
    """
    visualizer = FeatureVisualizer(save_dir=output_dir)
    
    # 提取原始尺寸信息
    img1_sizes = None
    img2_sizes = None
    if original_sizes and len(original_sizes) == 2:
        img1_sizes, img2_sizes = original_sizes
    
    visualizer.visualize_tensor(img1_batch, 'input_img1', original_sizes=img1_sizes, preserve_resolution=preserve_resolution)
    visualizer.visualize_tensor(img2_batch, 'input_img2', original_sizes=img2_sizes, preserve_resolution=preserve_resolution)
    print(f"已保存输入图像到: {output_dir}")
    return {'img1': img1_batch, 'img2': img2_batch}


class WarpVisualizer(nn.Module):
    """用于可视化warp过程的包装器类"""
    
    def __init__(self, model, output_dir='draw/output/warp'):
        """
        初始化warp可视化包装器
        
        参数:
            model: 原始的Warp模型
            output_dir: 输出目录
        """
        super(WarpVisualizer, self).__init__()
        self.model = model
        self.visualizer = FeatureVisualizer(save_dir=output_dir)
        
        # 为每个模块创建钩子
        self.hooks = []
        self.activations = {}
        
        # 注册钩子来捕获中间特征
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向钩子以捕获中间特征"""
        
        # 定义钩子函数
        def hook_fn(name):
            def fn(module, input, output):
                self.activations[name] = output
            return fn
        
        # 检查模型有哪些可用组件
        if hasattr(self.model, 'feature_extractor'):
            hook = self.model.feature_extractor.register_forward_hook(hook_fn('feature_extractor'))
            self.hooks.append(hook)
            
            # 检查feature_extractor是否有backbone或fpn
            if hasattr(self.model.feature_extractor, 'backbone'):
                hook = self.model.feature_extractor.backbone.register_forward_hook(hook_fn('backbone'))
                self.hooks.append(hook)
                
            if hasattr(self.model.feature_extractor, 'fpn'):
                hook = self.model.feature_extractor.fpn.register_forward_hook(hook_fn('fpn'))
                self.hooks.append(hook)
                
        # 检查特征匹配器
        match_component = None
        if hasattr(self.model, 'matcher'):
            match_component = self.model.matcher
        elif hasattr(self.model, 'feature_matcher'):
            match_component = self.model.feature_matcher
            
        if match_component is not None:
            hook = match_component.register_forward_hook(hook_fn('matcher'))
            self.hooks.append(hook)
            
        # 检查有效点判别器
        valid_points_component = None
        if hasattr(self.model, 'valid_points_estimator'):
            valid_points_component = self.model.valid_points_estimator
        elif hasattr(self.model, 'valid_point_discriminator'):
            valid_points_component = self.model.valid_point_discriminator
            
        if valid_points_component is not None:
            hook = valid_points_component.register_forward_hook(hook_fn('valid_points'))
            self.hooks.append(hook)
            
        # 其他组件
        if hasattr(self.model, 'regressNet1_part1'):
            hook = self.model.regressNet1_part1.register_forward_hook(hook_fn('regressNet1'))
            self.hooks.append(hook)
        
        if hasattr(self.model, 'regressNet2_part1'):
            hook = self.model.regressNet2_part1.register_forward_hook(hook_fn('regressNet2'))
            self.hooks.append(hook)
    
    def visualize_warp_pipeline(self, img1, img2, original_sizes=None, preserve_resolution=False):
        """
        可视化warp算法的完整流程
        
        参数:
            img1: 第一张图像
            img2: 第二张图像
            original_sizes: 原始图像尺寸元组，格式为 (img1_sizes, img2_sizes)，
                           其中每个元素是一个尺寸列表，如 [(width1, height1), (width2, height2), ...]
            preserve_resolution: 是否保留原始分辨率
        
        返回:
            变形后的图像和中间结果
        """
        batch_size, _, img_h, img_w = img1.shape
        device = img1.device
        
        # 清除上一次的激活
        self.activations = {}
        
        # 记录输入图像
        # 提取原始尺寸信息
        img1_sizes = None
        img2_sizes = None
        if original_sizes and len(original_sizes) == 2:
            img1_sizes, img2_sizes = original_sizes
        
        self.visualizer.visualize_tensor(img1, 'input_img1', original_sizes=img1_sizes, preserve_resolution=preserve_resolution)
        self.visualizer.visualize_tensor(img2, 'input_img2', original_sizes=img2_sizes, preserve_resolution=preserve_resolution)
        
        # 前向传播，获取网络输出
        with torch.no_grad():
            try:
                outputs = self.model(img1, img2)
                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    offset_1, offset_2, valid_scores = outputs[:3]
                    continuity_loss = outputs[3] if len(outputs) > 3 else torch.tensor(0.0)
                else:
                    raise ValueError("模型输出格式不符合预期")
                print("成功执行模型前向传播")
            except Exception as e:
                print(f"模型前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                return {'img1': img1, 'img2': img2}
        
        # 可视化所有捕获的特征
        for name, activation in self.activations.items():
            # 跳过非Tensor类型的激活
            if not isinstance(activation, torch.Tensor):
                continue
                
            try:
                # 根据不同模块使用不同的可视化方法
                if 'valid_points' in name:
                    if activation.dim() <= 2:  # 如果是2D或1D张量
                        activation = activation.unsqueeze(-1).unsqueeze(-1)  # 添加空间维度
                    self.visualizer.visualize_masks(activation, f'features_{name}')
                elif activation.dim() >= 4:  # 对于4D张量(批次、通道、高度、宽度)
                    self.visualizer.visualize_feature_maps(activation, f'features_{name}')
                elif activation.dim() == 3 and activation.size(0) == batch_size:  # 可能是掩码
                    self.visualizer.visualize_masks(activation, f'features_{name}')
                
                print(f"已可视化特征: {name}")
            except Exception as e:
                print(f"可视化特征 {name} 时出错: {e}")
        
        try:
            # 如果有可用于变形的函数，尝试可视化变形结果
            if WARP_MODEL_AVAILABLE and 'get_rigid_mesh' in globals() and 'get_norm_mesh' in globals():
                # 1. 计算单应性变换
                H_motion = offset_1.reshape(-1, 4, 2)
                src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
                if torch.cuda.is_available():
                    src_p = src_p.cuda()
                src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
                dst_p = src_p + H_motion
                
                try:
                    H = torch_DLT.tensor_DLT(src_p, dst_p)
                
                    # 应用单应性变换
                    M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                                  [0., img_h / 2.0, img_h / 2.0],
                                  [0., 0., 1.]])
                    if torch.cuda.is_available():
                        M_tensor = M_tensor.cuda()
                    M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                    M_tensor_inv = torch.inverse(M_tensor)
                    M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
                    H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)
                    
                    # 可视化单应性变换结果
                    mask = torch.ones_like(img2)
                    if torch.cuda.is_available():
                        mask = mask.cuda()
                    
                    warped_img_homo = torch_homo_transform.transformer(torch.cat((img2, mask), 1), H_mat, (img_h, img_w))
                    
                    # 应用原始分辨率
                    if preserve_resolution and img2_sizes:
                        self.visualizer.visualize_tensor(warped_img_homo[:, :3, ...], 'warped_img_homo', 
                                                       original_sizes=img2_sizes, preserve_resolution=preserve_resolution)
                        # 单独保存蒙版
                        if warped_img_homo.shape[1] > 3:
                            self.visualizer.visualize_tensor(warped_img_homo[:, 3:, ...], 'warped_mask_homo', 
                                                           original_sizes=img2_sizes, preserve_resolution=preserve_resolution)
                    else:
                        self.visualizer.visualize_tensor(warped_img_homo[:, :3, ...], 'warped_img_homo')
                    
                    # 2. 计算网格变形
                    try:
                        # 动态计算网格大小而不是使用固定值
                        offset_size = offset_2.size(1)
                        grid_points = int(offset_size / 2)
                        grid_side = int(np.sqrt(grid_points))
                        grid_h, grid_w = grid_side-1, grid_side-1
                        
                        mesh_motion = offset_2.reshape(-1, grid_side, grid_side, 2)
                        
                        # 首先使用单应性矩阵变换
                        mesh_warped = torch_homo_transform.transform_by_homo(
                            custom_get_rigid_mesh(batch_size, img_h, img_w, grid_h, grid_w, device),
                            H_mat
                        )
                        
                        # 然后使用局部网格变形
                        if 'torch_tps_transform' in globals():
                            try:
                                fixed_h, fixed_w = torch.meshgrid(
                                    torch.linspace(0, img_h, grid_h+1), 
                                    torch.linspace(0, img_w, grid_w+1)
                                )
                                fixed_p = torch.stack([fixed_w, fixed_h], -1).reshape(-1, 2)
                                if torch.cuda.is_available():
                                    fixed_p = fixed_p.cuda()
                                fixed_p = fixed_p.unsqueeze(0).expand(batch_size, -1, -1)
                                
                                # 添加偏移
                                moving_p = fixed_p + offset_2
                                moving_p = moving_p.reshape(batch_size, grid_h+1, grid_w+1, 2)
                                
                                # 使用TPS变换
                                tps_grid = torch_tps_transform.torch_tps_grid(moving_p, mesh_warped)
                                mesh_warped = tps_grid
                                
                                print("成功应用TPS变换")
                            except Exception as e:
                                print(f"TPS变换失败: {e}")
                        
                        # 绘制变形网格
                        grid_visualizations = []
                        for i in range(min(batch_size, self.visualizer.max_images)):
                            # 如果保留原始分辨率，使用原始尺寸创建网格图像
                            if preserve_resolution and img1_sizes and i < len(img1_sizes):
                                orig_w, orig_h = img1_sizes[i]
                                grid_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                            else:
                                grid_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                            
                            # 收集网格点坐标
                            points = []
                            for y in range(grid_h+1):
                                for x in range(grid_w+1):
                                    if preserve_resolution and img1_sizes and i < len(img1_sizes):
                                        orig_w, orig_h = img1_sizes[i]
                                        # 修改比例来匹配原始尺寸
                                        px = int(mesh_warped[i, y, x, 0].item() * (orig_w / img_w))
                                        py = int(mesh_warped[i, y, x, 1].item() * (orig_h / img_h))
                                    else:
                                        px = int(mesh_warped[i, y, x, 0].item())
                                        py = int(mesh_warped[i, y, x, 1].item())
                                    points.append((px, py))
                            
                            # 绘制网格线
                            for y in range(grid_h+1):
                                for x in range(grid_w):
                                    pt1 = points[y*(grid_w+1) + x]
                                    pt2 = points[y*(grid_w+1) + x + 1]
                                    cv2.line(grid_img, pt1, pt2, (0, 255, 0), 1)
                            
                            for y in range(grid_h):
                                for x in range(grid_w+1):
                                    pt1 = points[y*(grid_w+1) + x]
                                    pt2 = points[(y+1)*(grid_w+1) + x]
                                    cv2.line(grid_img, pt1, pt2, (0, 255, 0), 1)
                            
                            # 转换为tensor
                            grid_tensor = torch.from_numpy(grid_img.transpose(2, 0, 1)).float() / 255.0
                            grid_visualizations.append(grid_tensor)
                        
                        grid_tensor_batch = torch.stack(grid_visualizations)
                        if preserve_resolution and img1_sizes:
                            self.visualizer.visualize_tensor(grid_tensor_batch, 'mesh_deformation', 
                                                         original_sizes=img1_sizes[:len(grid_visualizations)], 
                                                         preserve_resolution=preserve_resolution)
                        else:
                            self.visualizer.visualize_tensor(grid_tensor_batch, 'mesh_deformation')
                        
                    except Exception as e:
                        print(f"网格变形可视化失败: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # 对比可视化
                    self.visualizer.side_by_side_comparison(
                        img1, img2, 'original_vs_target',
                        titles=['Source Image', 'Target Image']
                    )
                except Exception as e:
                    print(f"单应性变换失败: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"变形可视化过程中出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 可视化有效点掩码
        if valid_scores is not None:
            try:
                valid_scores_mask = valid_scores
                if valid_scores_mask.dim() <= 2:
                    valid_scores_mask = valid_scores_mask.unsqueeze(-1).unsqueeze(-1)
                if preserve_resolution and img1_sizes:
                    # 通过上采样来调整有效点掩码到原始分辨率
                    # 这里假设掩码与输入图像的空间尺寸相匹配，如果不是，可能需要额外处理
                    self.visualizer.visualize_masks(valid_scores_mask, 'valid_scores')
                else:
                    self.visualizer.visualize_masks(valid_scores_mask, 'valid_scores')
            except Exception as e:
                print(f"可视化有效点掩码出错: {e}")
        
        # 生成输出字典
        output_dict = {
            'img1': img1,
            'img2': img2,
            'valid_scores': valid_scores,
            'continuity_loss': continuity_loss
        }
        
        return output_dict
    
    def remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def load_images_from_directory(directory, device, target_size=None):
    """
    从目录加载所有图像，保留原始分辨率
    
    参数:
        directory: 图像目录
        device: 计算设备
        target_size: 目标尺寸，如果为None则保留原始分辨率
        
    返回:
        图像张量批次和原始尺寸列表
    """
    image_paths = glob.glob(os.path.join(directory, '*'))
    if not image_paths:
        raise ValueError(f"未在{directory}目录中找到图像")
    
    images = []
    original_sizes = []
    print(f"正在从{directory}加载{len(image_paths)}张图像...")
    
    for path in image_paths:
        # 仅处理常见图像格式
        if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            try:
                img = Image.open(path).convert('RGB')
                original_size = img.size  # 保存原始尺寸 (width, height)
                original_sizes.append(original_size)
                
                # 如果提供了目标尺寸，则调整大小用于处理
                if target_size:
                    img = img.resize(target_size, Image.BICUBIC)
                
                img_np = np.array(img) / 255.0  # 归一化到[0,1]
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [C, H, W]
                images.append(img_tensor)
                print(f"已加载图像: {path}, 原始尺寸: {original_size}")
            except Exception as e:
                print(f"无法加载图像 {path}: {e}")
    
    if not images:
        raise ValueError(f"无法加载{directory}中的任何图像")
    
    batch = torch.stack(images, dim=0).to(device)  # [B, C, H, W]
    return batch, original_sizes


def main():
    parser = argparse.ArgumentParser(description='Visualize warp process')
    parser.add_argument('--image1_dir', type=str, default='images/image1', help='Directory with first set of images')
    parser.add_argument('--image2_dir', type=str, default='images/image2', help='Directory with second set of images')
    parser.add_argument('--output_dir', type=str, default='draw/output/warp', help='Output directory')
    parser.add_argument('--model_path', type=str, default='/home/spikebai/owncode/UDTATIS/Warp/model/checkpoint_epoch_40.pth', help='Path to model checkpoint')
    parser.add_argument('--demo', action='store_true', help='使用演示模式，即使导入失败也尝试可视化')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--preserve_resolution', action='store_true', help='保留原始分辨率')
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256], help='处理用的目标图像尺寸 (宽度, 高度)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建设备
    device = torch.device(args.device)
    
    # 加载图像
    try:
        # 根据是否保留原始分辨率决定传入的target_size参数
        process_size = tuple(args.target_size) if not args.preserve_resolution else None
        
        # 加载images/image1文件夹中的所有图像
        img1_batch, img1_original_sizes = load_images_from_directory(args.image1_dir, device, process_size)
        
        # 加载images/image2文件夹中的所有图像
        img2_batch, img2_original_sizes = load_images_from_directory(args.image2_dir, device, process_size)
        
        # 确保两个批次的大小相同
        min_batch_size = min(img1_batch.size(0), img2_batch.size(0))
        img1_batch = img1_batch[:min_batch_size]
        img2_batch = img2_batch[:min_batch_size]
        img1_original_sizes = img1_original_sizes[:min_batch_size]
        img2_original_sizes = img2_original_sizes[:min_batch_size]
        
        print(f"处理 {min_batch_size} 对图像...")
        
        # 创建模型和可视化器
        model = None
        
        # 尝试使用原始的模型类
        if WARP_MODEL_AVAILABLE:
            try:
                if 'ImprovedWarpNetwork' in globals():
                    print("创建ImprovedWarpNetwork模型")
                    model = ImprovedWarpNetwork().to(device)
                elif 'Network' in globals():
                    print("创建Network模型")
                    model = Network().to(device)
                
                # 如果提供了模型路径，则加载权重
                if model is not None and args.model_path and os.path.exists(args.model_path):
                    print(f"加载模型权重: {args.model_path}")
                    try:
                        checkpoint = torch.load(args.model_path, map_location=device)
                        if isinstance(checkpoint, dict) and 'model' in checkpoint:
                            model.load_state_dict(checkpoint['model'])
                        else:
                            model.load_state_dict(checkpoint)
                        print("模型权重加载成功")
                    except Exception as e:
                        print(f"加载模型权重时出错: {e}")
                        if not args.demo:
                            model = None  # 如果不是演示模式，放弃模型
            except Exception as e:
                print(f"创建模型时出错: {e}")
                import traceback
                traceback.print_exc()
                if not args.demo:
                    model = None
        
        # 执行可视化
        if model is not None:
            # 使用模型进行可视化
            visualizer = WarpVisualizer(model, output_dir=args.output_dir)
            try:
                # 传递原始尺寸信息到可视化pipeline
                results = visualizer.visualize_warp_pipeline(
                    img1_batch, 
                    img2_batch, 
                    original_sizes=(img1_original_sizes, img2_original_sizes),
                    preserve_resolution=args.preserve_resolution
                )
                print(f"成功可视化warp过程。输出已保存到: {args.output_dir}")
            finally:
                visualizer.remove_hooks()
        else:
            # 使用简化可视化
            print("使用简化可视化工具（无模型）")
            simple_visualize(img1_batch, img2_batch, args.output_dir, 
                            original_sizes=(img1_original_sizes, img2_original_sizes),
                            preserve_resolution=args.preserve_resolution)
            
    except Exception as e:
        print(f"可视化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 
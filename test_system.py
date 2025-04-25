import os
import torch
import argparse
from torch.utils.data import DataLoader
import sys
import json

# 添加Warp和Composition路径到系统路径
sys.path.append('Warp/Codes')
sys.path.append('Composition/Codes')

def test_warp_dataset(use_virtual=True, use_gpu=False):
    print("\n===== Testing Warp Dataset =====")
    from Warp.Codes.dataset import UDISDataset, create_virtual_data_directories
    
    # 如果使用虚拟数据，先创建虚拟数据目录
    if use_virtual:
        data_paths = create_virtual_data_directories('data')
        train_path = data_paths['warp_train']
        test_path = data_paths['warp_test']
    else:
        # 使用配置文件中的路径
        with open('config.json', 'r') as f:
            config = json.load(f)
        train_path = config['warp']['train']['train_path']
        test_path = config['warp']['test']['test_path']
    
    # 创建训练数据集和数据加载器
    train_dataset = UDISDataset(root_dir=train_path, is_train=True, use_virtual=use_virtual)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # 创建测试数据集和数据加载器
    test_dataset = UDISDataset(root_dir=test_path, is_train=False, use_virtual=use_virtual)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 测试数据加载
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 获取一个批次的数据
    for batch_idx, (img1, img2, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"img1 shape: {img1.shape}")
        print(f"img2 shape: {img2.shape}")
        print(f"target keys: {target.keys()}")
        print(f"homography shape: {target['homography'].shape}")
        print(f"mesh shape: {target['mesh'].shape}")
        break
    
    # 测试GPU支持
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for testing")
    else:
        device = torch.device("cpu")
        print("Using CPU for testing")
    
    # 测试模型初始化和前向传播
    print("\n===== Testing Warp Model =====")
    try:
        from Warp.Codes.improved_network import ImprovedWarpNetwork
        model = ImprovedWarpNetwork().to(device)
        print(f"Model initialized successfully")
        
        # 前向传播测试
        img1 = img1.to(device)
        img2 = img2.to(device)
        with torch.no_grad():
            offset_1, offset_2, valid_scores, continuity_loss = model(img1, img2)
            print(f"Forward pass successful!")
            print(f"offset_1 shape: {offset_1.shape}")
            print(f"offset_2 shape: {offset_2.shape}")
            print(f"valid_scores shape: {valid_scores.shape}")
            print(f"continuity_loss: {continuity_loss.item()}")
    except Exception as e:
        print(f"Error during model test: {e}")
    
    return train_dataset, test_dataset

def test_composition_dataset(use_virtual=True, use_gpu=False):
    print("\n===== Testing Composition Dataset =====")
    
    # 导入数据集类
    from Composition.Codes.dataset import TrainDataset, TestDataset
    
    # 如果使用虚拟数据，检查是否已创建虚拟数据目录
    if use_virtual:
        from Warp.Codes.dataset import create_virtual_data_directories
        data_paths = create_virtual_data_directories('data')
        train_path = data_paths['comp_train']
        test_path = data_paths['comp_test']
    else:
        # 使用配置文件中的路径
        with open('config.json', 'r') as f:
            config = json.load(f)
        train_path = config['composition']['train']['train_path']
        test_path = config['composition']['test']['test_path']
    
    # 创建训练数据集和数据加载器
    train_dataset = TrainDataset(data_path=train_path, use_virtual=use_virtual)
    train_loader = DataLoader(dataset=train_dataset, batch_size=2, num_workers=0, shuffle=True)
    
    # 创建测试数据集和数据加载器
    test_dataset = TestDataset(data_path=test_path, use_virtual=use_virtual)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=0, shuffle=False)
    
    # 测试数据加载
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 获取一个批次的数据
    for batch_idx, (warp1, warp2, mask1, mask2) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"warp1 shape: {warp1.shape}")
        print(f"warp2 shape: {warp2.shape}")
        print(f"mask1 shape: {mask1.shape}")
        print(f"mask2 shape: {mask2.shape}")
        break
    
    # 测试GPU支持
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for testing")
    else:
        device = torch.device("cpu")
        print("Using CPU for testing")
    
    # 测试模型初始化和前向传播
    print("\n===== Testing Composition Model =====")
    try:
        from Composition.Codes.network import ImprovedDiffusionComposition, build_model
        
        diffusion_params = {
            'num_timesteps': 10,  # 减少以加快测试速度
            'beta_start': 1e-4,
            'beta_end': 0.02
        }
        
        model = ImprovedDiffusionComposition(diffusion_params=diffusion_params).to(device)
        print(f"Model initialized successfully")
        
        # 前向传播测试
        warp1 = warp1.to(device)
        warp2 = warp2.to(device)
        mask1 = mask1.to(device)
        mask2 = mask2.to(device)
        
        with torch.no_grad():
            # 测试模型的原始前向传播
            res, denoised = model(warp1, warp2, mask1, mask2)
            print(f"Model forward pass successful!")
            print(f"res shape: {res.shape}")
            print(f"denoised shape: {denoised.shape}")
            
            # 测试build_model函数
            batch_out = build_model(model, warp1, warp2, mask1, mask2)
            print(f"build_model successful!")
            print(f"batch_out keys: {batch_out.keys()}")
            for key, value in batch_out.items():
                print(f"{key} shape: {value.shape}")
    except Exception as e:
        print(f"Error during model test: {e}")
    
    return train_dataset, test_dataset

def test_main_script(use_virtual=True, use_gpu=False):
    """测试main.py脚本"""
    print("\n===== Testing Main Script =====")
    
    # 修改config.json以使用虚拟数据
    if use_virtual:
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            # 更新数据路径到虚拟数据目录
            from Warp.Codes.dataset import create_virtual_data_directories
            data_paths = create_virtual_data_directories('data')
            
            config['warp']['train']['train_path'] = data_paths['warp_train']
            config['warp']['test']['test_path'] = data_paths['warp_test']
            config['composition']['train']['train_path'] = data_paths['comp_train']
            config['composition']['test']['test_path'] = data_paths['comp_test']
            
            # 减少epochs以加快测试
            config['warp']['train']['max_epoch'] = 1
            config['composition']['train']['max_epoch'] = 1
            
            # 设置设备
            if use_gpu and torch.cuda.is_available():
                gpu_id = "0"
            else:
                gpu_id = "-1"  # 使用CPU
            
            config['warp']['train']['gpu'] = gpu_id
            config['warp']['test']['gpu'] = gpu_id
            config['composition']['train']['gpu'] = gpu_id
            config['composition']['test']['gpu'] = gpu_id
            
            # 保存更新后的配置
            with open('test_config.json', 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"Created test configuration file: test_config.json")
            
            # 测试main.py脚本的导入
            try:
                import main
                print("Successfully imported main.py")
            except Exception as e:
                print(f"Error importing main.py: {e}")
                
            # 打印测试命令提示
            print("\nTo test main.py, run the following command:")
            print("python main.py --config test_config.json --mode test --part all")
            
        except Exception as e:
            print(f"Error updating config: {e}")
    else:
        print("Skipping main.py test since virtual data is not enabled.")

def test_miniature_training(use_virtual=True, use_gpu=False):
    """测试小型训练循环"""
    print("\n===== Testing Miniature Training Loop =====")
    
    if not use_virtual:
        print("Skipping miniature training since virtual data is not enabled.")
        return
    
    # 测试Warp模型的训练循环
    print("\n----- Warp Training Loop -----")
    try:
        from Warp.Codes.dataset import UDISDataset, create_virtual_data_directories
        from Warp.Codes.improved_network import ImprovedWarpNetwork
        import torch.optim as optim
        import torch.nn as nn
        
        # 创建虚拟数据
        data_paths = create_virtual_data_directories('data')
        train_path = data_paths['warp_train']
        
        # 创建数据集和数据加载器
        train_dataset = UDISDataset(root_dir=train_path, is_train=True, use_virtual=True)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        
        # 设置设备
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU for training")
        else:
            device = torch.device("cpu")
            print("Using CPU for training")
        
        # 创建模型和优化器
        model = ImprovedWarpNetwork().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # 定义损失函数
        criterion = {
            'homography': nn.MSELoss(),
            'mesh': nn.MSELoss(),
            'feature': nn.CosineSimilarity(),
            'valid_point': nn.BCELoss()
        }
        
        # 执行一个小训练循环
        model.train()
        num_mini_batches = 2  # 只运行几个批次
        
        for batch_idx, (img1, img2, target) in enumerate(train_loader):
            if batch_idx >= num_mini_batches:
                break
                
            img1 = img1.to(device)
            img2 = img2.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            
            # 前向传播
            offset_1, offset_2, valid_scores, continuity_loss = model(img1, img2)
            
            # 计算损失
            loss_homography = criterion['homography'](offset_1, target['homography'])
            loss_mesh = criterion['mesh'](offset_2, target['mesh'])
            loss_feature = criterion['feature'](model.feature_matcher.last_feature1, 
                                             model.feature_matcher.last_feature2)
            
            # 有效点判别损失需要更多计算，这里简化处理
            loss_valid = torch.tensor(0.1).to(device)
            
            # 总损失
            total_batch_loss = (loss_homography + loss_mesh + 0.1 * loss_feature + 
                              0.5 * loss_valid + 0.2 * continuity_loss)
            
            # 反向传播
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            
            print(f"Batch {batch_idx}, Loss: {total_batch_loss.item():.4f}")
        
        print("Warp training mini-loop completed successfully!")
        
    except Exception as e:
        print(f"Error during Warp training test: {e}")
    
    # 测试Composition模型的训练循环
    print("\n----- Composition Training Loop -----")
    try:
        from Composition.Codes.dataset import TrainDataset
        from Composition.Codes.network import ImprovedDiffusionComposition, build_model
        from Composition.Codes.loss import cal_boundary_term, cal_smooth_term_stitch, cal_smooth_term_diff
        import torch.optim as optim
        from torch.cuda.amp import autocast, GradScaler
        
        # 创建数据集和数据加载器
        train_path = data_paths['comp_train']
        train_dataset = TrainDataset(data_path=train_path, use_virtual=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
        
        # 设置扩散参数
        diffusion_params = {
            'num_timesteps': 10,  # 减少以加快测试
            'beta_start': 1e-4,
            'beta_end': 0.02
        }
        
        # 创建模型和优化器
        model = ImprovedDiffusionComposition(diffusion_params=diffusion_params).to(device)
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': 1e-4},
            {'params': model.diffusion.parameters(), 'lr': 1e-4}
        ])
        
        # 添加混合精度训练
        scaler = GradScaler()
        
        # 执行一个小训练循环
        model.train()
        num_mini_batches = 2  # 只运行几个批次
        
        for batch_idx, (warp1, warp2, mask1, mask2) in enumerate(train_loader):
            if batch_idx >= num_mini_batches:
                break
                
            warp1 = warp1.to(device)
            warp2 = warp2.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            
            with autocast():
                batch_out = build_model(model, warp1, warp2, mask1, mask2)
                
                learned_mask1 = batch_out['learned_mask1']
                learned_mask2 = batch_out['learned_mask2']
                stitched_image = batch_out['stitched_image']
                denoised = batch_out['denoised']
                
                # 简化损失计算
                # 实际训练中会计算boundary_loss等
                total_loss = torch.mean((denoised - warp1)**2) + torch.mean((denoised - warp2)**2)
            
            # 反向传播
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print(f"Batch {batch_idx}, Loss: {total_loss.item():.4f}")
        
        print("Composition training mini-loop completed successfully!")
        
    except Exception as e:
        print(f"Error during Composition training test: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test UDIS2 System')
    parser.add_argument('--virtual', action='store_true', help='Use virtual data for testing')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()
    
    print("===== UDIS2 System Testing =====")
    print(f"Using virtual data: {args.virtual}")
    print(f"Using GPU: {args.gpu}")
    
    # 检查GPU可用性
    if args.gpu:
        if torch.cuda.is_available():
            print(f"CUDA is available. Found {torch.cuda.device_count()} device(s).")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. Will use CPU instead.")
    
    # 测试Warp模块
    warp_train_dataset, warp_test_dataset = test_warp_dataset(use_virtual=args.virtual, use_gpu=args.gpu)
    
    # 测试Composition模块
    comp_train_dataset, comp_test_dataset = test_composition_dataset(use_virtual=args.virtual, use_gpu=args.gpu)
    
    # 测试main.py脚本
    test_main_script(use_virtual=args.virtual, use_gpu=args.gpu)
    
    # 测试小型训练循环
    test_miniature_training(use_virtual=args.virtual, use_gpu=args.gpu)
    
    print("\n===== UDIS2 System Test Completed =====") 
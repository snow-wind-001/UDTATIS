# Unsupervised Disparity - Tolerant Algorithm for Terahertz Image Stitching (UDTATIS)

<p align="center">
  <img src="fig1.png" alt="UDTATIS Overview" width="800"/>
</p>

## Project Introduction

UDTATIS is an improved unsupervised deep image stitching system that combines the UDIS++ framework with EfficientLOFTR's feature extraction and matching capabilities. The system is specifically optimized for low-resolution images (such as terahertz images), introducing valid point discrimination, continuity constraints, and diffusion models to improve stitching quality.

### Key Features

- Unsupervised learning: No paired training data required
- Disparity tolerance: Able to handle image stitching scenarios with disparity
- Two-stage design: Separately handles geometric alignment and image fusion
- Efficient feature extraction: Uses EfficientLOFTR's feature extractor
- Precise feature matching: Introduces Transformer for feature matching
- Valid point discrimination: Automatically identifies and filters unreliable matching points
- Continuity constraint: Ensures spatial continuity of matching points
- Diffusion model: Uses diffusion model to optimize image fusion quality
- Adaptive normalization: Improves network training stability
- Multi-scale feature fusion: Enhances feature extraction capability

## Algorithm Architecture

The system is divided into two main stages:

### 1. Warp Stage

- **Feature Extraction**: Uses EfficientNet as backbone, combined with FPN for multi-scale feature extraction
- **Feature Matching**: Uses Transformer for feature matching, considering global context information
- **Valid Point Discrimination**:
  - Uses convolutional neural network to evaluate the reliability of each matching point
  - Generates training labels through homography transformation errors
  - Dynamically filters unreliable matching points
- **Continuity Constraint**:
  - Computes gradients of feature maps to constrain the continuity of matching points
  - Only considers continuity in valid point regions
  - Balances continuity with other loss terms
- **Warp Estimation**:
  - Global homography estimation: Uses regression network to estimate global transformation parameters
  - Local grid warp: Uses TPS (Thin Plate Spline) for local deformation

### 2. Composition Stage

- **Diffusion Model**:
  - Uses U-Net structure diffusion model for image fusion
  - Introduces time encoding and adaptive normalization
  - Supports multi-scale feature extraction and fusion
- **Attention Mechanism**:
  - Uses attention mechanism to enhance feature fusion
  - Adaptively adjusts feature weights
- **Residual Connection**:
  - Adds residual connections to improve gradient flow
  - Improves network training stability
- **Loss Functions**:
  - Boundary loss: Ensures smoothness of stitching boundaries
  - Smoothness loss: Ensures continuity of image content
  - Perceptual loss: Maintains high-level semantic information of images
  - Multi-scale loss: Considers feature matching at different scales
  - Dynamic weighting: Adaptively adjusts weights of various loss terms

## System Requirements

```bash
# Basic dependencies
numpy>=1.19.5
torch>=1.7.1
scikit-image>=0.15.0
tensorboard>=2.9.0
matplotlib>=3.5.1
torchvision>=0.8.2
opencv-python>=4.5.0
Pillow>=8.0.0
tqdm>=4.62.0

# Additional dependencies
efficientnet-pytorch
```

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/your-username/UDTATIS.git
cd UDTATIS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained weights:
- Download pre-trained weights from the EfficientLOFTR official repository
- Place the weight files in the `Warp/pretrained/` directory

## Dataset Preparation

Use the UDIS-D dataset for training and testing:
1. Download the UDIS-D dataset
2. Place the dataset in the `data/UDIS-D/` directory
3. Dataset structure:
```
data/UDIS-D/
  ├── training/
  │   ├── img1/
  │   └── img2/
  └── testing/
      ├── img1/
      └── img2/
```

## Data Flow Logic

In the UDTATIS system's workflow, a key part is the data flow from the Warp stage to the Composition stage. This process has been optimized in the code to ensure seamless transition from Warp to Composition:

### Warp to Composition Data Preparation

After the Warp model processes the original image pairs, it generates warped images and corresponding masks, which are saved to dedicated directories for the Composition stage:

1. **Data Storage Location**: All Warp-processed data is stored in the `data/UDIS-D/composition_data/` directory
2. **Directory Structure**:
   ```
   data/UDIS-D/composition_data/
   ├── train/             # Training dataset
   │   ├── warp1/         # Warped results of the first image
   │   ├── warp2/         # Warped results of the second image
   │   ├── mask1/         # Mask of the first image
   │   └── mask2/         # Mask of the second image
   └── test/              # Test dataset
       ├── warp1/
       ├── warp2/
       ├── mask1/
       └── mask2/
   ```

### Automatic Data Preparation Process

The system automatically performs data preparation in the following cases:

1. **When Training the Composition Module for the First Time**: When composition training data is detected to be missing, the system automatically calls the Warp model to process training image pairs
2. **When Testing the Composition Module for the First Time**: When composition test data is detected to be missing, the system automatically calls the Warp model to process test image pairs
3. **During End-to-End Testing**: In end2end mode, the system executes Warp testing and data preparation sequentially, then performs Composition testing

### Data Processing Flow

The specific data processing flow is as follows:

1. **Loading the Trained Warp Model**: The system prioritizes using the user-specified model; if not specified, it automatically finds the latest checkpoint
2. **Processing Original Image Pairs**: The Warp model performs the following operations on each image pair:
   - Extracts image features and performs matching
   - Estimates homography transformation and grid deformation
   - Applies transformations to generate warped images
   - Generates masks reflecting the warped regions
3. **Saving Processing Results**: Saves warped images and masks to specified directories
4. **Updating Configuration**: Automatically updates configuration for the Composition module to use newly generated data

### Recent Improvements to Composition Module

The Composition module has been significantly enhanced with the following improvements:

1. **Direct Image Processing**: The `apply_composition_mask_processing` function has been improved to process images directly rather than using subprocess calls, resulting in better performance and reliability.

2. **Testing and Debugging Tools**: Three specialized testing scripts have been developed:
   - `test_composition_directly.py`: Directly tests the `process_mask_for_composition` function
   - `test_composition_full.py`: Implements complete end-to-end testing with model prediction and post-processing
   - `test_composition_debug.py`: Provides detailed information during testing for debugging purposes

3. **Error Handling and Directory Management**: Proper error handling has been implemented, and the system now ensures all necessary directories are created, including the mask directory.

4. **Performance Monitoring**: All testing scripts now measure and report processing times, with tests showing:
   - Model prediction takes approximately 10 seconds per image
   - Post-processing is very fast (about 0.15 seconds)
   - Average total processing time is about 10 seconds per image

5. **Result Organization**: Processed images are now correctly saved in the appropriate directories, with merging results stored in the "merged" directory.

## Quick Start

Use main.py to easily train and test the entire system:

1. Train the entire system:
```bash
python main.py --mode train --part all
```

2. Train only the Warp part:
```bash
python main.py --mode train --part warp
```

3. Train only the Composition part (data will be prepared automatically):
```bash
python main.py --mode train --part composition
```

4. Test the entire system:
```bash
python main.py --mode test --part all
```

5. End-to-end testing (Warp results automatically input to Composition):
```bash
python main.py --mode end2end --model_path path/to/model
```

6. Only prepare Composition data without training:
```bash
python main.py --mode train --part composition --prepare_only
```

7. Prepare Composition data using a specific Warp model:
```bash
python main.py --mode train --part composition --prepare_only --model_path path/to/warp_model
```

8. Use a custom configuration file:
```bash
python main.py --config custom_config.json --mode train --part all
```

9. Test the system using virtual data (no real dataset needed):
```bash
python main.py --mode test --part all --virtual
```

## System Validation

Comprehensive system testing scripts are provided to validate various module functionalities:

```bash
# Test the entire system
python test_system.py

# Use virtual data
python test_system.py --virtual

# Use GPU acceleration for testing
python test_system.py --virtual --gpu
```

`test_system.py` will perform the following tests:
- Data loading and model functionality of the Warp module
- Data loading and model functionality of the Composition module
- Correctness of the main.py script
- Correctness of small-scale training loops

After successful testing, detailed functional verification results will be displayed to ensure all parts of the system are working properly.

## Composition Testing Tools

Several specialized tools are available for testing the Composition module:

1. **Basic Composition Testing**:
```bash
python test_composition_directly.py --data_dir data/UDIS-D/composition_data/test --output_dir Composition/direct_test_results
```

2. **Full Composition Testing with Model**:
```bash
python test_composition_full.py --model_path Composition/model/model_latest.pth --test_data data/UDIS-D/composition_data/test
```

3. **Interactive Debugging**:
```bash
python test_composition_debug.py --interactive --limit 5
```

These tools provide different levels of testing and debugging capabilities:
- `test_composition_directly.py`: Tests only the composition process without model prediction
- `test_composition_full.py`: Tests the complete pipeline including model prediction and composition
- `test_composition_debug.py`: Provides detailed logging and visualization for debugging purposes

## Model Visualization Tools

UDTATIS provides powerful visualization tools to help understand and debug the internal working mechanisms of both the Warp and Composition stages. These tools are located in the `draw` directory and capture the network's intermediate features and processing through a hook mechanism.

### Visualization Tools Overview

- **Warp Process Visualization**: Shows feature extraction, matching, valid point discrimination, and image warping process
- **Composition Process Visualization**: Shows the forward propagation, sampling process, and final fusion effect of the diffusion model
- **Feature Map Visualization**: Converts multi-channel features from intermediate layers into intuitive heat maps
- **Mask Visualization**: Shows fusion masks learned by the model
- **Comparison Visualization**: Shows side-by-side comparison of input images and processing results

### Usage

#### Visualize Warp Process

```bash
# Basic usage
python draw/visualize_warp.py

# Custom parameters
python draw/visualize_warp.py --image1_dir data/source --image2_dir data/target --output_dir results/warp_vis

# Using the helper script (more convenient)
bash draw/run_warp_visualization.sh
```

Main parameters:
- `--image1_dir`/`--image2_dir`: Input image directories
- `--output_dir`: Output result saving directory
- `--model_path`: Warp model weight path
- `--device`: Running device (cuda/cpu)

#### Visualize Composition Process

```bash
# Basic usage
python draw/visualize_composition.py --warp1 images/warp1.png --warp2 images/warp2.png --mode full

# Full parameters
python draw/visualize_composition.py --warp1 images/warp1.png --warp2 images/warp2.png --mask1 images/mask1.png --mask2 images/mask2.png --output_dir results/composition --mode sample --vis_steps 20

# Using the helper script (supports batch processing)
bash draw/run_composition_visualization.sh --warp1_dir images/warp1 --warp2_dir images/warp2 --mode sample
```

Main parameters:
- `--warp1`/`--warp2`: Paths of warped images
- `--mask1`/`--mask2`: Paths of corresponding masks (optional)
- `--mode`: Visualization mode (full/forward/sample)
- `--vis_steps`: Number of sampling steps to visualize
- `--low_memory`: Low memory mode

## Configuration Description

The system uses a config.json file for configuration, mainly including the following parts:

### Warp Configuration
```json
{
    "warp": {
        "train": {
            "gpu": "0",
            "batch_size": 8,
            "max_epoch": 100,
            "learning_rate": 1e-4,
            "train_path": "data/UDIS-D/training",
            "model_save_path": "Warp/model",
            "summary_path": "Warp/summary",
            "loss_weights": {
                "homography": 1.0,
                "mesh": 1.0,
                "feature": 0.1,
                "valid_point": 0.5,
                "continuity": 0.2
            }
        },
        "test": {
            "gpu": "0",
            "batch_size": 1,
            "test_path": "data/UDIS-D/testing",
            "result_path": "Warp/results",
            "model_path": "Warp/model/checkpoint_latest.pth"
        }
    }
}
```

### Composition Configuration
```json
{
    "composition": {
        "train": {
            "gpu": "0",
            "batch_size": 8,
            "max_epoch": 100,
            "learning_rate": 1e-4,
            "train_path": "data/UDIS-D/composition_data/train",
            "model_save_path": "Composition/model",
            "summary_path": "Composition/summary",
            "loss_weights": {
                "boundary": 1.0,
                "smooth": 1.0,
                "perceptual": 0.5,
                "multi_scale": 0.5,
                "diffusion": 1.0
            },
            "diffusion": {
                "num_timesteps": 1000,
                "beta_start": 1e-4,
                "beta_end": 0.02
            }
        },
        "test": {
            "gpu": "0",
            "batch_size": 1,
            "test_path": "data/UDIS-D/composition_data/test",
            "result_path": "Composition/results",
            "save_dirs": {
                "learn_mask1": "learn_mask1",
                "learn_mask2": "learn_mask2",
                "composition": "composition",
                "denoised": "denoised",
                "visualization": "visualization",
                "mask": "mask"
            }
        }
    }
}
```

## Advanced Usage

### Custom Loss Weights

Adjust loss weights to adapt to different scenarios, for example, increase the continuity loss weight:

```bash
python main.py --mode train --part warp --custom_loss_weights '{"continuity": 0.5}'
```

### Transfer Learning

Use pre-trained models to quickly adapt to new datasets:

```bash
python main.py --mode train --part all --pretrained path/to/model --learning_rate 1e-5
```

### Diffusion Model Parameter Adjustment

Modify diffusion process parameters to optimize image fusion quality:

```bash
python main.py --mode train --part composition --diffusion_timesteps 500 --beta_schedule 'linear'
```

## Result Evaluation

The system outputs the following evaluation metrics:
- PSNR (Peak Signal-to-Noise Ratio): Measures image quality
- SSIM (Structural Similarity Index): Measures structural similarity
- Valid point ratio: Measures the reliability of matching points
- Continuity loss: Measures the spatial continuity of matching points
- Diffusion quality assessment: Evaluates the denoising effect of the diffusion model
- Boundary smoothness: Evaluates the smoothness of stitching boundaries
- Visual quality assessment: Generates visualization results showing stitching effects

Output results will be saved in the directory specified in the configuration file, including:
- Warped images (Warp results)
- Valid point masks
- Learned fusion masks
- Stitched result images
- Images optimized by the diffusion model
- Visual comparison results

## Pre-trained Models

We provide pre-trained models for the two stages:
1. Warp stage model: [Download Link](https://drive.google.com/file/d/1GBwB0y3tUUsOYHErSqxDxoC_Om3BJUEt/view?usp=sharing)
2. Composition stage model: [Download Link](https://drive.google.com/file/d/1OaG0ayEwRPhKVV_OwQwvwHDFHC26iv30/view?usp=sharing)

## Citation

If you use this code, please cite the following paper:

```bibtex
@inproceedings{nie2023parallax,
  title={Parallax-Tolerant Unsupervised Deep Image Stitching},
  author={Nie, Lang and Lin, Chunyu and Liao, Kang and Liu, Shuaicheng and Zhao, Yao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7399--7408},
  year={2023}
}
```

## Contact

For any questions, please contact:
- Email: nielang@bjtu.edu.cn
- GitHub Issues: [Submit Issues](https://github.com/your-username/UDTATIS/issues)

## License

This project is licensed under the [MIT License](LICENSE). 
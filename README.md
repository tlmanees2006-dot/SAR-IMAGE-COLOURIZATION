# SAR Image Colorization

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![GitHub Stars](https://img.shields.io/github/stars/tlmanees2006-dot/SAR-IMAGE-COLOURIZATION.svg)](https://github.com/tlmanees2006-dot/SAR-IMAGE-COLOURIZATION/stargazers)

SAR (Synthetic Aperture Radar) image colorization using a U-Net architecture with ResNet encoder in PyTorch. This project converts grayscale SAR images to realistic RGB representations, enhancing interpretability for remote sensing applications.

## Features

- **U-Net with ResNet Encoder**: Deep learning architecture for semantic image colorization
- **PyTorch Implementation**: Modern deep learning framework
- **Jupyter Notebook Training**: Interactive training environment
- **SAR Image Processing**: Specialized for radar imagery colorization
- **Enhanced Interpretability**: Converts complex SAR data to visual RGB format

## Applications

- Remote sensing analysis
- Earth observation studies
- SAR image interpretation
- Environmental monitoring
- Research and educational purposes

## Architecture

The system uses a **U-Net with ResNet encoder** architecture:

1. **ResNet Encoder**: Extracts hierarchical features from grayscale SAR images
2. **U-Net Skip Connections**: Preserves spatial information across scales
3. **Decoder**: Reconstructs RGB images with enhanced visual quality
4. **Loss Functions**: Optimized for colorization tasks

## Requirements

### System Requirements
- Python 3.7 or higher
- CUDA-compatible GPU (recommended)
- 4GB+ RAM
- Sufficient storage for datasets

### Dependencies
```
torch
torchvision
numpy
opencv-python
Pillow
matplotlib
jupyter
```

## Installation

### Clone Repository
```bash
git clone https://github.com/tlmanees2006-dot/SAR-IMAGE-COLOURIZATION.git
cd SAR-IMAGE-COLOURIZATION
```

### Install Dependencies
```bash
pip install torch torchvision numpy opencv-python Pillow matplotlib jupyter
```

## Dataset Preparation

Prepare your SAR dataset with the following structure:
```
dataset/
├── grayscale/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── rgb/
    ├── image1.png
    ├── image2.png
    └── ...
```

### Supported Formats
- Input: `.png`, `.jpg`, `.jpeg`
- Output: `.png`, `.jpg`

## Training

### Using Jupyter Notebook
1. Open the training notebook:
```bash
jupyter notebook Train.ipynb
```

2. Follow the cells in the notebook to:
   - Load your dataset
   - Configure model parameters
   - Train the U-Net model
   - Monitor training progress
   - Save trained model

### Training Process
The `Train.ipynb` notebook includes:
- Data loading and preprocessing
- Model initialization (UNetWithResnetEncoder)
- Training loop with loss computation
- Validation and model checkpointing
- Visualization of results

## Inference

### Single Image Prediction
Use the trained model to colorize SAR images. The inference code is typically included in the notebook or can be implemented as:

```python
import torch
from model import UNetWithResnetEncoder
from PIL import Image
import numpy as np

# Load trained model
model = UNetWithResnetEncoder()
model.load_state_dict(torch.load('path/to/model.pth'))
model.eval()

# Load and preprocess image
# (Add your preprocessing code here)

# Inference
with torch.no_grad():
    colorized = model(input_tensor)

# Save result
# (Add your post-processing and saving code here)
```

## Model Performance

The U-Net with ResNet encoder provides:
- Effective feature extraction through ResNet backbone
- Spatial detail preservation via skip connections
- Stable training and convergence
- Good colorization quality for SAR imagery

## Project Structure

```
SAR-IMAGE-COLOURIZATION/
├── dataset/           # Dataset directory
├── model/            # Model definitions
├── utils/            # Utility functions
├── Train.ipynb       # Training notebook
└── README.md         # This file
```

## 🔧 Configuration

Model and training parameters can be adjusted in the `Train.ipynb` notebook:

- **Architecture**: UNetWithResnetEncoder
- **Learning Rate**: Configurable in training cells
- **Batch Size**: Adjustable based on GPU memory
- **Epochs**: Set according to dataset size
- **Loss Function**: Typically L1 or L2 loss for colorization

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Usage Rights

This project is available for educational and research purposes. Please ensure you comply with any applicable terms when using this code.

```bibtex
```

## Related Work

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [SAR Image Processing and Analysis](https://ieeexplore.ieee.org/)

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/tlmanees2006-dot/SAR-IMAGE-COLOURIZATION/issues)
- **Support**: Create an issue for questions or support

## Acknowledgments

- PyTorch team for the deep learning framework
- ResNet and U-Net architecture researchers
- Remote sensing and SAR imaging community

## Getting Started

1. **Clone the repository**
2. **Install dependencies** 
3. **Prepare your SAR dataset** in the required structure
4. **Open and run `Train.ipynb`** to train the model
5. **Use the trained model** for SAR image colorization

---

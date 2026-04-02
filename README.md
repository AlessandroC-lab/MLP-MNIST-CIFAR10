# MLP-MNIST-CIFAR10

Compact MLP classifiers for MNIST/CIFAR10 digit recognition (784→10 params).

## Performance
| Dataset  | Test Acc | Train Acc | Val Acc |
|----------|----------|-----------|---------|
| **MNIST** | **97.19%** | 98.30% | 97.08% |
| CIFAR10  | **49.84%** | 58.61% | 49.20% |

## Architecture
28×28→Flatten(784) → Dense(64,ReLU) → Drop(0.25)
→ Dense(128×3,ReLU) → Drop(0.25) → Dense(10,Softmax)
222k params | SGD(0.001,m=0.9) | 100 epochs

## Features
- **MLP baseline**: 97% MNIST, 50% CIFAR10 (color baseline)
- Keras Sequential + ModelCheckpoint
- Train/val/test confusion matrices
- Training curves + single-image test
- 90/10 split, batch=128

## Run
```bash
pip install tensorflow scikit-learn matplotlib jupyter
jupyter notebook *.ipynb
```

## Files
- `NN.ipynb`: **MNIST 97.19%**
- `NN_alternative.ipynb`: CIFAR10 49.84%
- `bestmdl.keras`: Saved models

**Complete coursework-ready MLP baseline (97% MNIST).**

# Dual-domain Modulation Network for Lightweight Image Super-Resolution

### [Paper (ArXiv)](https://arxiv.org/pdf/2503.10047?) | [Paper (IEEE)](https://arxiv.org/pdf/2503.10047?)

---
## Contents

The contents of this repository are as follows:

1. [Dependencies](#Dependencies)
2. [Train](#Train)
3. [Test](#Test)

## Dataset
We used only the first 800 images of <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">DIV2K</a> dataset to train our model.

The test set including Set5, Set14, B100, Urban100, Manga109, which can be downloaded from <a href="https://pan.baidu.com/s/1Vb68GWERriLmJRtYfm2uEg">here</a>.[Password:8888]

The code and datasets need satisfy the following structures:
```
├── DMNet  					# Train / Test Code
├── dataset  					# all datasets for this code
|  └── DIV2K_decoded  		#  train datasets with npy format
|  |  └── DIV2K_train_HR  		
|  |  └── DIV2K_train_LR_bicubic 			
|  └── benchmark  		#  test datasets with png format 
|  |  └── Set5
|  |  └── Set14
|  |  └── B100
|  |  └── Urban100
|  |  └── Manga109
 ─────────────────
```
---

## Results

Our SR Results can be downloaded from <a href="https://drive.google.com/file/d/1YT9f3cSDwhrEGP0k0bnntANRPfGrKM23/view?usp=drive_link">here</a>.

Pretrained models can be found in <a href="https://github.com/24wenjie-li/DMNet/tree/main/experiments">experiments</a>.

---

### Dependencies

> - torch 2.4.1
> - basicsr 1.4.2
> - cupy-cuda12x 12.3.0

```
# For install basicsr
pip install basicsr==1.4.2

python setup.py develop -i http://mirrors.aliyun.com/pypi/simple/

python -m pip install --upgrade pip

pip install numpy==1.24.4

pip install -v -e .
```

### Train

```
# For X2
torchrun --nproc_per_node=$GPU_NUM$ basicsr/train.py -opt options/train_DMNetx2.yml --launcher pytorch

# For X3
torchrun --nproc_per_node=$GPU_NUM$ basicsr/train.py -opt options/train_DMNetx3.yml --launcher pytorch

# For X4
torchrun --nproc_per_node=$GPU_NUM$ basicsr/train.py -opt options/train_DMNetx4.yml --launcher pytorch
```

### Test

```
# For X2
python basicsr/test.py -opt options/test_DMNetx2.yml

# For X3
python basicsr/test.py -opt options/test_DMNetx3.yml

# For X4
python basicsr/test.py -opt options/test_DMNetx4.yml
```

### Parmas and FLOPs

```
# Calculation DMNet
python inference.py
```
---

## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{li2025dual,
  title={Dual-domain modulation network for lightweight image super-resolution},
  author={Li, Wenjie and Guo, Heng and Hou, Yuefeng and Gao, Guangwei and Ma, Zhanyu},
  journal={IEEE Transactions on Multimedia},
  year={2025}
}
```

## Acknowledgement

The foundation for the training process is [BasicSR](https://github.com/XPixelGroup/BasicSR) , which profited from the outstanding contribution of [XPixelGroup](https://github.com/XPixelGroup) .

## Contact

This repo is currently maintained by lewj2408@gmail.com and is for academic research use only. 

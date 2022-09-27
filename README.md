
# SAPA: Similarity-Aware Point Affiliation for Feature Upsampling

<p align="center"><img src="upsampled_feat.png" width="450" title="SAPA"/></p>

This repository includes the official implementation of SAPA, a universal feature upsampling operator, presented in our paper:

**[SAPA: Similarity-Aware Point Affiliation for Feature Upsampling](https://arxiv.org/abs/2209.12866)**

Proc. Annual Conference on Neural Information Processing Systems (NeurIPS 2022)

[Hao Lu](https://sites.google.com/site/poppinace/), Wenze Liu, Zixuan Ye, Hongtao Fu, Yuliang Liu, Zhiguo Cao

Huazhong University of Science and Technology, China

## Highlights
- **Simple and effective:** SAPA defines a novel class of similarity-aware upsampling kernels, which can simultaenously encourage semantic smoothness and boundrary sharpness;
- **Generic:** SAPA can benefit a number of dense prediction tasks. In our paper, we validate semantic segmentation, object detection, depth estimation, and image matting;
- **Plug-and-play:** SAPA is applicable to any encoder-decoder architectures or feature pyramid networks;
- **Lightweight:** SAPA only introduces negligible extra computational overhead and number of parameters.
- **Memory-efficient:** We have provided a CUDA implementation to reduce memory cost and to improve efficiency.

<p align="center"><img src="qualitative_results.png" width="450" title="SAPA"/></p>

## Installation
Our codes are tested on Python 3.8.8 and PyTorch 1.9.0.
```shell
cd sapa
python setup.py develop
```

## Reported Results

Here is a brief summary of our reported results in the paper:
#### Semantic Segmentation on ADE20K

| Segformer B1  | mIoU  | FLOPs     | Params    | MaskFormer SwinBase   | mIoU  | FLOPs  | Params   | Mask2Former SwinBase   | mIoU  | FLOPs  | Params   |
| :--:          | :--:  | :--:      | :--:      | :--:                  | :--:  | :--:   | :--:     | :--:                  | :--:  | :--:   | :--:     | 
| Nearst        | --    | --        | --        | Nearst                | 52.70 | 195    | 102      | Nearst        | --    | --        | --        | 
| Bilinear      | 41.68 | 15.91     | 13.74     | Bilinear              | --    | --     | --       | Bilinear      | 53.90 | 223     | 107     | 
| CARAFE        | 42.82 | +1.45     | +0.44     | CARAFE                | *53.53* | +0.84  | +0.22  | CARAFE        | 53.94 | +0.63     | +0.07     | 
| IndexNet      | 41.50 | +30.65    | +12.60    | IndexNet              | 52.92 | +17.64 | +6.30    | IndexNet      | 54.71 | +13.44    | +2.10    | 
| A2U           | 41.45 | +0.41     | +0.06     | A2U                   | 52.73 | +0.23  | +0.03    | A2U           | 54.40 | +0.18     | +0.01     | 
| SAPA-I        | 43.05 | +0.75     | +0.00     | SAPA-I                | 53.25 | +0.43  | +0.00    | SAPA-I                | *55.05* | +0.33  | +0.00    |
| SAPA-B        | *43.20* | +1.02   | +0.1      | SAPA-B                | 53.15 | +0.59  | +0.05    | SAPA-B                | 54.98 | +0.45  | +0.02    |
| SAPA-G        | **44.39** | +1.02 | +0.1      | SAPA-G                | **53.78** | +0.59 | +0.05 | SAPA-G                | **55.22** | +0.45 | +0.02 |
-**Notes:** we observe that MaskFormer and Mask2Former are somewhat unstable. Due to limited resources, we only report one-run results. 

#### Image Matting on Adode Composition-1K
| A2U Matting | Params  | SAD  | MSE | Grad | Conn |
| :---:        | :---:    | :---: | :---:| :---: | :---: |
Nearest | 8.05 | 37.51 | 0.0096 | 19.07 | 35.72 |
Bilinear | 8.05 | 37.31 | 0.0103 | 21.38 | 35.39 |
CARAFE | +0.26 | 41.01 | 0.0118 | 21.39 | 39.01 |
IndexNet | +12.26 | 34.28 | 0.0081 | 15.94 | 31.91 |
A2U | +0.02 | 32.15 | 0.0082 | 16.39 | 29.25 |
SAPA-I | +0 | 34.25 | 0.0091 | 18.93 | 32.09 |
SAPA-B | +0.04 | 31.19 | 0.0079 | **15.48** | 28.30 |
SAPA-G | +0.04 | **30.98** | **0.0077** | 15.59 | **27.96** |


## Citation
If you find this work or code useful for your research, please cite:
```
@inproceedings{lu2022sapa,
  title={SAPA: Similarity-Aware Point Affiliation for Feature Upsampling},
  author={Lu, Hao and Liu, Wenze and Ye, Zixuan and Fu, Hongtao and Liu, Yuliang and Cao, Zhiguo},
  booktitle={Proc. Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

## Permission
This code is for academic purposes only. Contact: Hao Lu (hlu@hust.edu.cn)

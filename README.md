# MHNet
This repo is the official implementation of [Multi-View Higher-Order Neural Network for Diagnosing Neurodevelopmental Disorders using rs-fMRI]([https://link.springer.com/article/10.1007/s10278-025-01399-5])

## I. Usage:
MHNet is a dual-branch deep learning framework that analyzes rs-fMRI data to diagnose neurodevelopmental disorders like ASD and ADHD. It uniquely extracts both Euclidean features (local and high-order) and non-Euclidean features (topological and high-order) from multi-view brain functional networks. Experiments across three public datasets demonstrate MHNet outperforms existing methods, offering improved diagnostic capability and insights into brain region associations with neurodevelopmental disorders.

The data used in our work are from [ADNI](https://adni.loni.usc.edu/) and [ABIDE](http://preprocessed-connectomes-project.org/abide/). Please follow the relevant regulations to download from the websites.

## II. Requirements:
* numpy~=1.26.2
* scikit-learn~=1.2.2
* scipy~=1.10.1
* torch~=2.0.0
* torch-cluster~=1.6.0
* torch-geometric~=2.0.4
* torch-scatter~=2.0.9
* torch-sparse~=0.6.13
* torch-spline-conv~=1.2.1
## III. Citation：
If our paper or code is helpful to you, please cite our paper. If you have any questions, please feel free to ask me.
```
@article{li2025mhnet,
  title={MHNet: Multi-view High-Order Network for Diagnosing Neurodevelopmental Disorders Using Resting-State fMRI},
  author={Li, Yueyang and Zeng, Weiming and Dong, Wenhao and Cai, Luhui and Wang, Lei and Chen, Hongyu and Yan, Hongjie and Bian, Lingbin and Wang, Nizhuan},
  journal={Journal of Imaging Informatics in Medicine},
  pages={1--21},
  year={2025},
  publisher={Springer}
}
```

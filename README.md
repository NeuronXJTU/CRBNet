# [TCSVT 2024] CRBNet
This repo is the official implementation of [Ultrasound Nodule Segmentation Using Asymmetric Learning with Simple Clinical Annotation](https://ieeexplore.ieee.org/abstract/document/10508987) which is accepted at IEEE Transactions on Circuits and Systems for Video Technology(TCSVT).

## Requirements
Some important required packages include:
* Pytorch version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

## Usage
1. Clone the repo:
```
git clone https://github.com/NeuronXJTU/CRBNet.git
cd CRBNet
```
2. Download the processed data and put the data in `../data/node_data_thyroid` or `../data/node_data_breast`. The folder should be organized as follows:
```shell
./data/
├── node_data_thyroid
│   ├── imgs
│   ├── masks
│   ├── train_fold1.txt
│   ├── test_fold1.txt
│   ├── only_ellipse_all
│   ├── only_poly_all
│   ├── only_axis_all
│   ├── ...
├── node_data_breast
│   ├── imgs
│   ├── masks
│   ├── train_fold1.txt
│   ├── test_fold1.txt
│   ├── only_ellipse_all
│   ├── only_poly_all
│   ├── only_axis_all
│   ├── ...
```
The original image is in the imgs folder, the ground truth is in the masks folder, and the clinical annotations are in the only_axis_all folder. The only_ellipse_all folder contains the elliptic pseudo-labels generated based on clinical annotations. 

🔥🔥🔥 The **preprocessed thyroid and breast dataset** is available for downloading. Please contact zhaoxingyue@stu.xjtu.edu.cn or zhongyuli@xjtu.edu.cn for data. Please contact zhaoxingyue@stu.xjtu.edu.cn if you have any questions about the code.

3. Training & Testing

Run the following commands for training, testing and evaluating.
```shell
python train_IDAMP.py  # Our proposed method
python train_baseline.py # Use ground truth for training
python test_2D_node.py   # Test
```

## Cite
If this code is helpful for your research or you use our data，please cite:
```
@article{zhao2024ultrasound,
  title={Ultrasound Nodule Segmentation Using Asymmetric Learning with Simple Clinical Annotation},
  author={Zhao, Xingyue and Li, Zhongyu and Luo, Xiangde and Li, Peiqi and Huang, Peng and Zhu, Jianwei and Liu, Yang and Zhu, Jihua and Yang, Meng and Chang, Shi and others},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```


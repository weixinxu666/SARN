# SARN-Pytorch
Official pytorch codes and models for paper:\
[SARN: A Lightweight Stacked Attention Residual Network for Low-Light Image Enhancement](https://ieeexplore.ieee.org/abstract/document/9657795) \
2021 IEEE International Conference on Robotics and Automation Engineering (ICRAE 2021)


# Pretrained Model & Datasets
The pretrained models can be found in folder **/checkpoint**. \
We trained the models on the [LOL dataset](https://pan.baidu.com/s/1Div2cRLHWTUiYT6-vzkOrg) (password: gjhm). These two models in the folder **/checkpoint** were trained on the LOL real world dataset. Your can trained the models on the LOL synthetic dataset on your own.\
We tested our model on four datasets without Ground-Truth: [DICM, LIME, MEF, NPE, VV](https://pan.baidu.com/s/1utYsLd35dfQ3HZoR-Q5XBQ) (Password: p8vy).

Put the downloaded datasets in the folder **/data**.


# Training
You can train the model by runing **train_sarn_se.py** or **train_sarn_se_bam.py**. Please change the dir path of data in **dataset_lol.py** and **train_.py** before the training.


# Testing
You can test the pre-trained models on the lol eval datasets or your own data by runing **eval_sarn_se.py** or **eval_sarn_se_bam.py**. Please change the dir path of data before the testing.


# Model architecture
![Model](/pic/model.png)


# Experiments
## Qualitative
![lol_results](/pic/lol_results.png)

## Quantitative
![lol](/pic/lol.png)


## Speed
![speed](/pic/speed.png)


# Bugs (Sorry about that)
There are some bugs when you eval the model with BAM module (run **train_sarn_se_bam.py** and **eval_sarn_se_bam.py**). Due to a bug, we can only eval the images with Batchsize=1. And an image in your data can not be outputed.


# Requirements

````
pytorch==1.7+cuda10.1
torchvision==0.6.0
numpy==1.19.5
opencv-python-headless==4.5.5.92
tqdm==4.62.2
````

# Reference
There are other two papers of ours about low-light image enhancement:

[Da-drn: Degradation-aware deep retinex network for low-light image enhancement](https://arxiv.org/abs/2110.01809)

[Tsn-ca: A two-stage network with channel attention for low-light image enhancement](https://arxiv.org/abs/2110.02477)


# Citation
If you use this code for your research, please cite the following [paper](https://ieeexplore.ieee.org/abstract/document/9657795).

````
@inproceedings{wei2021sarn,
  title={SARN: A Lightweight Stacked Attention Residual Network for Low-LightImage Enhancement},
  author={Wei, Xinxu and Zhang, Xianshi and Li, Yongjie},
  booktitle={2021 6th International Conference on Robotics and Automation Engineering (ICRAE)},
  pages={275--279},
  year={2021},
  organization={IEEE}
}
````

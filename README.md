# PanelPose
This are the official source code and datasets for ***PanelPose: A 6D Panel Object Pose Estimation for Robotic Panel Inspection***.  



<img src="image/first.png" width="300" height="300"/><br/>


## Introduction
- **We'll upload the code and dataset step by step.** Now,the repository includes keypoint selection method (Edge-FPS and FLD-FPS) in the **tools** file, and the render code in the **lib** file. The rest of the code will be uploaded later.
- The dataset will be published in [dataset](https://cowtransfer.com/s/d198a5118fe34e) (code: xhbn26)
- To address the problem of highly-variable panel pose estimation, we propose a simple yet effective method denoted as PanelPose that explicitly takes the extra feature maps along with RGB image as CNN input. We extract edge and line features of RGB image and fuse these feature maps as a multi-feature fusion map (MFF Map). Moreover, at the output representation stage, we design a simple but effective keypoint selection algorithm considering the shape information of panel objects, which simplifies keypoint localization for precise pose estimation.

## Installation
- Install CUDA10.1, torch==1.6.0, torchvision==0.7.0
- Set up python environment from requirement.txt:
  ```shell
  pip3 install -r requirement.txt 
  ```

## Datasets
- The dataset includes model with texture, synthetic data, and real-world data.



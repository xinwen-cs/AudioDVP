## Hierarchical Cross-modal Talking Face Generation with Dynamic Pixel-wise Loss （ATVGnet）

By [Lele Chen](http://www.cs.rochester.edu/u/lchen63/) ,
[Ross K Maddox](https://www.urmc.rochester.edu/labs/maddox.aspx),
[ Zhiyao Duan](http://www2.ece.rochester.edu/~zduan/),
[Chenliang Xu](https://www.cs.rochester.edu/~cxu22/).

University of Rochester.

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Running](#running)
0. [Model](#model)
0. [Results](#results)
0. [Disclaimer and known issues](#disclaimer-and-known-issues)

### Introduction

This repository contains the original models (AT-net, VG-net) described in the paper [Hierarchical Cross-modal Talking Face Generation with Dynamic Pixel-wise Loss](http://www.cs.rochester.edu/u/lchen63/cvpr2019.pdf). The **demo video** is avaliable at https://youtu.be/eH7h_bDRX2Q. This code can be applied directly in [LRW](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) and [GRID](http://spandh.dcs.shef.ac.uk/gridcorpus/). The outputs from the model are visualized here: the first one is the synthesized landmark from ATnet, the rest of them are attention, motion map and final results from VGnet.

![model](https://github.com/lelechen63/ATVGnet/blob/master/img/visualization.gif)
![model](https://github.com/lelechen63/ATVGnet/blob/master/img/example.jpg)


### Citation

If you use any codes, models or the ideas from this repo in your research, please cite:

	@inproceedings{chen2019hierarchical,
	  title={Hierarchical cross-modal talking face generation with dynamic pixel-wise loss},
	  author={Chen, Lele and Maddox, Ross K and Duan, Zhiyao and Xu, Chenliang},
	  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  pages={7832--7841},
	  year={2019}
	}
### Running

0. This code is tested under Python 2.7. The model we provided is trained on LRW. However, it works fine on GRID,VOXCELB and other datasets. You can directly compare this model on other dataset with your own model. We treat this as fair comparison.
0. Pytorch environment:[Pytorch 0.4.1](https://pytorch.org/). (conda install pytorch=0.4.1 torchvision cuda90 -c pytorch)
0. Install requirements.txt (pip install -r requirement.txt)
0. Download the pretrained ATnet and VGnet weights at [google drive](https://drive.google.com/drive/folders/1WYhqKBFX6mLtdJ8sYVLdWUqp5FJDmphg?usp=sharing). Put the weights under `model` folder.
0. Run the demo code: `python demo.py`
	- `-device_ids`: gpu id
	- `-cuda`: using cuda or not
	- `-vg_model`: pretrained VGnet weight
	- `-at_model`: pretrained ATnet weight
	- `-lstm`:  use lstm or not
	- `-p`:  input example image
	- `-i`:  input audio file
	- `-lstm`:  use lstm or not
	- `-sample_dir`: folder to save the outputs
	- ...
0. Download and unzip the training data from [LRW](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)
0. Preprocess the data (Extract landmark and crop the image by dlib).
0. Train the ATnet model:  `python atnet.py`
	- `-device_ids`: gpu id
	- `-batch_size`: batch size 
	- `-model_dir`: folder to save weights
	- `-lstm`:  use lstm or not
	- `-sample_dir`: folder to save visualized images during training
	- ...


0. Test the model: `python atnet_test.py`
	- `-device_ids`: gpu id
	- `-batch_size`: batch size
	- `-model_name`: pretrained weights
	- `-sample_dir`: folder to save the outputs
	- `-lstm`:  use lstm or not
	- ...
0. Train the VGnet:	`python vgnet.py`
	- `-device_ids`: gpu id
	- `-batch_size`: batch size 
	- `-model_dir`: folder to save weights
	- `-sample_dir`: folder to save visualized images during training
	- ...
0. Test the VGnet: 	`python vgnet_test.py`
	- `-device_ids`: gpu id
	- `-batch_size`: batch size
	- `-model_name`: pretrained weights
	- `-sample_dir`: folder to save the outputs
	- ...

### Model

0. Overall ATVGnet
	![model](https://github.com/lelechen63/ATVGnet/blob/master/img/generator.png)

	
0. Regresssion based discriminator network

	![model](https://github.com/lelechen63/ATVGnet/blob/master/img/regress-disc.jpg)
	
	

### Results

0. Result visualization on different datasets:

	![visualization](https://github.com/lelechen63/ATVGnet/blob/master/img/compare.jpg)

0. Reuslt compared with other SOTA methods:

	![visualization](https://github.com/lelechen63/ATVGnet/blob/master/img/visualresults.jpg)

0. The studies on image robustness respective with landmark accuracy:

	![visualization](https://github.com/lelechen63/ATVGnet/blob/master/img/noise.png)

0. Quantitative results:

	![visualization](https://github.com/lelechen63/ATVGnet/blob/master/img/userstudy.jpg)
	

### Disclaimer and known issues

0. These codes are implmented in Pytorch.
0. In this paper, we train LRW and GRID seperately. 
0. The model are sensitive to input images. Please use the correct preprocessing code.
0. I didn't finish the data processing code yet. I will release it soon. But you can try the model and replace with your own image.
0. If you want to train these models using this version of pytorch without modifications, please notice that:
	- You need at lest 12 GB GPU memory.
	- There might be some other untested issues.
0. There is another intresting and useful research on audio to landmark genration. Please check it out at https://github.com/eeskimez/Talking-Face-Landmarks-from-Speech.	
### Todos

 - Release training data

License
----

MIT

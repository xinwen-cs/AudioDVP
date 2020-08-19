AudioDVP

This is the official implementation of Photorealistic Audio-driven Video Portraits.

Major Requirements

Ubuntu >= 18.04
PyTorch >= 1.2
NVCC >= 10.1
FFmpeg (with H.264 support)

Detailed environment setup of mine is in enviroment.yml for reference.

Major implementation difference against original paper

Geometry and texture parameter of 3DMM is now initialized from zero and shared among all samples during fitting,
because it is not reasonable to average 8 samples after preparing stage.

Using OpenCV other than PIL for image editing operation.

Usage

Download data(This is the most tedious step:()

Download BFM2009 from https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model.(Re-distribution is not allowed.)
Donwload expression basis from https://github.com/Juyong/3DFace.

Put the data in renderer/data like the structure below.

renderer/data
├── 01_MorphableModel.mat
├── BFM_exp_idx.mat
├── BFM_front_idx.mat
├── Exp_Pca.bin
├── facemodel_info.mat
├── select_vertex_id.mat
├── std_exp.txt
└── data.mat


Build data

cd renderer/
python build_data.py

Download Trump speech video from https://www.youtube.com/watch?v=6a1Mdq8-_wo and put it in data/video.




Compile CUDA kernel

cd renderer/kernels
python setup.py build_ext --inplace



Running demo script(Explanation of every step is provided.)


./scripts/demo.sh



Acknowledgment

This work is build upon many great open source code and data.

ATVGnet in the vendor directory is directly borrowed from https://github.com/lelechen63/ATVGnet under MIT License.

neural-face-renderer in the vendor directory is heavily borrowed from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix under BSD License.

The pre-trained ResNet model on VGGFace2 dataset is from https://github.com/cydonia999/VGGFace2-pytorch under MIT License.
We use resnet50_ft from https://drive.google.com/open?id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU.

Basel2009 3D face dataset is from https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model.

The expression basis of 3DMM is from https://github.com/Juyong/3DFace under GPL License.

Our renderer is heavily borrowed from https://github.com/google/tf_mesh_renderer and inspired by https://github.com/andrewkchan/pytorch_mesh_renderer.





Disclaimer

We made this code publicly available to benefit graphics and vision community.
Please DO NOT abuse the code for devil things.



Citation



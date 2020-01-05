# DCARN: 


# Code Structure
* `./grouping/*` - Partition code (geometric partitioning and superpoint graph construction)
* `./segmentation/*` - Learning code (superpoint embedding and contextual segmentation).

# Pre-requisites
This code is tested on Ubuntu 16.04 LTS with CUDA 8.0 and Pytorch 0.4.1.
1. Install [PyTorch](https://pytorch.org) and [torchnet](https://github.com/pytorch/tnt) with following commands: `pip install git+https://github.com/pytorch/tnt.git@master`.

2. You need to install some additional Python packages too: `pip install future python-igraph tqdm transforms3d pynvrtc fastrlock cupy h5py sklearn plyfile scipy`.

3. Install Boost (1.63.0 or later) and Eigen3, in Conda: `conda install -c anaconda boost; conda install -c omnia eigen3; conda install eigen; conda install -c r libiconv`.

4. Clone [cut pursuit repository](https://github.com/loicland/cut-pursuit) in `/partition` folder

5. Compile the ```libply_c``` and ```libcp``` libraries:
```
cd partition/ply_c
cmake . -DPYTHON_LIBRARY=$CONDAENVPath/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENVPath/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENVPath/include -DEIGEN3_INCLUDE_DIR=$CONDAENVPath/include/eigen3
make
cd ..
cd cut-pursuit/src
cmake . -DPYTHON_LIBRARY=$CONDAENVPath/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENVPath/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENVPath/include -DEIGEN3_INCLUDE_DIR=$CONDAENVPath/include/eigen3
make
```
where `$CONDAENVPath` is the path to your conda environment. 


# Usage
# Acknowledgements
The code for Training, evaluation and visualize is borrowed from [SPG] (https://github.com/loicland/superpoint_graph/tree/release)

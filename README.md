# DCARN: 


# Code Structure
* `./grouping/*` -  Geometric grouping and superpoint graph construction
* `./segmentation/*` - Superpoint embedding and segmentation.

# Pre-requisites
This code is tested on Ubuntu 16.04 LTS with CUDA 8.0 and Pytorch 0.4.1.
1. Install [PyTorch](https://pytorch.org) and [torchnet](https://github.com/pytorch/tnt) with following commands: `pip install git+https://github.com/pytorch/tnt.git@master`.

2. You need to install some additional Python packages too: `pip install future python-igraph tqdm transforms3d pynvrtc fastrlock cupy h5py sklearn plyfile scipy`.

3. Install Boost (1.63.0 or later) and Eigen3, in Conda: `conda install -c anaconda boost; conda install -c omnia eigen3; conda install eigen; conda install -c r libiconv`.

4. Clone [cut pursuit repository](https://github.com/loicland/cut-pursuit) in `/grouping` folder

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
## Stanford Large-Scale 3D Indoor Space
Download [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html) and extract `Stanford3dDataset_v1.2_Aligned_Version.zip` to `$S3DIR_DIR/data`, where `$S3DIR_DIR` is dataset directory.

To fix some issues with the dataset, apply path S3DIS_fix.diff with: cp S3DIS_fix.diff $S3DIR_DIR/data/s3dir/data; cd $S3DIR_DIR/data/s3dir/data; git apply S3DIS_fix.diff; rm S3DIS_fix.diff; cd -

### Grouping

To compute the groupings, run

```python grouping/partition.py --dataset s3dis --ROOT_PATH $S3DIR_DIR --voxel_width 0.03 --reg_strength 0.03```

### Training

First, reorganize point clouds into superpoints by:

```python segmentation/s3dis_dataset.py --S3DIS_PATH $S3DIR_DIR```

To train on the all 6 folds, run
```
for FOLD in 1 2 3 4 5 6; do \
CUDA_VISIBLE_DEVICES=0 python segmentation/main.py --dataset s3dis --S3DIS_PATH $S3DIR_DIR --cvfold $FOLD --epochs 350 --lr_steps '[275,320]' \
--test_nth_epoch 50 --model_config 'gru_10_0,f_13' --ptn_nfeat_stn 14 --nworkers 2 --odir "results/s3dis/best/cv${FOLD}"; \
done
```


To test this network on the full test set, run
```
for FOLD in 1 2 3 4 5 6; do \
CUDA_VISIBLE_DEVICES=0 python segmentation/main.py --dataset s3dis --S3DIS_PATH $S3DIR_DIR --cvfold $FOLD --epochs -1 --lr_steps '[275,320]' \
--test_nth_epoch 50 --model_config 'gru_10_0,f_13' --ptn_nfeat_stn 14 --nworkers 2 --odir "results/s3dis/best/cv${FOLD}" --resume RESUME; \
done
```

To visualize the results and all intermediary steps, use the visualize function in grouping directory as under:
```
python grouping/visualize.py --dataset s3dis --ROOT_PATH $S3DIR_DIR --res_file 'models/cv1/predictions_val' --file_path 'Area_1/conferenceRoom_1' --output_type igfpres
```

```output_type``` is defined as:
- ```'i'``` = input point cloud
- ```'g'``` = ground truth (if available), with the predefined class to color mapping
- ```'f'``` = geometric features with color code: red = linearity, green = planarity, blue = verticality
- ```'p'``` = groupings, with a random color for each superpoint
- ```'r'``` = result cloud, with the predefined class to color mapping
- ```'e'``` = error cloud, with green/red hue for correct/faulty prediction 
- ```'s'``` = superedge structure of the superpoint (toggle wireframe on meshlab to view it)

Add option ```--upsample 1``` if you want the prediction file to be on the original, unpruned data.


## Semantic3D
Download all point clouds and labels from [Semantic3D Dataset](http://www.semantic3d.net/) and place extracted training files to `$SEMA3D_DIR/data/train` and reduced test files into `$SEMA3D_DIR/data/test_reduced` where `$SEMA3D_DIR` is set to dataset directory. The label files of the training files must be put in the same directory.

### Grouping

To compute the grouping run

```python grouping/partition.py --dataset sema3d --ROOT_PATH $SEMA3D_DIR --voxel_width 0.05 --reg_strength 0.8 --ver_batch 5000000```

The code is tested on machine having 24GB of RAM. You can increase ```voxel_width``` parameter which increases pruning but it will effect model accuracy.

### Training

First, reorganize point clouds into superpoints by:

```python segmentation/sema3d_dataset.py --SEMA3D_PATH $SEMA3D_DIR```

To train and test the model, run the following
```
CUDA_VISIBLE_DEVICES=0 python segmentation/main.py --dataset sema3d --SEMA3D_PATH $SEMA3D_DIR --db_test_name testred --db_train_name trainval \
--epochs 500 --lr_steps '[350, 400, 450]' --test_nth_epoch 100 --model_config 'gru_10,f_8' --ptn_nfeat_stn 11 \
--nworkers 2 --pc_attrib xyzrgbelpsv --odir "results/sema3d/trainval_best"
```
To upsample the prediction to the unpruned data and write the .labels files for the reduced test set, run:

```python segmentation/write_Semantic3d.py --SEMA3D_PATH $SEMA3D_DIR --odir "results/sema3d/trainval_best" --db_test_name testred```

To visualize the results and intermediary steps (on the subsampled graph), use the visualize function in partition. For example:
```
python segmentation/visualize.py --dataset sema3d --ROOT_PATH $SEMA3D_DIR --res_file 'results/sema3d/trainval_best/prediction_testred' --file_path 'test_reduced/MarketplaceFeldkirch_Station4' --output_type ifprs
```

avoid ```--upsample 1``` as it can can take a very long time on the largest clouds.

Trained models for both datasets can be found at (https://github.com/saba155/DCARN_trained_models)

# Acknowledgements
The code for Training, evaluation and visualization is borrowed from [SPG] (https://github.com/loicland/superpoint_graph/tree/release)

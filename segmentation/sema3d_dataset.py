
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import spg


def get_datasets(args, test_seed_offset=0):

    train_names = ['bildstein_station1', 'bildstein_station5', 'domfountain_station1', 'domfountain_station3', 'station1_xyz', 'sg27_station1', 'sg27_station2', 'sg27_station5', 'sg27_station9', 'sg28_station4', 'untermaederbrunnen_station1']
    valid_names = ['bildstein_station3', 'domfountain_station2', 'sg27_station4', 'untermaederbrunnen_station3']

    if args.db_train_name == 'train':
        trainset = ['train/' + f for f in train_names]
    elif args.db_train_name == 'trainval':
        trainset = ['train/' + f for f in train_names + valid_names]

    if args.db_test_name == 'val':
        testset = ['train/' + f for f in valid_names]
    elif args.db_test_name == 'testred':
        testset = ['test_reduced/' + os.path.splitext(f)[0] for f in os.listdir(args.SEMA3D_PATH + '/DCARN/test_reduced')]
    elif args.db_test_name == 'testfull':
        testset = ['test_full/' + os.path.splitext(f)[0] for f in os.listdir(args.SEMA3D_PATH + '/DCARN/test_full')]

    # Load superpoints graphs
    testlist, trainlist = [], []
    for n in trainset:
        trainlist.append(spg.spg_reader(args, args.SEMA3D_PATH + '/DCARN/' + n + '.h5', True))
    for n in testset:
        testlist.append(spg.spg_reader(args, args.SEMA3D_PATH + '/DCARN/' + n + '.h5', True))

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist = spg.scaler01(trainlist, testlist)

    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.SEMA3D_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.SEMA3D_PATH, test_seed_offset=test_seed_offset))

def get_info(args):
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1

    return {
        'node_feats': 11 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'classes': 8,
        'inv_class_map': {0:'terrain_man', 1:'terrain_nature', 2:'veget_hi', 3:'veget_low', 4:'building', 5:'scape', 6:'artefact', 7:'cars'},
    }



def preprocess_pointclouds(SEMA3D_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""

    for n in ['train', 'test_reduced', 'test_full']:
        pathP = '{}/parsed/{}/'.format(SEMA3D_PATH, n)
        pathD = '{}/features/{}/'.format(SEMA3D_PATH, n)
        pathC = '{}/superpoint_graphs/{}/'.format(SEMA3D_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(0)

        for file in os.listdir(pathC):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')
                xyz = f['xyz'][:]
                rgb = f['rgb'][:].astype(np.float)
                elpsv = np.stack([ f['xyz'][:,2][:], f['linearity'][:], f['planarity'][:], f['scattering'][:], f['verticality'][:] ], axis=1)

                # rescale to [-0.5,0.5]; keep xyz
                elpsv[:,0] /= 100 # (rough guess)
                elpsv[:,1:] -= 0.5
                rgb = rgb/255.0 - 0.5

                P = np.concatenate([xyz, rgb, elpsv], axis=1)

                f = h5py.File(pathC + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(pathP + file, 'w') as hf:
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='DCARN')
    parser.add_argument('--SEMA3D_PATH', default='datasets/semantic3d')
    args = parser.parse_args()
    preprocess_pointclouds(args.SEMA3D_PATH)

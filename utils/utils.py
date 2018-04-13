#!/usr/bin/env python2.7
# Li Ding
# Mar. 2018
# Modified from https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/utils.py


from __future__ import division

import matplotlib.pylab as plt
import numpy as np
import os
import scipy.io as sio
import cv2
import glob
from tqdm import tqdm

plt.rcParams['figure.figsize'] = [10., 10.]


# ------------- Visualization -------------
def imshow_(x, **kwargs):
    if x.ndim == 2:
        plt.imshow(x, interpolation="nearest", **kwargs)
    elif x.ndim == 1:
        plt.imshow(x[:, None].T, interpolation="nearest", **kwargs)
        plt.yticks([])
    plt.axis("tight")


# ------------- Data -------------
def mask_data(X, Y, max_len=None, mask_value=0):
    if max_len is None:
        max_len = np.max([x.shape[0] for x in X])
    X_ = np.zeros([len(X), max_len, X[0].shape[1]]) + mask_value
    Y_ = np.zeros([len(X), max_len, Y[0].shape[1]]) + mask_value
    mask = np.zeros([len(X), max_len])
    for i in range(len(X)):
        l = X[i].shape[0]
        X_[i, :l] = X[i]
        Y_[i, :l] = Y[i]
        mask[i, :l] = 1
    return X_, Y_, mask[:, :, None]


# Unmask data
def unmask(X, M):
    if X[0].ndim == 1 or (X[0].shape[0] > X[0].shape[1]):
        return [X[i][M[i].flatten() > 0] for i in range(len(X))]
    else:
        return [X[i][:, M[i].flatten() > 0] for i in range(len(X))]


def match_lengths(X, Y, n_feat):
    # Check lengths of data and labels match
    if X[0].ndim == 1 or (X[0].shape[0] == n_feat):
        for i in range(len(Y)):
            length = min(X[i].shape[1], Y[i].shape[0])
            X[i] = X[i][:, :length]
            Y[i] = Y[i][:length]
    else:
        for i in range(len(Y)):
            length = min(X[i].shape[0], Y[i].shape[0])
            X[i] = X[i][:length]
            Y[i] = Y[i][:length]

    return X, Y


def remap_labels(Y_all):
    # Map arbitrary set of labels (e.g. {1,3,5}) to contiguous sequence (e.g. {0,1,2})
    ys = np.unique([np.hstack([np.unique(Y_all[i]) for i in range(len(Y_all))])])
    y_max = ys.max()
    y_map = np.zeros(y_max + 1, np.int) - 1
    for i, yi in enumerate(ys):
        y_map[yi] = i
    Y_all = [y_map[Y_all[i]] for i in range(len(Y_all))]
    return Y_all


def max_seg_count(Y):
    def seg_count(y):
        # Input label sequence
        return len(segment_labels(y))

    # Input: list of label sequences
    return max(map(seg_count, Y))


def subsample(X, Y, rate=1, dim=0):
    if dim == 0:
        X_ = [x[::rate] for x in X]
        Y_ = [y[::rate] for y in Y]
    elif dim == 1:
        X_ = [x[:, ::rate] for x in X]
        Y_ = [y[::rate] for y in Y]
    else:
        print("Subsample not defined for dim={}".format(dim))
        return None, None

    return X_, Y_


# ------------- Segment functions -------------
def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Yi_split


def segment_data(Xi, Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Xi_split = [np.squeeze(Xi[:, idxs[i]:idxs[i + 1]]) for i in range(len(idxs) - 1)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Xi_split, Yi_split


def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
    return intervals


def segment_lengths(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i + 1] - idxs[i]) for i in range(len(idxs) - 1)]
    return np.array(intervals)


# ------------- IO -------------
def save_predictions(dir_out, y_pred, y_truth, idx_task, experiment_name=""):
    if experiment_name != "":
        dir_out += "/{}/".format(experiment_name)
    # Make sure fiolder exists
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    truth_test_all_out = {"t{}_{}".format(idx_task, k): v for (k, v) in enumerate(y_truth)}
    predict_test_all_out = {"t{}_{}".format(idx_task, k): v for k, v in enumerate(y_pred)}
    sio.savemat(dir_out + "/{}_truth.mat".format(idx_task), truth_test_all_out)
    sio.savemat(dir_out + "/{}_predict.mat".format(idx_task), predict_test_all_out)


# ------------- Vision -------------
def load_images(uris, rez_im, uri_data):
    # Load images for CNN
    X = np.empty((len(uris), 3, rez_im, rez_im), dtype=np.float32)
    for i, x in enumerate(uris):
        im = cv2.imread(uri_data + x)
        im = cv2.resize(im, (rez_im, rez_im))
        X[i] = im.T
    return X


def check_images_available(x_uri, y, uri_data):
    # Check if there are any missing files
    no_file = []
    for i, x in enumerate(x_uri):
        if not os.path.isfile(uri_data + x):
            # print("Missing", x)
            no_file += [i]
    x_uri = np.array([x_uri[i] for i in range(len(x_uri)) if i not in no_file])
    y = np.array([y[i] for i in range(len(y)) if i not in no_file])

    if len(no_file) > 0:
        print("Missing #", len(no_file))

    return x_uri, y


# ----------- dataset -------------
class breakfast_dataset():
    def __init__(self, dir):
        print 'loading data from:', dir
        all = ["P%02d" % i for i in range(3, 56)]
        splits = [all[:13], all[13:26], all[26:39], all[39:52]]

        features = glob.glob(os.path.join(dir, 'breakfast_data', 's1', '*', '*.txt'))
        labels = glob.glob(os.path.join(dir, 'segmentation_coarse', '*', '*.txt'))
        features.sort()
        labels.sort()

        self.data = [[] for i in range(8)]  # x,y for 4 splits
        actions = []

        for f, l in tqdm(zip(features, labels)):
            _x = np.loadtxt(f)
            _y_raw = open(l).readlines()
            _y = np.repeat(0, int(_y_raw[-1].split()[0].split('-')[1]))
            for act in _y_raw:
                tm, lb = act.split()
                if lb not in actions:
                    actions.append(lb)
                _y[(int(tm.split('-')[0]) - 1):int(tm.split('-')[1])] = actions.index(lb)
            n = min(len(_x), len(_y))  # make sure the same length

            for i in range(4):
                if sum([k in f for k in splits[i]]) > 0:  # in the split
                    self.data[2 * i].append(_x[5:n, 1:])  # exclude index and first 5 frames (all-zero feature)
                    self.data[2 * i + 1].append(_y[5:n])
                    break

    def get_split(self, split, type):
        if split not in ['s1', 's2', 's3', 's4']:
            print "please enter split index from ['s1','s2','s3','s4']"
        else:
            split = int(split[1]) - 1
            if type == 'train':
                x = []
                y = []
                for i in range(4):
                    if i != split:
                        x += self.data[2 * i]
                        y += self.data[2 * i + 1]
                return x, y
            elif type == 'test':
                return self.data[2 * split], self.data[2 * split + 1]
            else:
                print "please enter type from ['train','test']"

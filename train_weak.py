#!/usr/bin/env python2.7
# Li Ding
# Mar. 2018


from __future__ import division

import os
import time
import numpy as np
from scipy import io as sio
from keras.utils import np_utils
from itertools import groupby
import cv2
from cv2 import resize

from utils import utils, metrics, tf_models, weak_model

# ---------- Directories & User inputs --------------
# Location of data/features folder
data_dir = './data'  # make sure there are ./data/breakfast_data/s1/... and ./data/segmentation_coarse/... placed

# Parameters
n_nodes = [48, 64, 96]
nb_epoch = 100
conv_len = 25
splits = ['s1', 's2', 's3', 's4']
sample_rate = 15
model_types = ['ED-TCN', 'TC-FPN']
save_predictions = True
batch_size = 8

# ------------------------------------------------------------------
# Evaluate using different filter lengths
if 1:
    # Load all the data, takes up to 3 min
    breakfast_data = utils.breakfast_dataset(data_dir)
    print
    asct = time.asctime()

    for model_type in model_types:
        # Initialize metrics
        trial_metrics = metrics.ComputeMetrics(overlap=.1, bg_class=0)
        trial_metrics_best = metrics.ComputeMetrics(overlap=.1, bg_class=0)
        trial_metrics_final = metrics.ComputeMetrics(overlap=.1, bg_class=0)
        print 'Model type:', model_type

        for split in splits:
            print 'Start:', split
            split_start = time.time()

            # Load data for each split
            X_train, y_train = breakfast_data.get_split(split, "train")
            X_test, y_test = breakfast_data.get_split(split, "test")

            X_train = [i[::sample_rate] for i in X_train]
            y_train = [i[::sample_rate] for i in y_train]
            y_train_real = y_train[:]  # real gt
            y_train_ocr = [np.unique(i) for i in y_train]  # gt occurrence
            y_train_seq = [[i[0] for i in groupby(y)] for y in y_train]  # gt sequence
            y_train_temp = [resize(np.array(i), (1, len(j)), interpolation=cv2.INTER_NEAREST).reshape(len(j)) for i, j
                            in zip(y_train_seq, y_train_real)]

            X_test = [i[::sample_rate] for i in X_test]
            y_test = [i[::sample_rate] for i in y_test]
            y_test_real = y_test[:]
            y_test_ocr = [np.unique(i) for i in y_test]
            y_test_seq = [[i[0] for i in groupby(y)] for y in y_test]
            y_test_temp = [resize(np.array(i), (1, len(j)), interpolation=cv2.INTER_NEAREST).reshape(len(j)) for i, j in
                           zip(y_test_seq, y_test_real)]

            n_layers = len(n_nodes)
            n_classes = 48
            max_len = int(np.max([i.shape[0] for i in X_train + X_test]))
            max_len = int(np.ceil(np.float(max_len) / (2 ** n_layers))) * 2 ** n_layers
            # print("Max length:", max_len)

            if trial_metrics.n_classes is None:
                trial_metrics.set_classes(n_classes)
            if trial_metrics_best.n_classes is None:
                trial_metrics_best.set_classes(n_classes)
            if trial_metrics_final.n_classes is None:
                trial_metrics_final.set_classes(n_classes)

            trial_metrics.add_predictions('train_remap', y_train_temp, y_train_real)
            trial_metrics.print_trials()

            n_feat = 64
            # print "# Feat:", n_feat

            # ------------------ Models ----------------------------

            ocr_train = np.array([np.sum(np_utils.to_categorical(y, n_classes), 0) for y in y_train_ocr])

            # One-hot encoding
            Y_test = [np_utils.to_categorical(y, n_classes) for y in y_test_real]
            minloss = 100
            bestloop = 0

            for loop in range(20):
                print 'loop', loop
                z = 1  # how many label instance to insert

                # Balance each classes
                class_cts = np.array([sum([np.sum(np.array(j) == i) for j in y_train_seq]) for i in range(n_classes)])
                class_cts = (1 / class_cts)**0.5
                class_cts /= (1/48*np.sum(class_cts))
                class_weight = dict(zip(range(n_classes), class_cts))

                # One-hot encoding
                Y_train = [resize(np_utils.to_categorical(y, n_classes), (n_classes, len(t))) for y, t in
                           zip(y_train_seq, y_train_real)]

                # In order process batches simultaneously, all data needs to be of the same length
                X_train_m, Y_train_m, M_train = utils.mask_data(X_train, Y_train, max_len, mask_value=-1)
                X_test_m, Y_test_m, M_test = utils.mask_data(X_test, Y_test, max_len, mask_value=-1)

                # Set training weights
                M_train_temp = M_train[:, :, 0]
                for i,j in zip(M_train_temp, y_train_temp):
                    i[:len(j)] += [class_weight[k] for k in j]
                    i[:len(j)] -= 1

                param_str = None
                model = None
                if model_type == 'ED-TCN':
                    model, param_str = tf_models.ED_TCN(n_nodes, conv_len, n_classes, n_feat, return_param_str=True,
                                                        max_len=max_len)
                elif model_type == 'TC-FPN':
                    model, param_str = weak_model.TCFPN(n_nodes, conv_len, n_classes, n_feat, return_param_str=True,
                                                        in_len=max_len)
                elif model_type == 'GRU':
                    model, param_str = weak_model.GRU64(n_nodes, conv_len, n_classes, n_feat, return_param_str=True,
                                                      in_len=max_len)
                print param_str

                model.fit(X_train_m, Y_train_m, epochs=nb_epoch, verbose=1, sample_weight=M_train_temp, shuffle=True,
                          batch_size=batch_size)

                u = 0.02  # threshold
                ran = 0.1  # randomness

                AP_trainraw = model.predict(X_train_m, verbose=1)
                AP_testraw = model.predict(X_test_m, verbose=1)
                AP_train = utils.unmask(AP_trainraw, M_train)
                AP_test = utils.unmask(AP_testraw, M_test)

                P_train = [p.argmax(1) for p in AP_train]
                P_test = [p.argmax(1) for p in AP_test]

                prob_train = np.array([np.max(p, 0) for p in AP_train])

                loss = np.mean(
                    np.sum(-(ocr_train * np.log(prob_train + 1e-7) + (1 - ocr_train) * np.log(1 - prob_train + 1e-7)),
                           axis=1))
                print 'loss:', loss

                y_train_seqn = []

                for i in range(len(X_train_m)):
                    x = X_train_m[i]
                    pmp = AP_trainraw[i]

                    # realignment
                    seq = y_train_seq[i]
                    seqn = seq[:]
                    k = 0
                    inds = np.arange(1, len(seq)) / len(seq) * len(y_train_real[i])
                    inds = inds.astype(np.int)

                    for ind in range(len(inds)):
                        if seq[ind] != seq[ind + 1]:
                            if pmp[inds[ind], seq[ind]] > pmp[inds[ind], seq[ind + 1]] + u:
                                for zz in range(z):
                                    rr = np.random.random()
                                    if rr > ran:
                                        seqn.insert(ind + k + 1, seq[ind])
                                    else:
                                        seqn.insert(ind + k + 1, seq[ind + 1])
                                k += z
                            elif pmp[inds[ind], seq[ind]] < pmp[inds[ind], seq[ind + 1]] - u:
                                for zz in range(z):
                                    rr = np.random.random()
                                    if rr > ran:
                                        seqn.insert(ind + k + 1, seq[ind + 1])
                                    else:
                                        seqn.insert(ind + k + 1, seq[ind])
                                k += z
                    y_train_seqn.append(seqn)

                y_train_seq = y_train_seqn[:]
                y_train_temp = [resize(np.array(i), (1, len(j)), interpolation=cv2.INTER_NEAREST).reshape(len(j)) for
                                i, j in
                                zip(y_train_seq, y_train_real)]

                trial_metrics.add_predictions('train', P_train, y_train_real)
                trial_metrics.add_predictions('test', P_test, y_test_real)
                trial_metrics.add_predictions('train_remap', y_train_temp, y_train_real)

                if loss < minloss and loop > 1:
                    minloss = loss
                    bestloop = loop
                    bestmodel = model
                    trial_metrics_best.add_predictions('train_remap', y_train_temp, y_train_real)
                    trial_metrics_best.add_predictions('train', P_train, y_train_real)
                    trial_metrics_best.add_predictions('test', P_test, y_test_real)
                    trial_metrics_final.add_predictions('test' + split, P_test, y_test_real)

                split_end = time.time()
                print 'Time elapsed:', time.strftime("%H:%M:%S", time.gmtime(split_end - split_start))
                print
                #print 'True labels count:', [sum([np.sum(j == i, axis=-1) for j in y_test_real]) for i in
                #                             range(n_classes)]
                #print 'Pred labels count:', [sum([np.sum(j == i, axis=-1) for j in P_test]) for i in range(n_classes)]
                #print
                trial_metrics.print_trials()
                print

                if loop - bestloop > 2:
                    print 'Early Stopping at', loop

                    for test_align in range(10):

                        y_test_seqn = []

                        for i in range(len(X_test_m)):
                            x = X_test_m[i]
                            pmp = AP_testraw[i]

                            # realignment
                            seq = y_test_seq[i]
                            seqn = seq[:]
                            k = 0
                            inds = np.arange(1, len(seq)) / len(seq) * len(y_test_real[i])
                            inds = inds.astype(np.int)

                            for ind in range(len(inds)):
                                if seq[ind] != seq[ind + 1]:
                                    if pmp[inds[ind], seq[ind]] > pmp[inds[ind], seq[ind + 1]] + u:
                                        for zz in range(z):
                                            rr = np.random.random()
                                            if rr > ran:
                                                seqn.insert(ind + k + 1, seq[ind])
                                            else:
                                                seqn.insert(ind + k + 1, seq[ind + 1])
                                        k += z
                                    elif pmp[inds[ind], seq[ind]] < pmp[inds[ind], seq[ind + 1]] - u:
                                        for zz in range(z):
                                            rr = np.random.random()
                                            if rr > ran:
                                                seqn.insert(ind + k + 1, seq[ind + 1])
                                            else:
                                                seqn.insert(ind + k + 1, seq[ind])
                                        k += z
                            y_test_seqn.append(seqn)

                        y_test_seq = y_test_seqn[:]
                    y_test_temp = [resize(np.array(i), (1, len(j)), interpolation=cv2.INTER_NEAREST).reshape(len(j))
                                   for i, j in
                                   zip(y_test_seq, y_test_real)]
                    trial_metrics_best.add_predictions('test_align', y_test_temp, y_test_real)
                    break

                # ----- Save prediction -----
                if save_predictions:
                    dir_out = "prediction/{}/{}/{}".format('weak', asct, param_str)

                    # Make sure folder exists
                    if not os.path.isdir(dir_out):
                        os.makedirs(dir_out)

                    out = {"P": P_test, "Y": y_test, "S": AP_test}
                    sio.savemat(dir_out + "/{}.mat".format(split + '_test_iter' + str(loop)), out)
                    out = {"tr_map": y_train_temp, "tr_gt": y_train_real, "tr_prob": AP_train}
                    sio.savemat(dir_out + "/{}.mat".format(split + '_tr_iter' + str(loop)), out)

            print
            print 'Best iter:', bestloop
            print 'Min Loss:', minloss
            trial_metrics_best.print_trials()
            print

        print 'Done!'
        trial_metrics_final.print_trials()
        trial_metrics_final.print_scores()
        print

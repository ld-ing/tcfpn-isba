#!/usr/bin/env python2.7
# Li Ding
# Mar. 2018
# Modified from https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/TCN_main.py


from __future__ import division

import os
import time
import numpy as np
from scipy import io as sio
from keras.utils import np_utils

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
        # Initialize dataset loader & metrics
        trial_metrics_tr = metrics.ComputeMetrics(overlap=.1, bg_class=0)
        trial_metrics_te = metrics.ComputeMetrics(overlap=.1, bg_class=0)
        
        # Load data for each split
        print 'Model type:', model_type

        for split in splits:
            print 'Start:', split
            split_start = time.time()
            asct = time.asctime()

            # Load data for each split
            X_train, y_train = breakfast_data.get_split(split, "train")
            X_test, y_test = breakfast_data.get_split(split, "test")

            X_train = [i[::sample_rate] for i in X_train]
            y_train = [i[::sample_rate] for i in y_train]
            X_test = [i[::sample_rate] for i in X_test]
            y_test = [i[::sample_rate] for i in y_test]

            n_classes = 48
            trial_metrics_tr.set_classes(n_classes)
            trial_metrics_te.set_classes(n_classes)

            train_lengths = [x.shape[0] for x in X_train]
            test_lengths = [x.shape[0] for x in X_test]
            n_train = len(X_train)
            n_test = len(X_test)

            n_feat = 64
            print "# Feat:", n_feat

            # ------------------ Models ----------------------------

            # Go from y_t = {1...C} to one-hot vector (e.g. y_t = [0, 0, 1, 0])
            Y_train = [np_utils.to_categorical(y, n_classes) for y in y_train]
            Y_test = [np_utils.to_categorical(y, n_classes) for y in y_test]

            # In order process batches simultaneously all data needs to be of the same length
            # So make all same length and mask out the ends of each.
            n_layers = len(n_nodes)
            max_len = max(max(train_lengths), max(test_lengths))
            max_len = int(np.ceil(np.float(max_len) / (2 ** n_layers))) * 2 ** n_layers
            print "Max length:", max_len

            X_train_m, Y_train_, M_train = utils.mask_data(X_train, Y_train, max_len, mask_value=-1)
            X_test_m, Y_test_, M_test = utils.mask_data(X_test, Y_test, max_len, mask_value=-1)
            M_train_temp = M_train[:, :, 0]

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

            model.fit(X_train_m, Y_train_, nb_epoch=nb_epoch, batch_size=batch_size,
                      verbose=1, sample_weight=M_train_temp)

            AP_train = model.predict(X_train_m, verbose=0)
            AP_test = model.predict(X_test_m, verbose=0)
            AP_train = utils.unmask(AP_train, M_train)
            AP_test = utils.unmask(AP_test, M_test)

            P_train = [p.argmax(1) for p in AP_train]
            P_test = [p.argmax(1) for p in AP_test]

            # --------- Metrics ----------
            print
            print 'training'
            trial_metrics_tr.add_predictions(split, P_train, y_train)
            trial_metrics_tr.print_trials()
            print
            print 'testing'
            trial_metrics_te.add_predictions(split, P_test, y_test)
            trial_metrics_te.print_trials()
            print

            # ----- Save prediction -----
            if save_predictions:
                dir_out = "prediction/{}/{}/{}".format('supv', asct, param_str)

                # Make sure folder exists
                if not os.path.isdir(dir_out):
                    os.makedirs(dir_out)

                out = {"P": P_test, "Y": y_test, "S": AP_test}
                sio.savemat(dir_out + "/{}.mat".format(split), out)

            split_end = time.time()
            print 'Time elapsed:', time.strftime("%H:%M:%S", time.gmtime(split_end - split_start))
            print
        print
        print 'training'
        trial_metrics_tr.print_scores()
        print
        print 'testing'
        trial_metrics_te.print_scores()
        print

    print "Done!"

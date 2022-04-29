import argparse
import logging
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import trange
import copy

from pFedGP.Learner import pFedGPFullLearner

from backbone import CNNTarget
from utils import get_device, set_logger, set_seed, detach_to_numpy, calc_metrics, str2bool

import os
import sys
import glob
import pickle
import re
from typing import List, Dict

import pandas as pd
from sklearn.metrics import confusion_matrix

import putemg_features
from putemg_features import biolab_utilities
import wandb

parser = argparse.ArgumentParser(description="Personalized Federated Learning")


parser.add_argument("--putemg_folder", type=str, default='../../putemg-downloader/Data-HDF5/')
parser.add_argument("--result_folder", type=str, default='../shallow_learn_results/')
parser.add_argument("--nf", type=str2bool, default=True)
parser.add_argument("--nc", type=str2bool, default=True)
#Moshe skip filter and shallow learn
##################################
#       Optimization args        #
##################################
parser.add_argument("--num-steps", type=int, default=200)
parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--inner-steps", type=int, default=1, help="number of inner steps")
parser.add_argument("--num-client-agg", type=int, default=5, help="number of kernels")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")

################################
#       GP args        #
################################
parser.add_argument('--loss-scaler', default=1., type=float, help='multiplicative element to the loss function')
parser.add_argument('--kernel-function', type=str, default='RBFKernel',
                    choices=['RBFKernel', 'LinearKernel', 'MaternKernel'],
                    help='kernel function')
parser.add_argument('--objective', type=str, default='predictive_likelihood',
                    choices=['predictive_likelihood', 'marginal_likelihood'])
parser.add_argument('--predict-ratio', type=float, default=0.5,
                    help='ratio of samples to make predictions for when using predictive_likelihood objective')
parser.add_argument('--num-gibbs-steps-train', type=int, default=5, help='number of sampling iterations')
parser.add_argument('--num-gibbs-draws-train', type=int, default=20, help='number of parallel gibbs chains')
parser.add_argument('--num-gibbs-steps-test', type=int, default=5, help='number of sampling iterations')
parser.add_argument('--num-gibbs-draws-test', type=int, default=30, help='number of parallel gibbs chains')
parser.add_argument('--outputscale', type=float, default=8., help='output scale')
parser.add_argument('--lengthscale', type=float, default=1., help='length scale')
parser.add_argument('--outputscale-increase', type=str, default='constant',
                    choices=['constant', 'increase', 'decrease'],
                    help='output scale increase/decrease/constant along tree')

#############################
#       General args        #
#############################
parser.add_argument("--num-workers", type=int, default=4, help="number of workers")
parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
parser.add_argument("--eval-every", type=int, default=200, help="eval every X selected steps")
parser.add_argument("--save-path", type=str, default="./output/pFedGP", help="dir path for output file")
parser.add_argument("--seed", type=int, default=42, help="seed value")
parser.add_argument('--wandb', type=str2bool, default=False)

args = parser.parse_args()

set_logger()
set_seed(args.seed)

exp_name = f'putEMG_pFedGP-Full_seed_{args.seed}_wd_{args.wd}_' \
           f'lr_{args.lr}_num_steps_{args.num_steps}_inner_steps_{args.inner_steps}_' \
           f'objective_{args.objective}_predict_ratio_{args.predict_ratio}'

# Weights & Biases
if args.wandb:
    wandb.init(project="gp-pfl", entity="aviv_and_idan", name=exp_name)
    wandb.config.update(args)

#
# if args.exp_name != '':
#     exp_name += '_' + args.exp_name
#
# logging.info(str(args))
# args.out_dir = (Path(args.save_path) / exp_name).as_posix()
# out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
# logging.info(out_dir)

putemg_folder = os.path.abspath(args.putemg_folder)
result_folder = os.path.abspath(args.result_folder)

if not os.path.isdir(putemg_folder):
    print('{:s} is not a valid folder'.format(putemg_folder))
    exit(1)

if not os.path.isdir(result_folder):
    print('{:s} is not a valid folder'.format(result_folder))
    exit(1)

#putemg_folder = os.path.abspath(sys.argv[1])

if not os.path.isdir(putemg_folder):
    print('{:s} is not a valid folder'.format(putemg_folder))
    exit(1)

filtered_data_folder = os.path.join(result_folder, 'filtered_data')
calculated_features_folder = os.path.join(result_folder, 'calculated_features')

# list all hdf5 files in given input folder
all_files = [f for f in sorted(glob.glob(os.path.join(putemg_folder, "*.hdf5")))]
#Moshe take only gesture sequential for part of users
user_trains = [f'emg_gestures-{user}-{traj}' for user in ['03', '04', '05', '06', '07'] for traj in ['sequential',
                                                                                                     'repeats_short',
                                                                                                     'repeats_long']]
train_user_files = [f for f in all_files if any([a for a in user_trains if a in f])]
all_files = train_user_files
# if not skipped filter the input data and save to consequent output files
if not args.nf:
    # create folder for filtered data
    if not os.path.exists(filtered_data_folder):
        os.makedirs(filtered_data_folder)

    # by each filename in download folder
    for file in all_files:
        basename = os.path.basename(file)
        filename = os.path.splitext(basename)[0]
        print('Denoising file: {:s}'.format(basename))

        # read raw putEMG data file and run filter
        df: pd.DataFrame = pd.read_hdf(file)
        biolab_utilities.apply_filter(df)

        # save filtered data to designated folder with prefix filtered_
        output_file = filename + '_filtered.hdf5'
        print('Saving to file: {:s}'.format(output_file))
        df.to_hdf(os.path.join(filtered_data_folder, output_file),
                  'data', format='table', mode='w', complevel=5)
else:
    print('Denoising skipped!')
    print()


# if not skipped calculate features from filtered files
if not args.nc:
    # create folder for calculated features
    if not os.path.exists(calculated_features_folder):
        os.makedirs(calculated_features_folder)

    # by each filename in download folder
    for file in all_files:
        basename = os.path.basename(file)
        filename = os.path.splitext(basename)[0]

        filtered_file_name = filename + '_filtered.hdf5'
        print('Calculating features for {:s} file'.format(filtered_file_name))

        # for filtered data file run feature extraction, use xml with limited feature set
        ft: pd.DataFrame = putemg_features.features_from_xml('./features_shallow_learn.xml',
                                                             os.path.join(filtered_data_folder, filtered_file_name))

        # save extracted features file to designated folder with features_filtered_ prefix
        output_file = filename + '_filtered_features.hdf5'
        print('Saving result to {:s} file'.format(output_file))
        ft.to_hdf(os.path.join(calculated_features_folder, output_file),
                  'data', format='table', mode='w', complevel=5)
else:
    print('Feature extraction skipped!')
    print()

# create list of records
all_feature_records = [biolab_utilities.Record(os.path.basename(f)) for f in all_files]

# data can be additionally filtered based on subject id
records_filtered_by_subject = biolab_utilities.record_filter(all_feature_records)
# records_filtered_by_subject = record_filter(all_feature_records,
#                                             whitelists={"id": ["01", "02", "03", "04", "07"]})
# records_filtered_by_subject = pu.record_filter(all_feature_records, whitelists={"id": ["01"]})

# load feature data to memory
dfs: Dict[biolab_utilities.Record, pd.DataFrame] = {}
for r in records_filtered_by_subject:
    print("Reading features for input file: ", r)
    filename = os.path.splitext(r.path)[0]
    dfs[r] = pd.DataFrame(pd.read_hdf(os.path.join(calculated_features_folder,
                                                   filename + '_filtered_features.hdf5')))

# create k-fold validation set, with 3 splits - for each experiment day 3 combination are generated
# this results in 6 data combination for each subject
# Moshe splits = 1 instead of 3
splits_all = biolab_utilities.data_per_id_and_date(records_filtered_by_subject, n_splits=3)

device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

# defines feature sets to be used in shallow learn
feature_sets = {
    # "RMS": ["RMS"],
    "Hudgins": ["MAV", "WL", "ZC", "SSC"]   # ,
    # "Du": ["IAV", "VAR", "WL", "ZC", "SSC", "WAMP"]
}

# defines gestures to be used in shallow learn
gestures = {
    0: "Idle",
    1: "Fist",
    2: "Flexion",
    3: "Extension",
    4: "Pinch index",
    5: "Pinch middle",
    6: "Pinch ring",
    7: "Pinch small"
}

num_classes = 8
classes_per_client = 8
num_clients = len(splits_all.values())

# defines channels configurations for which classification will be run
channel_range = {
    "24chn": {"begin": 1, "end": 24},
    # "8chn_1band": {"begin": 1, "end": 8},
    "8chn_2band": {"begin": 9, "end": 16},
    # "8chn_3band": {"begin": 17, "end": 24}
}

@torch.no_grad()
def eval_model(global_model, GPs, feature_set_name, features):
    results = defaultdict(lambda: defaultdict(list))
    targets = []
    preds = []
    step_results = []

    global_model.eval()

    for client_id in range(num_clients):
        running_loss, running_correct, running_samples = 0., 0., 0.

        # iterate over each internal data
        for i_s, subject_data in enumerate(list(splits_all.values())[client_id]):
            is_first_iter = True
            # get data of client
            # prepare training and testing set based on combination of k-fold split, feature set and gesture set
            # this is also where gesture transitions are deleted from training and test set
            # only active part of gesture performance remains
            data = biolab_utilities.prepare_data(dfs, subject_data, features, list(gestures.keys()))

            # list columns containing only feature data
            regex = re.compile(r'input_[0-9]+_[A-Z]+_[0-9]+')
            cols = list(filter(regex.search, list(data["train"].columns.values)))

            # strip columns to include only selected channels, eg. only one band
            cols = [c for c in cols if (ch_range["begin"] <= int(c[c.rindex('_') + 1:]) <= ch_range["end"])]

            # extract limited training x and y, only with chosen channel configuration
            train_x = torch.tensor(data["train"][cols].to_numpy(), dtype=torch.float32)
            train_y = torch.LongTensor(data["train"]["output_0"].to_numpy())

            # # extract limited testing x and y, only with chosen channel configuration
            test_x = torch.tensor(data["test"][cols].to_numpy(), dtype=torch.float32)
            test_y_true = torch.LongTensor(data["test"]["output_0"].to_numpy())

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_x, train_y),
                shuffle=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )

            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(test_x, test_y_true),
                shuffle=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )

            # build tree at each step
            GPs[client_id], label_map, Y_train, X_train = build_tree(global_model, client_id, train_loader)
            GPs[client_id].eval()
            client_data_labels = []
            client_data_preds = []

            for batch_count, batch in enumerate(test_loader):
                img, label = tuple(t.to(device) for t in batch)
                Y_test = torch.tensor([label_map[l.item()] for l in label], dtype=label.dtype,
                                             device=label.device)

                X_test = global_model(img)
                loss, pred = GPs[client_id].forward_eval(X_train, Y_train, X_test, Y_test, is_first_iter)

                running_loss += loss.item()
                running_correct += pred.argmax(1).eq(Y_test).sum().item()
                running_samples += len(Y_test)

                is_first_iter = False
                targets.append(Y_test)
                preds.append(pred)

                client_data_labels.append(Y_test)
                client_data_preds.append(pred)

            # calculate confusion matrix
            cm = confusion_matrix(detach_to_numpy(torch.cat(client_data_labels, dim=0)),
                                  detach_to_numpy(torch.max(torch.cat(client_data_preds, dim=0), dim=1)[1]))

            # save classification results to output structure
            step_results.append({"id": client_id, "split": i_s, "clf": 'pFedGP',
                                 "feature_set": feature_set_name,
                                 "cm": cm, "y_true": detach_to_numpy(torch.cat(client_data_labels, dim=0)),
                                 "y_pred": detach_to_numpy(torch.max(torch.cat(client_data_preds, dim=0), dim=1)[1])})

        # erase tree (no need to save it)
        GPs[client_id].tree = None

        results[client_id]['loss'] = running_loss / (batch_count + 1)
        results[client_id]['correct'] = running_correct
        results[client_id]['total'] = running_samples

    target = detach_to_numpy(torch.cat(targets, dim=0))
    full_pred = detach_to_numpy(torch.cat(preds, dim=0))
    labels_vs_preds = np.concatenate((target.reshape(-1, 1), full_pred), axis=1)

    return results, labels_vs_preds, step_results


def get_optimizer(network):
    return torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9) \
        if args.optimizer == 'sgd' else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)


@torch.no_grad()
def build_tree(net, client_id, loader):
    """
    Build GP tree per client
    :return: List of GPs
    """
    for k, batch in enumerate(loader):
        batch = (t.to(device) for t in batch)
        train_data, clf_labels = batch

        z = net(train_data)
        X = torch.cat((X, z), dim=0) if k > 0 else z
        Y = torch.cat((Y, clf_labels), dim=0) if k > 0 else clf_labels

    # build label map
    client_labels, client_indices = torch.sort(torch.unique(Y))
    label_map = {client_labels[i].item(): client_indices[i].item() for i in range(client_labels.shape[0])}
    offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
                                 device=Y.device)

    GPs[client_id].build_base_tree(X, offset_labels)  # build tree
    return GPs[client_id], label_map, offset_labels, X


criteria = torch.nn.CrossEntropyLoss()

###############################
# init net and GP #
###############################
for ch_range_name, ch_range in channel_range.items():
    logging.info("======================== " + ch_range_name + " =======================")

    output: Dict[str, any] = dict()

    output["gestures"] = gestures
    output["classifiers"] = {"pFedGP": {"predictor": "pFedGP", "args": {}}}
    output["feature_sets"] = feature_sets
    output["results"]: List[Dict[str, any]] = list()

    # for each feature set
    for feature_set_name, features in feature_sets.items():
        logging.info("======================== " + feature_set_name + " =======================")

        if ch_range_name == '24chn':
            n_features = 24 if feature_set_name == "RMS" else 144 if feature_set_name == "Du" else 96  # "Hudgins"
        else:  # 8chn_2band
            n_features = 8 if feature_set_name == "RMS" else 48 if feature_set_name == "Du" else 32  # "Hudgins"

        clients = splits_all
        gp_counter = 0

        # NN
        net = CNNTarget(n_features=n_features)
        net = net.to(device)

        GPs = torch.nn.ModuleList([])
        for client_id in range(num_clients):
            GPs.append(pFedGPFullLearner(args, classes_per_client))  # GP instances

        results = defaultdict(list)

        ################
        # init metrics #
        ################
        last_eval = -1
        best_step = -1
        best_acc = -1
        test_best_based_on_step, test_best_min_based_on_step = -1, -1
        test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
        step_iter = trange(args.num_steps)
        test_avg_loss = 10
        test_avg_acc = 0

        for step in step_iter:

            # print tree stats every 100 epochs
            to_print = True if step % 100 == 0 else False

            # select several clients
            client_ids = np.random.choice(num_clients, size=args.num_client_agg, replace=False)

            # initialize global model params
            params = OrderedDict()
            for n, p in net.named_parameters():
                params[n] = torch.zeros_like(p.data)

            # iterate over each client
            train_avg_loss = 0
            num_samples = 0

            for j, client_id in enumerate(client_ids):
                curr_global_net = copy.deepcopy(net)
                curr_global_net.train()
                optimizer = get_optimizer(curr_global_net)

                # get the first value to
                # values_view = splits_all.values()
                # value_iterator = iter(values_view)
                # first_value = next(value_iterator)

                # iterate over each internal data
                for subject_data in list(splits_all.values())[client_id]:

                    # get data of client
                    # prepare training and testing set based on combination of k-fold split, feature set and gesture set
                    # this is also where gesture transitions are deleted from training and test set
                    # only active part of gesture performance remains
                    data = biolab_utilities.prepare_data(dfs, subject_data, features, list(gestures.keys()))

                    # list columns containing only feature data
                    regex = re.compile(r'input_[0-9]+_[A-Z]+_[0-9]+')
                    cols = list(filter(regex.search, list(data["train"].columns.values)))

                    # strip columns to include only selected channels, eg. only one band
                    cols = [c for c in cols if (ch_range["begin"] <= int(c[c.rindex('_') + 1:]) <= ch_range["end"])]

                    # extract limited training x and y, only with chosen channel configuration
                    train_x = torch.tensor(data["train"][cols].to_numpy(), dtype=torch.float32)
                    train_y = torch.LongTensor(data["train"]["output_0"].to_numpy())

                    train_loader = torch.utils.data.DataLoader(
                        torch.utils.data.TensorDataset(train_x, train_y),
                        shuffle=True,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers
                    )

                    # build tree at each step
                    GPs[client_id], label_map, _, __ = build_tree(curr_global_net, client_id, train_loader)
                    GPs[client_id].train()

                    for i in range(args.inner_steps):

                        # init optimizers
                        optimizer.zero_grad()

                        for k, batch in enumerate(train_loader):
                            batch = (t.to(device) for t in batch)
                            img, label = batch

                            z = curr_global_net(img)
                            X = torch.cat((X, z), dim=0) if k > 0 else z
                            Y = torch.cat((Y, label), dim=0) if k > 0 else label

                        offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
                                                     device=Y.device)

                        loss = GPs[client_id](X, offset_labels, to_print=to_print)
                        loss *= args.loss_scaler

                        # propagate loss
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(curr_global_net.parameters(), 50)
                        optimizer.step()

                        train_avg_loss += loss.item() * offset_labels.shape[0]
                        num_samples += offset_labels.shape[0]

                        step_iter.set_description(
                            f"Step: {step+1}, client: {client_id}, Inner Step: {i}, Loss: {loss.item()}"
                        )

                for n, p in curr_global_net.named_parameters():
                    params[n] += p.data
                # erase tree (no need to save it)
                GPs[client_id].tree = None

            train_avg_loss /= num_samples

            # average parameters
            for n, p in params.items():
                params[n] = p / args.num_client_agg
            # update new parameters
            net.load_state_dict(params)

            if (step + 1) == args.num_steps:
                test_results, labels_vs_preds_val, step_results = eval_model(net, GPs, feature_set_name, features)
                test_avg_loss, test_avg_acc = calc_metrics(test_results)
                logging.info(f"Step: {step + 1}, Test Loss: {test_avg_loss:.4f},  Test Acc: {test_avg_acc:.4f}")
                for i in step_results:
                    output["results"].append(i)

            if args.wandb:
                wandb.log(
                    {
                        'custom_step': step,
                        'train_loss': train_avg_loss,
                        'test_avg_loss': test_avg_loss,
                        'test_avg_acc': test_avg_acc,
                    }
                )

    # for each channel configuration dump classification results to file
    file = os.path.join(result_folder, "classification_result_" + exp_name + "_" + ch_range_name + ".bin")
    pickle.dump(output, open(file, "wb"))
    if args.wandb:
        wandb.save(file)

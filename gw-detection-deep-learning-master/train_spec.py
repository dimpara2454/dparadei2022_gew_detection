import argparse
import logging
import os
import numpy as np

np.random.seed(0)
import h5py
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from bisect import bisect

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torch.multiprocessing

# torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn', force=True)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.dataset import SlicerDataset, SlicerDatasetSNR
from utils.train_utils import WarmUpLR, initialize_xavier, progress_bar
from modules.loss import reg_BCELoss
from modules.resnet2d import ResNet18, ResNet50, ResNet8
from modules.whiten import SpecCropWhitenNet


def decode_snr_schedule(sch_str):
    # .e.g.: '5:1-3,5:1-1.5,5:1-1.25,10:1-1,2:0.8-1.2
    steps = sch_str.split(',')
    epochs = []
    s_ranges = []
    for step in steps:
        ep, range_ = step.split(':')
        epochs.append(int(ep))
        s_min, s_max = range_.split('-')
        s_ranges.append([float(s_min), float(s_max)])
    epochs.append(1)
    s_ranges.append(([-np.inf, np.inf]))
    epochs = np.cumsum(epochs)
    return epochs, s_ranges


def get_snr_by_epoch(sch_epochs, sch_ranges, epoch):
    print(
        f'Epoch: {epoch}, schedule: {[(sch_epoch, sch_range) for sch_epoch, sch_range in zip(sch_epochs, sch_ranges)]}')
    index = bisect(sch_epochs, epoch)
    if index >= len(sch_epochs):
        return sch_ranges[-1]
    # print(f'Chose {index}-th pair: {sch_ranges[index]}')
    return sch_ranges[index]


class MyCorrelationModel(nn.Module):
    def __init__(self, resnet_type='resnet8'):
        super(MyCorrelationModel, self).__init__()
        if resnet_type == 'resnet8':
            self.feature_extractor = ResNet8(1, n_classes=1, fe_only=True)
        elif resnet_type == 'resnet18':
            self.feature_extractor = ResNet18(1, n_classes=1, fe_only=True)
        elif resnet_type == 'resnet50':
            self.feature_extractor = ResNet50(1, n_classes=1, fe_only=True)
        else:
            print("Unrecognized resnet_type, must be one of ['resnet8', 'resnet18', 'resnet50']")
            exit(0)
        self.bn = nn.BatchNorm1d(1, affine=True)
        # self.bn = nn.Identity()

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, w, h = x.size()
        x = x.view(-1, nz * c, w, h)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-1))
        return out

    def forward(self, x):
        x1 = x[:, :1, :, :]
        x2 = x[:, 1:, :, :]
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        corr = self._fast_xcorr(x1, x2)
        # corr_norm = corr.squeeze(2) / (torch.norm(x1, 2) * torch.norm(x2, 2))
        corr_norm = self.bn(corr).squeeze(2)
        # print(corr_norm.size())
        # return self.bn(corr_norm).clip_(0, 1)
        return torch.sigmoid(corr_norm)

class SimilarityModel(nn.Module):
    def __init__(self, resnet_type='resnet8'):
        super(SimilarityModel, self).__init__()
        if resnet_type == 'resnet8':
            self.feature_extractor = ResNet8(1, n_classes=1, fe_only=True)
        elif resnet_type == 'resnet18':
            self.feature_extractor = ResNet18(1, n_classes=1, fe_only=True)
        elif resnet_type == 'resnet50':
            self.feature_extractor = ResNet50(1, n_classes=1, fe_only=True)
        else:
            print("Unrecognized resnet_type, must be one of ['resnet8', 'resnet18', 'resnet50']")
            exit(0)
        self.conv_pool = nn.Conv2d(512, 128, 14, padding=0)

    def forward(self, x):
        x1 = x[:, :1, :, :]
        x2 = x[:, 1:, :, :]
        x1 = torch.flatten(self.conv_pool(self.feature_extractor(x1)), 1, 3)
        x2 = torch.flatten(self.conv_pool(self.feature_extractor(x2)), 1, 3)
        return torch.cosine_similarity(x1, x2, dim=1).unsqueeze_(1)

class SeparateClassificationModel(nn.Module):
    def __init__(self, resnet_type='resnet8'):
        super(SeparateClassificationModel, self).__init__()
        if resnet_type == 'resnet8':
            self.feature_extractor = ResNet8(1, n_classes=2)
        elif resnet_type == 'resnet18':
            self.feature_extractor = ResNet18(1, n_classes=2)
        elif resnet_type == 'resnet50':
            self.feature_extractor = ResNet50(1, n_classes=2)
        else:
            print("Unrecognized resnet_type, must be one of ['resnet8', 'resnet18', 'resnet50']")
            exit(0)

    def forward(self, x):
        x1 = x[:, :1, :, :]
        x2 = x[:, 1:, :, :]
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        return torch.softmax((x1 + x2), dim=1)


class JointClassificationModel(nn.Module):
    def __init__(self, resnet_type='resnet8'):
        super(JointClassificationModel, self).__init__()
        if resnet_type == 'resnet8':
            self.feature_extractor = ResNet8(2, n_classes=2)
        elif resnet_type == 'resnet18':
            self.feature_extractor = ResNet18(2, n_classes=2)
        elif resnet_type == 'resnet50':
            self.feature_extractor = ResNet50(2, n_classes=2)
        else:
            print("Unrecognized resnet_type, must be one of ['resnet8', 'resnet18', 'resnet50']")
            exit(0)

    def forward(self, x):
        out = self.feature_extractor(x)
        return torch.softmax(out, dim=1)


# Set default weights filename
default_weights_fname = 'weights.pt'

# Set data type to be used
dtype = torch.float32

sample_rate = 2048
delta_t = 1. / sample_rate
delta_f = 1 / 1.25


def main(args):
    output_dir = args.output_dir
    train_network = args.train

    if not os.path.exists(output_dir):
        logging.info(f'Creating output directory {output_dir}...')
        os.makedirs(output_dir)

    network_type = args.network_type
    # where to save/load the weights after training
    weights_path = os.path.join(output_dir, default_weights_fname)
    dataset = 4

    val_hdf = os.path.join(args.data_dir, f'dataset-{dataset}/v2/val_background_s24w6d1_1.hdf')
    val_npy = os.path.join(args.data_dir, f'dataset-{dataset}/v2/val_injections_s24w6d1_1.25s.npy')

    train_device = args.train_device
    if network_type == 'xcorr':
        base_model = MyCorrelationModel(resnet_type=args.resnet_type).to(train_device, dtype=dtype)
        n_classes = 1
        print(base_model)
    elif network_type == 'similarity':
        base_model = SimilarityModel(resnet_type=args.resnet_type).to(train_device, dtype=dtype)
        n_classes = 1
        print(base_model)
    elif network_type == 'sepclass':
        base_model = SeparateClassificationModel(resnet_type=args.resnet_type).to(train_device, dtype=dtype)
        n_classes = 2
        print(base_model)
    elif network_type == 'jointclass':
        base_model = JointClassificationModel(resnet_type=args.resnet_type).to(train_device, dtype=dtype)
        n_classes = 2
        print(base_model)
    else:
        print("Unrecognized network_type, must be one of ['xcorr', 'sepclass', 'jointclass', 'similarity']")
        exit(0)

    # norm = AdaptiveBatchNorm1d(2).to(train_device)
    # norm = DAIN_Layer(input_dim=2).to(train_device)
    norm = nn.InstanceNorm2d(num_features=128).to(train_device)
    base_model.apply(initialize_xavier)

    net = SpecCropWhitenNet(base_model, norm).to(train_device)

    if args.resume_from is not None:
        print(f'Loading weights: {args.resume_from}')
        net.load_state_dict(torch.load(args.resume_from))

    net.whiten.max_filter_len = 0.5
    net.whiten.legacy = False

    validation_dataset = SlicerDataset(val_hdf, val_npy, slice_len=int(args.slice_dur * sample_rate),
                                       slice_stride=int(args.slice_stride * sample_rate),
                                       max_seg_idx=int(np.floor(args.slice_dur)), n_classes=n_classes)
    val_dl = DataLoader(validation_dataset, batch_size=25, shuffle=True, num_workers=args.num_workers,
                        pin_memory=train_device)

    if train_network:
        suffix = '_' + args.suffix if args.suffix is not None else ''

        background_hdf = os.path.join(args.data_dir, f'dataset-{dataset}/v2/train_background_s24w61w_1.hdf')
        injections_hdf = os.path.join(args.data_dir, f'dataset-{dataset}/v2/train_injections_s24w61w_1.hdf')
        injset = dataset if dataset != 3 else 4
        inj_npy = os.path.join(args.data_dir, f'dataset-{injset}/v2/train_injections_s24w61w_1.25s_all.npy')

        use_curriculum_learning = args.snr_schedule != ''
        if use_curriculum_learning:
            print('Using curriculum learning based on SNR difficulty...')
            sch_epochs, sch_ranges = decode_snr_schedule(args.snr_schedule)
            min_snr, max_snr = sch_ranges[0]
            training_dataset = SlicerDatasetSNR(background_hdf, inj_npy, slice_len=int(args.slice_dur * sample_rate),
                                                slice_stride=int(args.slice_stride * sample_rate),
                                                max_seg_idx=int(np.floor(args.slice_dur)),
                                                injections_hdf=injections_hdf, min_snr=min_snr, max_snr=max_snr,
                                                p_augment=args.p_augment)
        else:
            print('No curriculum learning used...')
            training_dataset = SlicerDataset(background_hdf, inj_npy, slice_len=int(args.slice_dur * sample_rate),
                                             slice_stride=int(args.slice_stride * sample_rate),
                                             max_seg_idx=int(np.floor(args.slice_dur)), n_classes=n_classes,
                                             p_augment=args.p_augment)
        batch_size = args.batch_size
        # DataLoaders handle efficient loading of the data into batches
        train_dl = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=train_device)

        # setup loss
        if args.loss == 'smooth':
            loss = reg_BCELoss(dim=n_classes)
        else:
            loss = nn.BCELoss()

        # setup optimizer
        learning_rate = args.learning_rate
        if args.optimizer == 'adam':
            opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
        else:
            opt = torch.optim.SGD(net.parameters(), lr=learning_rate)
        milestones = [int(m) for m in args.lr_milestones.split(',')]
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=args.gamma)
        n_wrm = args.warmup_epochs
        wrm = WarmUpLR(opt, int(len(train_dl) * n_wrm))

        # train/val loop
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        n_epochs = args.epochs
        # for epoch in tqdm(range(n_epochs), desc="Optimizing network"):
        for epoch in range(n_epochs):

            net.train()
            # train losses
            training_running_loss = 0.
            training_batches = 0
            # train accuracy
            total = 0
            correct = 0
            # val accuracy
            total_val = 0
            correct_val = 0

            if use_curriculum_learning:
                s_min, s_max = get_snr_by_epoch(sch_epochs, sch_ranges, epoch)
                training_dataset.set_snr_range(s_min, s_max)

            for idx, (training_samples, training_labels, training_abs_inj_times) in enumerate(train_dl):
                training_samples = training_samples.to(device=train_device)
                training_labels = training_labels.to(device=train_device)

                # Optimizer step on a single batch of training data
                opt.zero_grad()
                if epoch < n_wrm:
                    wrm.step()

                training_output = net(training_samples, training_abs_inj_times)
                training_loss = loss(training_output, training_labels)
                training_loss.backward()
                # Clip gradients to make convergence somewhat easier
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_norm)
                # Make the actual optimizer step and save the batch loss
                opt.step()

                # get predictions & gt to measure accuracy
                predicted = (training_output[:, 0] > 0.5).float()
                gt = training_labels[:, 0]
                total += training_output.size(0)
                correct += predicted.eq(gt).sum().item()
                train_acc = 100. * (correct / total)

                # update running loss
                training_running_loss += training_loss.clone().cpu().item()
                training_batches += 1

                progress_bar(idx, len(train_dl),
                             f'Epoch {epoch} | Loss {training_running_loss / training_batches:.2f} | Acc {train_acc:.2f}')

            # Evaluation on the validation dataset
            net.eval()
            with torch.no_grad():

                # error analysis in last epoch
                positive_correct = 0
                positive_total = 0
                positive_corr = []

                negative_correct = 0
                negative_total = 0
                negative_corr = []

                val_predictions = []
                val_loss = []
                val_groundtruth = []

                validation_running_loss = 0.
                validation_batches = 0
                for val_idx, (validation_samples, validation_labels, validation_abs_inj_times) in enumerate(val_dl):
                    validation_samples = validation_samples.to(device=train_device)
                    validation_labels = validation_labels.to(device=train_device)

                    # Evaluation of a single validation batch
                    validation_output = net(validation_samples, validation_abs_inj_times)
                    validation_loss = loss(validation_output, validation_labels)

                    # get predictions & gt to measure accuracy
                    # _, predicted_val = validation_output.max(1)
                    # _, gt_val = validation_labels.max(1)
                    predicted_val = (validation_output[:, 0] > 0.5).float()
                    gt_val = validation_labels[:, 0]
                    total_val += validation_output.size(0)
                    correct_val += predicted_val.eq(gt_val).sum().item()
                    val_acc = 100. * (correct_val / total_val)
                    validation_running_loss += validation_loss.clone().cpu().item()

                    # get predictions & gt to measure accuracy
                    val_predictions.extend(predicted_val)
                    val_groundtruth.extend(gt_val.cpu().numpy())
                    pos_idx = gt_val == 0
                    neg_idx = ~pos_idx
                    positive_corr.extend(predicted_val[pos_idx].eq(gt_val[pos_idx]).cpu().numpy())
                    negative_corr.extend(predicted_val[neg_idx].eq(gt_val[neg_idx]).cpu().numpy())
                    positive_total += pos_idx.sum()
                    negative_total += neg_idx.sum()
                    positive_correct += predicted_val[pos_idx].eq(gt_val[pos_idx]).sum().item()
                    negative_correct += predicted_val[neg_idx].eq(gt_val[neg_idx]).sum().item()
                    validation_batches += 1
                    progress_bar(val_idx, len(val_dl),
                                 f'Validation | Loss {validation_running_loss / validation_batches:.2f} | Acc {val_acc:.2f}'
                                 f' (+:{100 * (positive_correct / positive_total):.3f}%,-:{100 * (negative_correct / negative_total):.3f}%)')

            # Print information on the training and validation loss in the current epoch and save current network state
            validation_loss = validation_running_loss / validation_batches
            training_loss = training_running_loss / training_batches
            output_string = '%04i Train Loss: %f | Val Loss: %f || Train Acc: %.3f%% | Val Acc: %.3f%% (+:%.3f%%,-:%.3f%%)' % (
                epoch, training_loss, validation_loss,
                train_acc, val_acc, 100 * (positive_correct / positive_total),
                100 * (negative_correct / negative_total))
            train_losses.append(training_loss)
            val_losses.append(validation_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            logging.info(output_string)
            sch.step()

            torch.save(net.state_dict(), weights_path)

        # training over, save network
        torch.save(net.state_dict(), weights_path)

        ### training plots
        fig, axs = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
        fig.suptitle('Training loss & acc')
        axs[0].plot(train_losses, label='train')
        axs[0].plot(val_losses, label='val')
        axs[0].title.set_text('Loss')

        axs[1].plot(train_accs, label='train')
        axs[1].plot(val_accs, label='val')
        axs[1].title.set_text('Accuracy')

        fig.savefig(f'{output_dir}/training_curves{suffix}.png')

        # Print information on the training and validation loss in the current epoch and save current network state
        # validation_loss = validation_running_loss / validation_batches
        positive_acc = 100 * (positive_correct / positive_total)
        negative_acc = 100 * (negative_correct / negative_total)
        total_acc = 100 * (correct_val / total_val)
        print(f'Validation accuracy: {total_acc}% (positive: {positive_acc}%, negative: {negative_acc}%)')

        # save to json for plotting later
        with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
            train_dict = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'val_pos_acc': positive_acc.cpu().item(),
                'val_neg_acc': negative_acc.cpu().item(),
            }
            json.dump(train_dict, f)


if __name__ == '__main__':
    seconds_per_month = 30 * 24 * 60 * 60
    seconds_per_week = 7 * 24 * 60 * 60

    parser = argparse.ArgumentParser()

    testing_group = parser.add_argument_group('testing')
    training_group = parser.add_argument_group('training')

    parser.add_argument('--verbose', action='store_true', help="Print update messages.")
    parser.add_argument('--train', action='store_true', help="Train the network before applying.")
    parser.add_argument('-o', '--output-dir', type=str, help="Path to the directory where the outputs will be stored.")
    parser.add_argument('--network-type', type=str,
                        help='Type of network to load.',
                        default='xcorr', choices=['xcorr', 'sepclass', 'jointclass', 'similarity'])
    parser.add_argument('--resnet-type', type=str,
                        help='Type of resnet to use as feature extractor.',
                        default='resnet8', choices=['resnet8', 'resnet18', 'resnet50'])
    parser.add_argument('--slice-dur', type=float, default=3.25, help='Duration (in s) of original slices, e.g., 3.25.'
                                                                      'After the PSD is calculated these slices are further cropped to 1s.')
    parser.add_argument('--slice-stride', type=float, default=2., help='Slice stride.')
    parser.add_argument('--suffix', type=str, default=None, help='plots suffix')

    training_group.add_argument('--resume-from', type=str, default=None,
                                help='If set, weights will be loaded from this path and training will resume from these weights.')
    training_group.add_argument('--data-dir', type=str, default='',
                                help='Path to directory containing dataset-1 folder')
    training_group.add_argument('--learning-rate', type=float, default=5e-5,
                                help="Learning rate of the optimizer. Default: 0.00005")
    training_group.add_argument('--lr-milestones', type=str, default='20,50',
                                help='Epochs at which we multiply lr by gamma')
    training_group.add_argument('--gamma', type=float, default=0.5,
                                help='Rate to multiply learning rate by at milestones.')
    training_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                                help='Type of optimizer.')
    training_group.add_argument('--epochs', type=int, default=10, help="Number of training epochs. Default: 10")
    # try: --snr-schedule 4:12.5-100,2:8.5-100,2:1-8.5,2:1-12.5
    training_group.add_argument('--snr-schedule', type=str, default='',
                                help='Formatted string for waveform SNR filtering.'
                                     'First number is epochs, then the range is separated by -')
    training_group.add_argument('--batch-size', type=int, default=32,
                                help="Batch size of the training algorithm. Default: 32")
    training_group.add_argument('--warmup-epochs', type=float, default=0,
                                help="If >0, the learning rate will be annealed from 1e-8 to learning rate in warmup_epochs")
    training_group.add_argument('--clip-norm', type=float, default=100.,
                                help="Gradient clipping norm to stabilize the training. Default 100.")
    training_group.add_argument('--p-augment', type=float, default=0,
                                help="Percentage of samples where L1 noise is randomly replaced with different segment.")
    training_group.add_argument('--train-device', type=str, default='cpu',
                                help="Device to train the network. Use 'cuda' for the GPU."
                                     "Also, 'cpu:0', 'cuda:1', etc. (zero-indexed). Default: cpu")
    training_group.add_argument('--num-workers', type=int, default=8,
                                help="Number of workers to use when loading training data. Default: 8")
    training_group.add_argument('--loss', type=str, choices=['smooth', 'bce'], default='bce',
                                help="Type of loss function. Standard BCE, or label smoothing"
                                     "(e.g., 0.9 instead of 1, 0.1 instead of 0")
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=logging.INFO,
                        datefmt='%d-%m-%Y %H:%M:%S')

    main(args)

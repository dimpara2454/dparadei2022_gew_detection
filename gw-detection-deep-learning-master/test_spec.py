#! /usr/bin/env python3
import argparse
import h5py
import torch
from torch import nn

from utils.eval_utils import get_triggers, get_clusters
from modules.dain import DAIN_Layer
from modules.whiten import CropWhitenNet, SpecCropWhitenNet
from modules.resnet2d import ResNet8, ResNet18, ResNet50
from train_spec import MyCorrelationModel, SeparateClassificationModel, JointClassificationModel

# Set data type to be used
dtype = torch.float32

sample_rate = 2048
delta_t = 1. / sample_rate
delta_f = 1 / 1.25


def main(args):
    device = 'cuda:1' if torch.cuda.device_count() > 0 else 'cpu'
    # weights filename
    network_type = args.network_type
    if network_type == 'xcorr':
        base_model = MyCorrelationModel(resnet_type=args.resnet_type).to(device, dtype=dtype)
        print(base_model)
    elif network_type == 'sepclass':
        base_model = SeparateClassificationModel(resnet_type=args.resnet_type).to(device, dtype=dtype)
        print(base_model)
    elif network_type == 'joinclass':
        base_model = JointClassificationModel(resnet_type=args.resnet_type).to(device, dtype=dtype)
        print(base_model)
    else:
        print("Unrecognized network_type, must be one of ['xcorr', 'sepclass', 'jointclass']")
        exit(0)

    norm = nn.InstanceNorm2d(num_features=128).to(device)

    net = SpecCropWhitenNet(base_model, norm).to(device)
    net.whiten.max_filter_len = 0.5
    net.whiten.legacy = False
    net.deploy = True
    weights_path = args.weights
    net.load_state_dict(torch.load(weights_path, map_location=device))

    net.eval()

    # run on foreground
    inputfile = args.inputfile
    outputfile = args.outputfile
    step_size = 2.1
    slice_dur = 3.25
    trigger_threshold = 0.5
    cluster_threshold = 0.35
    var = 0.5

    test_batch_size = 4

    with torch.no_grad():
        triggers = get_triggers(net,
                                inputfile,
                                step_size=step_size,
                                trigger_threshold=trigger_threshold,
                                device=device,
                                verbose=True,
                                batch_size=test_batch_size,
                                whiten=False,
                                slice_length=int(slice_dur * 2048),
                                num_workers=0)

    time, stat, var = get_clusters(triggers, cluster_threshold, var=var)

    with h5py.File(outputfile, 'w') as outfile:
        print("Saving clustered triggers into %s." % outputfile)

        outfile.create_dataset('time', data=time)
        outfile.create_dataset('stat', data=stat)
        outfile.create_dataset('var', data=var)

        print("Triggers saved, closing file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('inputfile', type=str, help="The path to the input data file.")
    parser.add_argument('outputfile', type=str, help="The path where to store the triggers.")

    parser.add_argument('--weights', type=str, help='Custom weights path.', default=None)
    parser.add_argument('--network-type', type=str,
                        help='Type of network to load.',
                        default='xcorr', choices=['xcorr', 'speclass', 'jointclass'])
    parser.add_argument('--resnet-type', type=str,
                        help='Type of resnet to use as feature extractor.',
                        default='resnet8', choices=['resnet8', 'resnet18', 'resnet50'])

    args = parser.parse_args()

    main(args)

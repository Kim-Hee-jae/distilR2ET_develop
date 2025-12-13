import os
import sys
sys.path.append("./outside-code")

import yaml
import numpy as np
import torch
import torch.nn as nn

import argparse
from tqdm import tqdm

from os import listdir, makedirs
from os.path import exists, join, splitext

from src.model_shape_aware import RetNet
from inference_bvh import load_from_bvh, getmodel


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='R2ET for motion retargeting')
    parser.add_argument(
        '--config',
        default='./config/inference_cfg.yaml',
        help='path to the configuration file',
    )
    parser.add_argument(
        '--save_path', default='./', help='path to the configuration file'
    )
    parser.add_argument('--phase', default='test', help='train or test')
    parser.add_argument('--load_inp_data', type=dict, default=dict(), help='')
    parser.add_argument('--weights', default='', help='xxx.pt weights for generator')
    parser.add_argument(
        '--device', type=int, default=0, nargs='+', help='only 0 avaliable'
    )
    parser.add_argument(
        '--num_joint', type=int, default=22, help='number of the joints'
    )
    parser.add_argument(
        '--ret_model_args',
        type=dict,
        default=dict(),
        help='the arguments of retargetor',
    )
    parser.add_argument(
        '--k', type=float, default=0.8, help='adjustable k for balacing gate'
    )

    return parser




def main(arg):
    ret_model = getmodel(arg.weights, arg)
    parents = np.array(
        [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]
    )

    src_char_list = listdir(arg.load_inp_data['inp_bvh_dir_path'])
    for src_char in src_char_list:
        print(f'Source Character: {src_char}')
        save_dir_path = join(arg.save_path, src_char)
        if exists(save_dir_path):
            print(f'Folder: {save_dir_path} already exists.')
            print(f'Skip {src_char}')
            continue
        else:
            makedirs(save_dir_path)

        inp_shape_path = join(arg.load_inp_data['inp_shape_dir_path'], src_char+'.npz')
        bvh_dir_path = join(arg.load_inp_data['inp_bvh_dir_path'], src_char)
        bvh_list = [bvh_name for bvh_name in listdir(bvh_dir_path) if bvh_name.endswith('.bvh')]
        for bvh_name in tqdm(bvh_list):
            inp_bvh_path = join(bvh_dir_path, bvh_name)
            save_path = join(save_dir_path, f'{splitext(bvh_name)[0]}.npz')

            # load data
            (
                inp_seq,
                inpskel,
                tgtskel,
                inp_shape,
                tgt_shape,
                inpquat,
                inp_height,
                tgt_height,
                local_mean,
                local_std,
                quat_mean,
                quat_std,
                global_mean,
                global_std,
                tgtanim,
                tgtname,
                tgtftime,
                inpanim,
                inpname,
                inpftime,
                inpjoints,
                tgtjoints,
            ) = load_from_bvh(
                arg.device[0], 
                inp_shape_path=inp_shape_path,
                tgt_shape_path=arg.load_inp_data['tgt_shape_path'],
                stats_path=arg.load_inp_data['stats_path'],
                inp_bvh_path=inp_bvh_path,
                tgt_bvh_path=arg.load_inp_data['tgt_bvh_path']
                )

            # PATCH: tgtskel from t-pose
            if tgtskel.shape[1] != inpskel.shape[1]:
                tgtskel = tgtskel[:,:1].repeat(1, inpskel.shape[1], 1)


            with torch.no_grad():
                localB_rt, globalB_rt, quatB_rt, delta_qs, delta_qg = ret_model(
                    inp_seq,
                    None,       # seqB: inference -> None
                    inpskel,
                    tgtskel,
                    inp_shape,
                    tgt_shape,
                    inpquat,
                    inp_height,
                    tgt_height,
                    local_mean,
                    local_std,
                    quat_mean,
                    quat_std,
                    parents,
                    arg.k,
                    arg.phase,
                    )

            delta_qs = delta_qs.cpu()
            delta_qg = delta_qg.cpu()
            quatB_rt = quatB_rt.cpu()
            np.savez(save_path, delta_qs=delta_qs, delta_qg=delta_qg, quatB_rt=quatB_rt)




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert k in key
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()

    main(arg)
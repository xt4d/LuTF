import torch
import sys, os
import json
import importlib
import pickle
import numpy as np
import argparse

import src.utils as utils
from config import Configuration

from src.model.landuse_lutf_model import LUTFModel
from src.model.landuse_lutf_model import LUTFGeoModel
from src.model.landuse_lutf_model import LUTFGeoPlanModel


def sample_sequence(model, seq_x, seq_h, prompt_length, steps, temperature=1.0, top_k=5, do_sample=False):
    bsz, seql = seq_x.shape[0], seq_x.shape[1]
    generated = seq_x[:, :prompt_length].permute(1, 0).contiguous()
    with torch.no_grad():
        for _ in range(steps):
            roll_h = model.decoder.decode(generated, seq_h, add_bos=False)
            pred = model.decoder.logits_head(roll_h)
            next_token_logits = pred[-1, :, :]
            filtered_logits = utils.top_k_logits(next_token_logits, k=top_k)
            if do_sample:
                next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            else:
                __, next_token = torch.topk(torch.softmax(filtered_logits, dim=-1), k=1, dim=-1)
            generated = torch.cat((generated, next_token.reshape(1, bsz)), dim=0)

    generated = generated.permute(1, 0).contiguous()
    return generated[:, 1:].cpu().tolist(), roll_h


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--vis_type', type=str)
    parser.add_argument('--comp_type', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=98052)
    parser.add_argument('--evaluate_data_root', type=str, default='./data/evaluate/')

    args = parser.parse_args()
    print(args.__repr__())

    if args.model_type == None or args.config_path == None:
        print('Model and config are not given!')
        exit()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    '''generator parameters'''
    config_path = args.config_path
    model_type = args.model_type
    vis_type = args.vis_type
    comp_type = args.comp_type

    with open(config_path, 'r') as fin:
        fstr = fin.read()
        config = json.loads(fstr, object_hook=Configuration)
        print(json.dumps(json.loads(fstr), indent=4))

    device = args.device

    class_names = config.data_format.class_names
    seq_length = config.data_format.width * config.data_format.height

    if model_type == 'lutf_g_p':
        model_class = LUTFGeoPlanModel
    elif model_type == 'lutf_g':
        model_class = LUTFGeoModel
    elif model_type == 'lutf':
        model_class = LUTFModel
    else:
        print('No model found:', model_type)

    model = model_class(
        len(class_names), 
        seq_length, 
        model_dim = config.model_params.model_dim,
        head_num = config.model_params.head_num,
        dec_layer_num = config.model_params.layer_num,
        enc_layer_num = config.model_params.enc_layer_num,
        comp_enc_layer_num = config.model_params.comp_enc_layer_num
    ).to(device)

    out_folder = os.path.join(
        config.test_config.output_root, 
        os.path.basename(os.path.normpath(args.evaluate_data_root))
    )
    os.makedirs(out_folder, exist_ok=True)

    ckpt_path = config.test_config.ckpt_path

    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    batch_size = config.test_config.batch_size

    mx_list = []
    comp_list = []

    with open(os.path.join(args.evaluate_data_root, '%s_%s.pkl' % (vis_type, comp_type)), 'rb') as fin:
        samples = pickle.load(fin)
        mx_list = samples['mx']
        comp_list = samples['comp']
        
    scene_sampled = []

    print('#Evaluation samples', len(mx_list))

    for bi in range(0, len(mx_list), batch_size):

        if (bi // batch_size) % (len(mx_list) // batch_size // 10) == 0:
            print('Trench', bi)
        
        mx = torch.tensor(mx_list[bi:bi+batch_size], dtype=torch.long, device=device)
        comp = torch.tensor(comp_list[bi:bi+batch_size], dtype=torch.float, device=device)

        bsz = comp.shape[0]

        seq_h = model.encode(mx, comp)
        seq_x = torch.zeros((bsz, 1), dtype=torch.long, device=device)
        generated, roll_h = sample_sequence(model, seq_x, seq_h, 1, 1024, do_sample=True)

        scene_sampled += generated

    with open(os.path.join(out_folder, '%s_%s.pkl' % (vis_type, comp_type)), 'wb+') as fout:
        pickle.dump({
            'generated': scene_sampled,
        }, fout)

import argparse
import sys, os
import time
from datetime import datetime
import numpy as np
import json
import importlib

import torch
from torch.utils.data import DataLoader

from config import Configuration


def worker_init_fn_seed(worker_id):
    seed = int(time.time() * 1000) % 1000000
    seed += worker_id
    np.random.seed(seed)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lutf.json')
    parser.add_argument('--seed', type=int, default=98052)
    args = parser.parse_args()

    return args


def train(local_rank, trainer, config, train_data, valid_data):

    train_config = config.train_config

    print('Local Rank', local_rank)

    if isinstance(local_rank, int):
        device = 'cuda:%d' % local_rank
    else:
        device = local_rank

    trainer.build_model(device)
    trainer.build_optimizer()

    start_ep = 0
    min_eval_loss = float('inf')

    # model loading should be after DDP warp.
    if train_config.resume_path and os.path.exists(train_config.resume_path):
        start_ep, min_eval_loss = trainer.load_model(train_config.resume_path, device=device)
        print(local_rank, 'Epoch', start_ep, 'Loaded', train_config.resume_path)

    train_dataloader = DataLoader(
        train_data,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn_seed
    )

    valid_dataloader = DataLoader(
        valid_data,
        batch_size=train_config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=train_config.num_workers,
        worker_init_fn=worker_init_fn_seed
    )

    print(
        'Start training', datetime.now(), 
        'train data', len(train_data), 
        'valid data', len(valid_data), 
        'start epoch', start_ep, 
        'loss', min_eval_loss
    )

    for epoch in range(start_ep, train_config.epochs):

        if local_rank == 0 and (epoch % train_config.save_cycle) == 0 and epoch > start_ep:

            if not os.path.exists(train_config.save_dir):
                os.makedirs(train_config.save_dir, exist_ok=True)

            trainer.save_model(
                os.path.join(train_config.save_dir, 'ckpt_epoch%d.pt' % epoch),
                epoch,
                min_eval_loss
            )

            trainer.save_model(
                os.path.join(train_config.save_dir, 'latest.pt'),
                epoch,
                min_eval_loss
            )
            print('save latest', epoch)

            with torch.no_grad():

                trainer.set_eval()

                tot_bcnt = 0
                tot_loss = 0.0

                for _ in range(10):
                    for bi, batch in enumerate(valid_dataloader):
                        tot_loss += trainer.eval_step(batch, device)
                        tot_bcnt += 1

                print('eval', tot_loss / tot_bcnt)

                if tot_loss / tot_bcnt < min_eval_loss:
                    min_eval_loss = tot_loss / tot_bcnt

                    trainer.save_model(
                        os.path.join(train_config.save_dir, 'best.pt'),
                        epoch,
                        min_eval_loss
                    )
                    print('save best', epoch, min_eval_loss)

        trainer.set_train()
        trainer.reset_loss()

        for bi, batch in enumerate(train_dataloader):

            trainer.train_step(batch, device, log_loss=(local_rank == 0))

        if (local_rank == 0) and (epoch % train_config.print_cycle) == 0:
            trainer.print_loss(epoch)


if __name__ == '__main__':

    args = get_args()
    print(args.__repr__())

    config = None
    with open(args.config, 'r') as fin:
        fstr = fin.read()
        config = json.loads(fstr, object_hook=Configuration)
        print(json.dumps(json.loads(fstr), indent=4))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Trainer = get_obj_from_str(config.trainer)
    trainer = Trainer(config)

    train_data, valid_data = trainer.build_dataset()

    if config.train_config.device != 'cuda':
        train(config.train_config.device, trainer, config, train_data, valid_data)
    elif config.train_config.num_gpus == 1:
        train(0, trainer, config, train_data, valid_data)
    else:
        print('Run multi-gpu training')

        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '98052'
        # torch.multiprocessing.set_start_method('spawn')

        # processes = []
        # for rank in range(config.train_config.num_gpus):
        #     p = torch.multiprocessing.Process(target=train, args=(rank, config, train_data, valid_data))
        #     p.start()
        #     processes.append(p)

        # for p in processes:
        #     p.join()

import torch
import os

from ..data.landuse_comp_dataset import LanduseCompDataset

class BaseTrainer:

    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.train_data = None
        self.valid_data = None
        self.reset_loss()


    def reset_loss(self):
        pass


    def build_model(self):
        pass


    def build_optimizer(self):
        pass
    
    def set_train(self):
        self.model.train()


    def set_eval(self):
        self.model.eval()


    def save_model(self, fpath, epoch, eval_loss):

        torch.save(
            { 
                'config': self.config, 
                'model': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 
                'epochs': int(epoch),
                'eval_loss': eval_loss,
            },
            fpath
        )


    def load_model(self, fpath, device='cpu'):

        ckpt = torch.load(fpath, map_location=device)

        self.model.load_state_dict(ckpt['model'], strict=False)

        start_ep = -1
        eval_loss = float('inf')

        if 'optimizer' in ckpt and self.optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'epochs' in ckpt:
            start_ep = int(ckpt['epochs'])
        if 'eval_loss' in ckpt:
            eval_loss = float(ckpt['eval_loss'])

        return start_ep, eval_loss


    def build_dataset(self):

        train_data_list = LanduseCompDataset.load_data_list(
            self.config.train_dataset.data_root, 
            self.config.train_dataset.data_file)

        print('Train samples', len(train_data_list))

        self.train_data = LanduseCompDataset(
            train_data_list, 
            self.config.data_format.class_names,
            use_geo_measures = self.config.train_dataset.use_geo_measures,
            use_plan = self.config.train_dataset.use_plan,
            use_random_comb = self.config.train_dataset.use_random_comb,
            width = self.config.data_format.width, 
            height = self.config.data_format.height
        )

        val_data_list = LanduseCompDataset.load_data_list(
            self.config.valid_dataset.data_root, 
            self.config.valid_dataset.data_file)

        print('Valid samples', len(val_data_list))

        self.valid_data = LanduseCompDataset(
            val_data_list, 
            self.config.data_format.class_names,
            use_geo_measures = self.config.valid_dataset.use_geo_measures,
            use_plan = self.config.valid_dataset.use_plan,
            use_random_comb = self.config.valid_dataset.use_random_comb,
            width = self.config.data_format.width, 
            height = self.config.data_format.height
        )

        return self.train_data, self.valid_data

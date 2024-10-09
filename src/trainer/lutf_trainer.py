import torch
from datetime import datetime

from .base_trainer import BaseTrainer
from ..model.landuse_lutf_model import LUTFModel


class LUTFTrainer(BaseTrainer):

    def __init__(self, config):
        super(LUTFTrainer, self).__init__(config)


    def build_model(self, device):

        self.model = LUTFModel(
            self.train_data.get_class_num(),
            self.train_data.max_length,
            model_dim = self.config.model_params.model_dim,
            head_num = self.config.model_params.head_num,
            dec_layer_num = self.config.model_params.layer_num,
            enc_layer_num = self.config.model_params.enc_layer_num,
            comp_enc_layer_num = self.config.model_params.comp_enc_layer_num
        ).to(device)

        return self.model


    def build_optimizer(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.train_config.learning_rate)
        return self.optimizer

    
    def eval_step(self, batch, device):

        mx = batch['mx'].to(device)
        x = batch['x'].to(device)
        comp = batch['comp'].to(device)

        loss = self.model(x, mx, comp)

        return loss.detach().item()

    
    def train_step(self, batch, device, log_loss=False):

        mx = batch['mx'].to(device)
        x = batch['x'].to(device)
        comp = batch['comp'].to(device)
        
        self.optimizer.zero_grad()

        loss = self.model(x, mx, comp)
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if log_loss:
            self.cum_loss += loss.detach().item()
            self.bcnt += 1

        return loss.detach().item()

    
    def reset_loss(self):

        self.cum_loss = 0.0
        self.bcnt = 0


    def print_loss(self, epoch):

        print(
            '%s - Epoch %d - Train' % (datetime.now(), epoch), 
            self.cum_loss / self.bcnt
        )

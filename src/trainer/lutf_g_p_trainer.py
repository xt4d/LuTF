import torch
from datetime import datetime

from .base_trainer import BaseTrainer
from ..model.landuse_lutf_model import LUTFGeoPlanModel

class LUTFGeoPlanTrainer(BaseTrainer):

    def __init__(self, config):
        super(LUTFGeoPlanTrainer, self).__init__(config)


    def build_model(self, device):

        self.model = LUTFGeoPlanModel(
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

        x = batch['x'].to(device)
        mx = batch['mx'].to(device)
        comp = batch['comp'].to(device)
        x_dist = batch['dist'].to(device)
        x_direct = batch['direct'].to(device)
        x_degree = batch['degree'].to(device)
        x_plan = batch['plan'].to(device)

        loss_ce, loss_g1, loss_g2, loss_g3, loss_plan = self.model(x, x_dist, x_direct, x_degree, x_plan, mx, comp)

        loss = loss_ce + (loss_g1 + loss_g2 + loss_g3) * self.config.train_config.loss_weight.geo_loss + loss_plan * self.config.train_config.loss_weight.plan_loss

        return loss.detach().item()

    
    def train_step(self, batch, device, log_loss=False):

        x = batch['x'].to(device)
        mx = batch['mx'].to(device)
        comp = batch['comp'].to(device)
        x_dist = batch['dist'].to(device)
        x_direct = batch['direct'].to(device)
        x_degree = batch['degree'].to(device)
        x_plan = batch['plan'].to(device)

        self.optimizer.zero_grad()

        loss_ce, loss_g1, loss_g2, loss_g3, loss_plan = self.model(x, x_dist, x_direct, x_degree, x_plan, mx, comp)

        loss = loss_ce \
            + (loss_g1 + loss_g2 + loss_g3) * self.config.train_config.loss_weight.geo_loss \
            + loss_plan * self.config.train_config.loss_weight.plan_loss
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if log_loss:
            self.cum_loss += loss.detach().item()
            self.cum_loss_ce += loss_ce.detach().item()
            self.cum_loss_g1 += loss_g1.detach().item()
            self.cum_loss_g2 += loss_g2.detach().item()
            self.cum_loss_g3 += loss_g3.detach().item()
            self.cum_loss_plan = loss_plan.detach().item()
            self.bcnt += 1

        return loss.detach().item()

    
    def reset_loss(self):

        self.cum_loss = 0.0
        self.cum_loss_ce = 0.0
        self.cum_loss_g1 = 0.0
        self.cum_loss_g2 = 0.0
        self.cum_loss_g3 = 0.0
        self.cum_loss_plan = 0.0
        self.bcnt = 0


    def print_loss(self, epoch):

        print(
            '%s - Epoch %d - Train' % (datetime.now(), epoch), 
            self.cum_loss / self.bcnt,
            self.cum_loss_ce / self.bcnt, 
            self.cum_loss_g1 / self.bcnt, 
            self.cum_loss_g2 / self.bcnt, 
            self.cum_loss_g3 / self.bcnt,
            self.cum_loss_plan / self.bcnt
        )

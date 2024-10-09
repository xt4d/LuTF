import torch
import math
import numpy as np

class CompEncoder(torch.nn.Module):

    def __init__(self, landuse_dim, model_dim = 64, head_num = 8, layer_num = 6):

        super(CompEncoder, self).__init__()

        self.landuse_dim = landuse_dim
        self.head_num = head_num
        self.model_dim = model_dim
        self.layer_num = layer_num

        self.landuse_emb_layer = torch.nn.Embedding(landuse_dim, model_dim-1)

        comp_encoder_layer = torch.nn.TransformerEncoderLayer(d_model = model_dim, nhead = head_num)
        self.comp_encoder = torch.nn.TransformerEncoder(comp_encoder_layer, num_layers = layer_num)


    def encode_comp(self, comp):

        c1 = self.landuse_emb_layer(comp[:, :, 0].long())
        c2 = comp[:, :, 1].unsqueeze(-1)

        c2 = 2 * c2 - 1.0

        xc = torch.cat((c1, c2), dim=-1)
        return self.comp_encoder(xc)


    def forward(self, comp):

        comp = comp.permute(1, 0, 2).contiguous()

        return self.encode_comp(comp)


class SequenceEncoder(torch.nn.Module):

    def __init__(self, landuse_dim, seq_length, model_dim = 64, head_num = 8, layer_num = 6):

        super(SequenceEncoder, self).__init__()

        self.landuse_dim = landuse_dim
        self.seq_length = seq_length
        self.model_dim = model_dim
        self.head_num = head_num
        self.layer_num = layer_num

        self.pos_emb = torch.nn.Parameter(torch.randn(seq_length, model_dim))
        self.emb_layer = torch.nn.Embedding(landuse_dim, model_dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model = model_dim, nhead = head_num)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers = layer_num)


    def encode(self, x):

        e = self.emb_layer(x) + self.pos_emb.unsqueeze(1)
        h = self.encoder(e)

        return h


    def forward(self, x):

        x = x.permute(1, 0).contiguous()

        return self.encode(x)



class CausalDecoder(torch.nn.Module):

    def __init__(self, landuse_dim, seq_length, bos_value = 0, model_dim = 64, head_num = 8, layer_num = 6):

        super(CausalDecoder, self).__init__()

        self.seq_length = seq_length
        self.landuse_dim = landuse_dim
        self.head_num = head_num
        self.model_dim = model_dim
        self.layer_num = layer_num
        self.bos_value = bos_value

        self.pos_emb = torch.nn.Parameter(torch.randn(seq_length, model_dim))
        self.emb_layer = torch.nn.Embedding(landuse_dim, model_dim)

        decoder_layer = torch.nn.TransformerDecoderLayer(d_model = model_dim, nhead = head_num)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers = layer_num)

        self.logits_head = torch.nn.Linear(model_dim, landuse_dim)


    def decode(self, x, comp_h, add_bos=True):

        qlen = x.shape[0]
        bsz = x.shape[1]

        attn_mask = CausalDecoder.generate_square_subsequent_mask(qlen).to(x.device)

        x_with_bos = x

        if add_bos:
            bos_token = torch.ones((1, bsz), dtype=x.dtype, device=x.device) * self.bos_value
            x_with_bos = torch.cat((bos_token, x), dim=0)[:-1]

        e = self.emb_layer(x_with_bos) + self.pos_emb[:qlen, :].unsqueeze(1)

        h = self.decoder(e, comp_h, tgt_mask = attn_mask)

        return h


    def category_loss(self, h, x):
        
        tot = x.shape[0]*x.shape[1]
    
        logits = self.logits_head(h)
        
        loss_ce = torch.nn.functional.cross_entropy(logits.reshape((tot, -1)), x.reshape(tot))
        
        return loss_ce
        
    
    def forward(self, x, comp_h):

        x = x.permute(1, 0).contiguous()

        dec_h = self.decode(x, comp_h)

        loss = self.category_loss(dec_h, x)

        return loss


    @staticmethod
    def generate_square_subsequent_mask(sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



class SpatialCausalDecoder(CausalDecoder):

    def __init__(self, landuse_dim, seq_length, bos_value=0, model_dim=64, head_num=8, layer_num=6):

        super(SpatialCausalDecoder, self).__init__(landuse_dim, seq_length, bos_value, model_dim, head_num, layer_num)

        self.distance_head = torch.nn.Linear(model_dim, 1)
        self.direction_head = torch.nn.Linear(model_dim, 1)
        self.degree4_head = torch.nn.Linear(model_dim, 1)


    def spatial_loss(self, dec_h, d1, d2, d3):
        
        pred_d1 = torch.sigmoid(self.distance_head(dec_h)).squeeze(-1)
        pred_d2 = torch.sigmoid(self.direction_head(dec_h)).squeeze(-1)
        pred_d3 = torch.sigmoid(self.degree4_head(dec_h)).squeeze(-1)
        
        loss1 = torch.nn.functional.l1_loss(pred_d1, d1)
        loss2 = torch.nn.functional.l1_loss(pred_d2, d2)
        loss3 = torch.nn.functional.l1_loss(pred_d3, d3)
        
        return loss1, loss2, loss3

    
    '''x, d1, d2, d3: [bsz, qlen]'''
    '''comp_h: [clen, bsz, dim]'''
    def forward(self, x, d1, d2, d3, comp_h):

        x = x.permute(1, 0).contiguous()
        d1 = d1.permute(1, 0).contiguous()
        d2 = d2.permute(1, 0).contiguous()
        d3 = d3.permute(1, 0).contiguous()

        dec_h = self.decode(x, comp_h)

        loss_ce = self.category_loss(dec_h, x)
        loss1, loss2, loss3 = self.spatial_loss(dec_h, d1, d2, d3)

        return loss_ce, loss1, loss2, loss3



class ProgressSpatialCausalDecoder(SpatialCausalDecoder):

    def __init__(self, landuse_dim, seq_length, bos_value=0, model_dim=64, head_num=8, layer_num=6, prog_dp=0.2):

        super(ProgressSpatialCausalDecoder, self).__init__(landuse_dim, seq_length, bos_value, model_dim, head_num, layer_num)

        self.progress_head = torch.nn.Linear(model_dim, landuse_dim)
        self.progress_dropout = torch.nn.Dropout(p=prog_dp)
    
    
    def progress_loss(self, dec_h, prog):
        
        dec_h = self.progress_dropout(dec_h)

        pred_prog = torch.sigmoid(self.progress_head(dec_h))
        loss = torch.nn.functional.mse_loss(pred_prog, prog) * dec_h.shape[0]

        # pred_prog = torch.sigmoid(self.progress_head(dec_h))
        # proba = pred_prog / torch.sum(pred_prog, dim=-1, keepdim=True)

        # loss = torch.nn.functional.kl_div(torch.log(proba + 1e-14), prog, reduction='mean')
        
        return loss
        
    
    '''x, d1, d2, d3: [bsz, len]'''
    '''prog: [bsz, len, dim]'''
    '''comp_h: [len, bsz, dim]'''
    def forward(self, x, d1, d2, d3, prog, comp_h):

        x = x.permute(1, 0).contiguous()
        d1 = d1.permute(1, 0).contiguous()
        d2 = d2.permute(1, 0).contiguous()
        d3 = d3.permute(1, 0).contiguous()
        prog = prog.permute(1, 0, 2).contiguous()

        dec_h = self.decode(x, comp_h)
        
        loss1, loss2, loss3 = self.spatial_loss(dec_h, d1, d2, d3)
        loss_prog = self.progress_loss(dec_h, prog)
        loss_ce = self.category_loss(dec_h, x)

        return loss_ce, loss1, loss2, loss3, loss_prog



class PlanSpatialCausalDecoder(SpatialCausalDecoder):

    def __init__(self, landuse_dim, seq_length, bos_value=0, model_dim=64, head_num=8, layer_num=6, prog_dp=0.1):

        super(PlanSpatialCausalDecoder, self).__init__(landuse_dim, seq_length, bos_value, model_dim, head_num, layer_num)

        self.plan_head = torch.nn.Linear(model_dim, landuse_dim)
        self.plan_dropout = torch.nn.Dropout(p=prog_dp)
    
    
    def plan_loss(self, dec_h, pp):
        
        dec_h = self.plan_dropout(dec_h)

        pred_plan = torch.sigmoid(self.plan_head(dec_h))
        proba = pred_plan / torch.sum(pred_plan, dim=-1, keepdim=True)

        loss = torch.nn.functional.kl_div(torch.log(proba + 1e-14), pp, reduction='mean')
        
        return loss
        
    
    '''x, d1, d2, d3: [bsz, len]'''
    '''pp: [bsz, len, dim]'''
    '''comp_h: [len, bsz, dim]'''
    def forward(self, x, d1, d2, d3, pp, comp_h):

        x = x.permute(1, 0).contiguous()
        d1 = d1.permute(1, 0).contiguous()
        d2 = d2.permute(1, 0).contiguous()
        d3 = d3.permute(1, 0).contiguous()
        pp = pp.permute(1, 0, 2).contiguous()

        dec_h = self.decode(x, comp_h)
        
        loss1, loss2, loss3 = self.spatial_loss(dec_h, d1, d2, d3)
        loss_pp = self.plan_loss(dec_h, pp)
        loss_ce = self.category_loss(dec_h, x)

        return loss_ce, loss1, loss2, loss3, loss_pp

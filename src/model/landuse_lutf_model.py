import torch
from .landuse_modules import CompEncoder, SequenceEncoder, CausalDecoder, SpatialCausalDecoder, PlanSpatialCausalDecoder


class LUTFBaseModel(torch.nn.Module):

    def __init__(self, landuse_dim, seq_length, model_dim = 64, head_num = 8, enc_layer_num = 6, comp_enc_layer_num = 3):

        super(LUTFBaseModel, self).__init__()

        self.seq_length = seq_length

        self.comp_encoder = CompEncoder(
            landuse_dim, 
            model_dim = model_dim, 
            head_num = head_num, 
            layer_num = comp_enc_layer_num
        )

        self.seq_encoder = SequenceEncoder(
            landuse_dim, 
            seq_length,
            model_dim = model_dim, 
            head_num = head_num, 
            layer_num = enc_layer_num
        )

        self.decoder = None


    def encode(self, mx, comp):

        comp_h = self.comp_encoder(comp)
        mx_h = self.seq_encoder(mx)[:int(self.seq_length**0.5)]

        h = torch.cat((mx_h, comp_h), dim=0)
        return h


    def forward(self):
        pass



'''1: LUTF'''
class LUTFModel(LUTFBaseModel):

    def __init__(self, landuse_dim, seq_length, bos_value = 0, model_dim = 64, head_num = 8, dec_layer_num = 6, enc_layer_num = 6, comp_enc_layer_num = 3):

        super(LUTFModel, self).__init__(
            landuse_dim,
            seq_length,
            model_dim,
            head_num,
            enc_layer_num,
            comp_enc_layer_num
        )

        self.decoder = CausalDecoder(
            landuse_dim,
            seq_length,
            bos_value = bos_value,
            model_dim = model_dim,
            head_num = head_num,
            layer_num = dec_layer_num
        )


    def forward(self, x, mx, comp):

        h = self.encode(mx, comp)

        loss = self.decoder(x, h)

        return loss



'''2: LUTF + G'''
class LUTFGeoModel(LUTFBaseModel):

    def __init__(self, landuse_dim, seq_length, bos_value = 0, model_dim = 64, head_num = 8, dec_layer_num = 6, enc_layer_num = 6, comp_enc_layer_num = 3):

        super(LUTFGeoModel, self).__init__(
            landuse_dim,
            seq_length,
            model_dim,
            head_num,
            enc_layer_num,
            comp_enc_layer_num
        )

        self.decoder = SpatialCausalDecoder(
            landuse_dim,
            seq_length,
            bos_value = bos_value,
            model_dim = model_dim,
            head_num = head_num,
            layer_num = dec_layer_num
        )


    def forward(self, x, d1, d2, d3, mx, comp):

        h = self.encode(mx, comp)

        loss_ce, loss1, loss2, loss3 = self.decoder(x, d1, d2, d3, h)

        return loss_ce, loss1, loss2, loss3



'''3: LUTF + G + P'''
class LUTFGeoPlanModel(LUTFBaseModel):

    def __init__(self, landuse_dim, seq_length, bos_value = 0, model_dim = 64, head_num = 8, dec_layer_num = 6, enc_layer_num = 6, comp_enc_layer_num = 3):

        super(LUTFGeoPlanModel, self).__init__(
            landuse_dim,
            seq_length,
            model_dim,
            head_num,
            enc_layer_num,
            comp_enc_layer_num
        )

        self.decoder = PlanSpatialCausalDecoder(
            landuse_dim,
            seq_length,
            bos_value = bos_value,
            model_dim = model_dim,
            head_num = head_num,
            layer_num = dec_layer_num
        )


    def forward(self, x, d1, d2, d3, pp, mx, comp):

        h = self.encode(mx, comp)

        loss_ce, loss1, loss2, loss3, loss_pp = self.decoder(x, d1, d2, d3, pp, h)

        return loss_ce, loss1, loss2, loss3, loss_pp

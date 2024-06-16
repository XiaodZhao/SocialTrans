import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
        def __init__(self, d_model, ob_T,dropout=0.1):
                super(PositionalEncoding, self).__init__()
                self.dropout = nn.Dropout(p=dropout)
                pe = torch.zeros(ob_T, d_model)
                position = torch.arange(0, ob_T, dtype=torch.float).unsqueeze(1)
                div_term = torch.pow(10000.0, (torch.arange(0, d_model, 2).float() / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                # *pe ob_T x d_model
                pe = pe.unsqueeze(0).transpose(0, 1)
                # *pe 1 x ob_T x d_model ---> ob_T x 1 x d_model
                self.register_buffer('pe', pe)

        def forward(self, x):
                # todo x: [ob_T, batch_size, d_model]
                # print(x.shape,'x.shape')
                # print(self.pe.shape,'self.pe.shape')
                x = x + self.pe
                
                return self.dropout(x)
                # return x


class SDPATT_inter(nn.Module):
        def __init__(self):
            super(SDPATT_inter, self).__init__()

        def forward(self, Q, K, V, attn_mask):

            # todo         Q: [batch_size, Nn, n_heads, ob_T, d_k]
            # todo         K: [batch_size, Nn, n_heads, ob_T, d_k]
            # todo         V: [batch_size, Nn, n_heads, ob_T, d_v]
            # todo attn_mask: [batch_size, Nn, n_heads, ob_T, ob_T]

            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(-1)) # *scores : [batch_size, Nn, n_heads, ob_T, ob_T]
            scores.masked_fill_(~attn_mask, -1e9)
            soft = nn.Softmax(dim=-1)
            attn = soft(scores)
            res_inter = torch.matmul(attn, V)
            return res_inter, attn


class SDPATT_self(nn.Module):
        def __init__(self):
            super(SDPATT_self, self).__init__()
        def forward(self, Q, K, V):
            # todo         Q: [batch_size, ob_T, d_k]
            # todo         K: [batch_size, ob_T, d_k]
            # todo         V: [batch_size, ob_T, d_v]
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(-1)) # *scores : [batch_size, ob_T, ob_T]
            soft = nn.Softmax(dim=-1)
            attn_self = soft(scores)  # *attn : [batch_size, ob_T, ob_T]
            # context = torch.matmul(attn, V)
            return  attn_self


class FF_self(nn.Module):
        def __init__(self,d_model,d_ff):
            super(FF_self, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),   # !d_ff 2048
                nn.ReLU(),
                nn.Linear(d_ff, d_model, bias=False)
            )
            self.ln = nn.LayerNorm(d_model)
        def forward(self, inputs):
            # !inputs: [batch_size, ob_T, d_model]
            residual = inputs
            output = self.fc(inputs)
            return self.ln(output + residual) # ![batch_size, ob_T, d_model]


class FF_inter(nn.Module):
        def __init__(self,d_model = 512,d_ff = 2048):
            super(FF_inter, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),   # !d_ff 2048
                nn.ReLU(),
                nn.Linear(d_ff, d_model, bias=False)
            )
            self.ln = nn.LayerNorm(d_model)
        def forward(self, inputs):
            # !inputs: [batch_size, Nn, ob_T, d_model]
            residual = inputs
            output = self.fc(inputs)
            return self.ln(output + residual) # ![batch_size, Nn, ob_T, d_model]


class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, d_k, d_v, n_heads):
            super(MultiHeadAttention, self).__init__()
            self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)   
            self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
            self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
            self.fc  = nn.Linear(n_heads * d_v, d_model, bias=False)
            self.ln = nn.LayerNorm(d_model)
            self.n_heads_sub = n_heads
            self.d_k_sub = d_k
            self.d_v_sub = d_v
        def forward(self, input_Q, input_K, input_V, attn_mask):
            # todo input_Q: [batch_size, Nn, ob_T, d_model]
            # todo input_K: [batch_size, Nn, ob_T, d_model]
            # todo input_V: [batch_size, Nn, ob_T, d_model]
            # todo attn_mask: [batch_size, Nn, ob_T, ob_T]--------------------------------  
            residual, batch_size, Nn = input_Q, input_Q.size(0),input_Q.size(1)
            Q = self.W_Q(input_Q).view(batch_size, Nn, -1, self.n_heads_sub, self.d_k_sub).transpose(2,3)  # !Q: [batch_size, Nn, n_heads, ob_T, d_k]
            K = self.W_K(input_K).view(batch_size, Nn, -1, self.n_heads_sub, self.d_k_sub).transpose(2,3)  # !K: [batch_size, Nn, n_heads, ob_T, d_k]
            V = self.W_V(input_V).view(batch_size, Nn, -1, self.n_heads_sub, self.d_v_sub).transpose(2,3)  # !V: [batch_size, Nn, n_heads, ob_T, d_v]

            attn_mask = attn_mask.unsqueeze(2).repeat(1, 1, self.n_heads_sub, 1, 1) # !attn_mask : [batch_size, Nn, n_heads, ob_T, ob_T]

            # !res_inter: [batch_size, n_heads, ob_T, d_v], attn: [batch_size, n_heads, Nn, ob_T, len_k]
            res_inter, attn_inter = SDPATT_inter()(Q, K, V, attn_mask)   # ![batch_size, Nn, n_heads, ob_T, d_v]
            res_inter = res_inter.transpose(2,3).reshape(batch_size, Nn, -1, self.n_heads_sub * self.d_v_sub) # !res_inter: [batch_size, Nn, ob_T, n_heads * d_v]
            output = self.fc(res_inter) # ![batch_size, Nn, ob_T, d_model]
            return self.ln(output + residual), attn_inter     # todo Residual connection with normalization


class SIlayer(nn.Module):
    def __init__(self,d_model, d_k, d_v, n_heads,d_ff):
            super(SIlayer, self).__init__()
            self.SDPATT_self = SDPATT_self()
            self.MHA = MultiHeadAttention(d_model, d_k, d_v, n_heads)
            self.SI = nn.Linear(n_heads * d_model, d_model, bias=False)
            self.FF_inter = FF_inter(d_model,d_ff)
            self.FF_self = FF_self(d_model,d_ff)
            self.SIlayer_heads = n_heads
            self.SIlayer_d_model = d_model


    def forward(self,x_emb,neighbor_emb,attn_mask,att_factor):
        # print('--------x_emb-----------',x_emb.shape,neighbor_emb.shape)
        #* x_emb [N, ob_T, d_model]   neighbor_emb  [N, Nn, ob_T, d_model]
        N = x_emb.size(0)
        res_inter,att_inter = self.MHA(neighbor_emb, neighbor_emb, neighbor_emb, attn_mask)
        #* res_inter [N, Nn, ob_T, d_model]   neighbor_emb  [N, Nn, n_heads, ob_T, ob_T]
        att_inter = att_inter.mean(1)   # *[N, n_heads, ob_T, ob_T]
        att_self = self.SDPATT_self(x_emb,x_emb,x_emb).unsqueeze(1).repeat(1,self.SIlayer_heads,1,1) # *[N, n_heads, ob_T, ob_T]
        att = att_self + att_factor * att_inter
        x_emb = x_emb.unsqueeze(1)
        res_si = torch.matmul(att, x_emb)    # * res_si [N, n_heads, ob_T, d_model]
        # print('--------------',res_si.shape)
        x_emb = res_si.transpose(1,2).reshape(N,-1,self.SIlayer_heads*self.SIlayer_d_model)
        x_emb = self.SI(x_emb)
        x_emb = self.FF_self(x_emb)
        neighbor_emb = self.FF_inter(res_inter)
        return x_emb,neighbor_emb


class SIencoder(nn.Module):
    def __init__(self,n_layers,d_model, d_k, d_v, n_heads,d_ff):
            super(SIencoder, self).__init__()

            self.layers = nn.ModuleList([SIlayer(d_model,d_k,d_v,n_heads,d_ff) for _ in range(n_layers)])
    def forward(self,x_emb,neighbor_emb,attn_mask,att_factor):

        for layer in self.layers:
            x_emb,neighbor_emb = layer(x_emb,neighbor_emb,attn_mask,att_factor)

        return x_emb,neighbor_emb


class SocialTrans(torch.nn.Module):
    def __init__(self,x_in,neighbor_in,ob_T , pred_T , ob_radius = 2):
        super(SocialTrans,self).__init__()
        self.ob_radius = ob_radius
        self.horizon_ob = ob_T
        self.horizon_pred = pred_T
        self.num_pred = 20
        self.att_factor = nn.Parameter(torch.rand(1))   # !tensor([0.5671])  Randomly drawn from a uniform distribution between 0 and 1
        # self.att_factor = torch.rand(1).item()
        self.d_model = 512  # ?Embedding Size
        self.d_ff = 2048 # ?FeedForward dimension
        self.d_k = self.d_v = 64  # ?dimension of K(=Q), V
        self.n_layers = 2  # ?number of Encoder of Decoder Layer
        self.n_heads = 8  # ?number of heads in Multi-Head Attention
        self.PosEn = PositionalEncoding(self.d_model,self.horizon_ob)
        self.SIencoder = SIencoder(self.n_layers,self.d_model, self.d_k, self.d_v, self.n_heads,self.d_ff)
        self.x_emb = nn.Linear(x_in, self.d_model)
        self.neighbor_emb = nn.Linear(neighbor_in, self.d_model)
        
        self.pred = nn.Linear(self.d_model,2)
        # self.pred_n = nn.ModuleList([nn.Linear(self.d_model,2) for _ in range(self.num_pred)])
    def forward(self,x,neighbor):
        # !x: ob_T x N x 6
        # !neighbor: ob_T x N x Nn x 6
        N = neighbor.size(1)
        Nn = neighbor.size(2)
        velocity = x[:,:,2:4].norm(dim=-1).unsqueeze(-1)
        v_angle = torch.atan(x[:,:,5]/x[:,:,4]).unsqueeze(-1)
        x_f  = torch.cat((x[:,:,0:2], velocity, v_angle),dim = -1)
        x_f = torch.nan_to_num(x_f, nan=0.0)
        # print('x_f-----------------------',x_f[:,:10,:])
        state = x
        x = state[...,:2]
        v = x[1:] - x[:-1]                      # *ob_T x N x 2          
        a = v[1:] - v[:-1]
        v = torch.cat((state[1:2,...,2:4], v))   # *ob_T x N x 2
        a = torch.cat((state[1:2,...,4:6], a))
        neighbor_x = neighbor[...,:2]
        neighbor_v = neighbor[...,2:4]
        dp = neighbor_x - x.unsqueeze(-2) # *ob_T x N x Nn x 2
        dv = neighbor_v - v.unsqueeze(-2) # *ob_T x N x Nn x 2
        dist = dp.norm(dim=-1) # *ob_T x N x Nn
        mask = dist <= self.ob_radius # *ob_T x N x Nn
        mask = mask.permute(1,2,0).contiguous()
        attn_mask = mask.unsqueeze(-1).repeat(1,1,1,self.horizon_ob)
        # todo attn_mask: [batch_size, Nn, ob_T, ob_T]
        dot_dp_v = (dp @ v.unsqueeze(-1)).squeeze(-1)  # *ob_T x N x Nn x 1 --> ob_T x N x Nn
        bearing = dot_dp_v / ( dist * (v.norm(dim=-1).unsqueeze(-1)))
        nan_indices = torch.isnan(bearing)
        bearing[nan_indices] = 0
        dot_dp_dv = (dp.unsqueeze(-2) @ dv.unsqueeze(-1)).view(dp.size(0),N,Nn)      #   
        tau = -dot_dp_dv / dv.norm(dim=-1)              # *ob_T x N x Nn
        tau[torch.isnan(tau)] = 0.0
        tau = torch.clamp(tau, 0, 7)            # !Limit the output of tau between 0-7
        mpd = (dp + tau.unsqueeze(-1)*dv).norm(dim=-1)  # *ob_T x N x Nn
        neighbor_f = torch.stack((dist, bearing, mpd), -1) # *ob_T x N x Nn x 3
        neighbor_f = torch.nan_to_num(neighbor_f, nan=0.0)
        # print('neighbor_f-----------------------',neighbor_f[:,:10,:])
        # !--------------------------------Feature extraction completed------------------------------------------------
        x_emb = self.x_emb(x_f)  # *ob_T x N x 512
        neighbor_emb = self.neighbor_emb(neighbor_f)

        x_emb = self.PosEn(x_emb)
        for i in range(neighbor_emb.size(2)):
            neighbor_emb[:,:,i,:] = self.PosEn(neighbor_emb[:,:,i,:])
        x_emb = x_emb.transpose(0,1)
        neighbor_emb = neighbor_emb.permute(1,2,0,3).contiguous()
        #* x_emb [N, ob_T, d_model]   neighbor_emb  [N, Nn, ob_T, d_model]
        x_emb,neighbor_emb = self.SIencoder(x_emb,neighbor_emb,attn_mask,self.att_factor)
        # print()
        # !---------------------Single trajectory prediction
        # output = self.pred(x_emb)
        # return output.transpose(0,1)   #* x_emb [ob_T,N, 2]
        # output = torch.nan_to_num(output, nan=0.0)
        # !--------------------Multiple trajectory predictions with multiple Linear layers
        # outputs = []
        # for layer in self.pred_n:
        #     outputs.append(layer(x_emb))
        # return torch.stack(outputs,dim=0).transpose(1,2)   #* x_emb [n_pred, ob_T,  N, 2]
        output = self.pred(x_emb).unsqueeze(0).repeat(self.num_pred,1,1,1)
        return output.transpose(1,2)   #* x_emb [20, ob_T,N, 2]
        # output = torch.nan_to_num(output, nan=0.0)

    def loss(self,y_, y, batch_first=False):
        # !y_, y: S x L x N x 2
        if torch.is_tensor(y):
            err = (y_ - y).norm(dim=-1) # !S x L x N
        else:
            err = np.linalg.norm(np.subtract(y_, y), axis=-1)
        if len(err.shape) == 1:
            fde = err[-1]
            ade = err.mean()
        elif batch_first:
            fde = err[..., -1]
            ade = err.mean(-1)
        else:
            # fde = err[..., -2, :]
            # ade = err[..., 0:-2, :].mean(-2)

            fde = err[..., -1, :]
            ade = err.mean(-2)  # !S x N
        # print('\n','ade fde')
        # print(ade.size())
        # print(fde.size())
        if len(y_.size()) == 3:
            ade = torch.min(ade, dim=0)[0]
            fde = torch.min(fde, dim=0)[0]
            return {'ade':ade, 'fde':fde}
        else:
            ade = torch.min(ade, dim=0)[0].mean()
            fde = torch.min(fde, dim=0)[0].mean()
            return {'ade':ade, 'fde':fde}

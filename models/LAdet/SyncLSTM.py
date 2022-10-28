import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SyncLSTM(nn.Module):
    def __init__(self, channel_size = 256, spatial_size = 32, k = 3, TM_Flag = False, compressed_size = 64):
        super(SyncLSTM, self).__init__()
        self.k = k
        self.spatial_size = spatial_size
        self.channel_size = channel_size
        self.compressed_size = compressed_size
        self.lstmcell = MotionLSTM(32, self.compressed_size)
        self.init_c = nn.parameter.Parameter(torch.rand(self.compressed_size, spatial_size, spatial_size))
        self.TM_Flag = TM_Flag

        self.ratio = int(math.sqrt(channel_size / compressed_size))
        self.conv_pre_1 = nn.Conv2d(self.channel_size, self.ratio * self.compressed_size, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(self.ratio * self.compressed_size, self.compressed_size, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(self.ratio * self.compressed_size)
        self.bn_pre_2 = nn.BatchNorm2d(self.compressed_size)
        self.conv_after_1 = nn.Conv2d(self.compressed_size, self.ratio * self.compressed_size, kernel_size=3, stride=1, padding=1)
        self.conv_after_2 = nn.Conv2d(self.ratio * self.compressed_size, self.channel_size, kernel_size=3, stride=1, padding=1)
        self.bn_after_1 = nn.BatchNorm2d(self.ratio * self.compressed_size)
        self.bn_after_2 = nn.BatchNorm2d(self.channel_size)
        
    def forward(self, x_raw, delta_t):
        batch, seq, channel, h, w = x_raw.shape
        if self.compressed_size != self.channel_size:
            x = F.relu(self.bn_pre_1(self.conv_pre_1(x_raw.view(-1,channel,h,w))))
            x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))
            x = x.view(batch, seq, self.compressed_size, h, w)
        else:
            x = x_raw
        
        if delta_t[0] > 0:
            self.delta_t = delta_t[0]
            h = x[:,0]
            c = self.init_c
            for i in range(1, self.k):
                h,c = self.lstmcell(x[:,i], (h,c))
            for t in range(int(self.delta_t - 1)):
                h,c = self.lstmcell(h, (h,c))
            else:
                res = h
        else:
            res = x[:,-1]
        if self.compressed_size != self.channel_size:
            res = F.relu(self.bn_after_1(self.conv_after_1(res)))
            res = F.relu(self.bn_after_2(self.conv_after_2(res)))
            # res = res.view(batch,channel,h,w)
        else:
            res = res
        return res.unsqueeze(1)

class MotionLSTM(nn.Module):
    def __init__(self, spatial_size, input_channel_size, hidden_size = 0):
        super().__init__()
        self.input_channel_size = input_channel_size  # channel size
        self.hidden_size = hidden_size
        self.spatial_size = spatial_size

        #i_t 
        # self.U_i = nn.Parameter(torch.Tensor(input_channel_size, hidden_size)) 
        # self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 
        
        self.U_i = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.V_i = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.b_i = nn.Parameter(torch.Tensor(1, self.input_channel_size, self.spatial_size, self.spatial_size))
        
        # #f_t 
        # self.U_f = nn.Parameter(torch.Tensor(input_channel_size, hidden_size)) 
        # self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 
        # self.b_f = nn.Parameter(torch.Tensor(hidden_size)) 
        self.U_f = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.V_f = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.b_f = nn.Parameter(torch.Tensor(1, self.input_channel_size, self.spatial_size, self.spatial_size))

        # #c_t 
        # self.U_c = nn.Parameter(torch.Tensor(input_channel_size, hidden_size)) 
        # self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 
        # self.b_c = nn.Parameter(torch.Tensor(hidden_size)) 
        self.U_c = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.V_c = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.b_c = nn.Parameter(torch.Tensor(1, self.input_channel_size, self.spatial_size, self.spatial_size))

        # #o_t 
        # self.U_o = nn.Parameter(torch.Tensor(input_channel_size, hidden_size)) 
        # self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 
        # self.b_o = nn.Parameter(torch.Tensor(hidden_size)) 
        self.U_o = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.V_o = STPN_MotionLSTM(height_feat_size = self.input_channel_size)
        self.b_o = nn.Parameter(torch.Tensor(1, self.input_channel_size, self.spatial_size, self.spatial_size))

        # self.init_weights()

    # def init_weights(self):
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)

    def forward(self,x,init_states=None): 
        """ 
        assumes x.shape represents (batch_size, sequence_size, input_channel_size) 
        """ 
        h, c = init_states 
        i = torch.sigmoid(self.U_i(x) + self.V_i(h) + self.b_i) 
        f = torch.sigmoid(self.U_f(x) + self.V_f(h) + self.b_f) 
        g = torch.tanh(self.U_c(x) + self.V_c(h) + self.b_c) 
        o = torch.sigmoid(self.U_o(x) + self.V_o(x) + self.b_o) 
        c_out = f * c + i * g 
        h_out = o *  torch.tanh(c_out) 

        # hidden_seq.append(h_t.unsqueeze(0)) 

        # #reshape hidden_seq p/ retornar 
        # hidden_seq = torch.cat(hidden_seq, dim=0) 
        # hidden_seq = hidden_seq.transpose(0, 1).contiguous() 
        return (h_out, c_out)



class STPN_MotionLSTM(nn.Module):
    def __init__(self, height_feat_size = 16):
        super(STPN_MotionLSTM, self).__init__()



        # self.conv3d_1 = Conv3D(4, 8, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_2 = Conv3D(8, 8, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(height_feat_size, 2*height_feat_size, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(2*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(2*height_feat_size, 4*height_feat_size, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(4*height_feat_size, 4*height_feat_size, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(6*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(2*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(3*height_feat_size , height_feat_size, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(height_feat_size, height_feat_size, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(2*height_feat_size)
        self.bn1_2 = nn.BatchNorm2d(2*height_feat_size)

        self.bn2_1 = nn.BatchNorm2d(4*height_feat_size)
        self.bn2_2 = nn.BatchNorm2d(4*height_feat_size)

        self.bn7_1 = nn.BatchNorm2d(2*height_feat_size)
        self.bn7_2 = nn.BatchNorm2d(2*height_feat_size)

        self.bn8_1 = nn.BatchNorm2d(1*height_feat_size)
        self.bn8_2 = nn.BatchNorm2d(1*height_feat_size)

    def forward(self, x):

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))
        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_2, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))
        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x



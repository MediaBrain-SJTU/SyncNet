from threading import local
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from coperception.models.LAdet.SyncFormer import SyncFormer
from coperception.models.det.base import IntermediateModelBase
from .SyncLSTM import SyncLSTM
from .SyncFormer import SyncFormer
from .SyncBaseline import SyncBaseline
class SyncDiscoNet(IntermediateModelBase):
    """DiscoNet.

    https://github.com/ai4ce/DiscoNet

    Args:
        config (object): The config object.
        layer (int, optional): Collaborate on which layer. Defaults to 3.
        in_channels (int, optional): The input channels. Defaults to 13.
        kd_flag (bool, optional): Whether to use knowledge distillation. Defaults to True.
        num_agent (int, optional): The number of agents (including RSU). Defaults to 5.

    """

    def __init__(self, config, layer=3, in_channels=13, kd_flag=True, num_agent=5, compress_level=0, only_v2i=False, LA_Falg =True):
        super().__init__(config, layer, in_channels, kd_flag, num_agent, compress_level, only_v2i, LA_Flag=True)
        self.tau = config.tau
        self.k = config.k
        if self.layer == 3:
            self.pixel_weighted_fusion = PixelWeightedFusionSoftmax(256)
        elif self.layer == 2:
            self.pixel_weighted_fusion = PixelWeightedFusionSoftmax(128)
        if config.compensation_flag == 'SyncLSTM':
            self.compensation = SyncLSTM(channel_size=int(256/(2**compress_level)), k=self.k)
        if config.compensation_flag == 'SyncFormer':
            self.compensation = SyncFormer(channel_size=int(256/(2**compress_level)), k=self.k)
        if config.compensation_flag == 'Baseline':
            self.compensation = SyncBaseline()
        self.compensation_loss = nn.SmoothL1Loss(reduction='sum')
    def forward(self, bevs, bevs_supervision, trans_matrices, trans_matrices_newest, trans_matrices_supervision, num_agent_tensor, center_agent = 1, batch_size=1):
        """Forward pass.

        Args:
            bevs (tensor): BEV data
            trans_matrices (tensor): Matrix for transforming features among agents.
            num_agent_tensor (tensor): Number of agents to communicate for each agent.
            batch_size (int, optional): The batch size. Defaults to 1.

        Returns:
            result, all decoded layers, and fused feature maps if kd_flag is set.
            else return result and list of weights for each agent.
        s"""
        Batch, his_k, h, w, z = bevs.shape
        

        # self.test_trans(bevs[:,-1], 0, num_agent_tensor[0][0], trans_matrices, center_agent=1)
        if trans_matrices_newest[0][1][1][:,3][:3].abs().mean()>0.5:
            self.test_trans_newest(bevs[1], 0, num_agent_tensor[0][0], trans_matrices_newest[0][1], center_agent=1)
        batch_size = int(Batch / self.agent_num)
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch*num_agent, k, z, h, w)
        encoded_layers = self.u_encoder(bevs)

        # Compensation Module

        # Trans to the coordinate of each agent's newest frame
        encoded_features_async = torch.flip(encoded_layers[4],(2,)).view(Batch, his_k, encoded_layers[4].size(1), encoded_layers[4].size(2), encoded_layers[4].size(3))
        encoded_layers_supervision = self.u_encoder(bevs_supervision.permute(0, 1, 4, 2, 3))
        encoded_features_supervision = torch.flip(encoded_layers_supervision[4],(2,)).view(Batch, 1, encoded_layers[4].size(1), encoded_layers[4].size(2), encoded_layers[4].size(3))

        encoded_features_async_edge_list = []
        encoded_features_async_edge_index_list = []
        

        for batch in range(batch_size):
            for agent_id in range(num_agent_tensor[batch][0]):
                if agent_id != center_agent:  
                    index = batch_size * agent_id + batch
                    encoded_features_supervision[index][0] = self.trans_to_somewhere(encoded_features_supervision[index][0], trans_matrices_supervision[batch][agent_id][center_agent])
                    for t in range(his_k):
                        # encoded_features_async[index][t] = self.trans_to_somewhere(encoded_features_async[index][t], trans_matrices_newest[batch][agent_id][t])
                        encoded_features_async[index][t] = self.trans_to_somewhere(encoded_features_async[index][t], trans_matrices_newest[batch][agent_id][t])
                    if agent_id != center_agent:
                        encoded_features_async_edge_list.append(encoded_features_async[index]) 
                        encoded_features_async_edge_index_list.append(index)
        
        # encoded_features_async_edge = torch.cat([encoded_features_async[0:center_agent*batch_size], encoded_features_async[(center_agent+1)*batch_size:]])
        encoded_features_async_edge = torch.stack(encoded_features_async_edge_list, 0)
        encoded_features_supervision_edge = torch.cat([encoded_features_supervision[encoded_features_async_edge_index_list]])
        # get supervision feature
        

        # encoded_features_supervision_edge = torch.cat([encoded_features_supervision[0:center_agent*batch_size], encoded_features_supervision[(center_agent+1)*batch_size:]]).squeeze(1)
        

        encoded_features_sync = torch.zeros(encoded_features_supervision.shape).to(encoded_features_supervision.device)
        tau=[self.tau[i] for i in range(len(self.tau)) if i != center_agent]
        encoded_features_compensated_edge = self.compensation(encoded_features_async_edge, tau)
        
        encoded_features_sync[encoded_features_async_edge_index_list] = encoded_features_compensated_edge
        # encoded_features_sync[0:center_agent*batch_size] = encoded_features_compensated_edge[0:center_agent*batch_size]
        # encoded_features_sync[(center_agent+1)*batch_size:] = encoded_features_compensated_edge[center_agent*batch_size:]
        # encoded_features_sync[center_agent*batch_size:(center_agent+1)*batch_size] = encoded_features_async[center_agent*batch_size:(center_agent+1)*batch_size,-1].unsqueeze(1)

        device = bevs.device
        # feat_maps, size = super().get_feature_maps_and_size(encoded_layers)
        
        feat_maps = encoded_layers[3].view(Batch, his_k, encoded_layers[3].size(1), encoded_layers[3].size(2), encoded_layers[3].size(3))[:,-1].clone()
        size = feat_maps[0].unsqueeze(0).size()

        
        # feat_maps[0:center_agent*batch_size]=self.u_encoder.com_decompresser(encoded_features_compensated_edge[0:center_agent*batch_size])
        # feat_maps[(center_agent+1)*batch_size:]=self.u_encoder.com_decompresser(encoded_features_compensated_edge[center_agent*batch_size:])

        # feat_maps_supervision =  encoded_layers_supervision[3].view(Batch, 1, encoded_layers[3].size(1), encoded_layers[3].size(2), encoded_layers[3].size(3))
        feat_maps_supervision = encoded_features_supervision

        compensation_loss_f = F.smooth_l1_loss(encoded_features_compensated_edge, encoded_features_supervision_edge, reduction='sum') / encoded_features_compensated_edge.shape[0]
        if self.u_encoder.compress_level > 0:
            feat_maps = F.relu(self.u_encoder.bn_decompress(self.u_encoder.com_decompresser(torch.flip(encoded_features_sync.squeeze(1),(2,)))))
            feat_maps_supervision = F.relu(self.u_encoder.bn_decompress(self.u_encoder.com_decompresser(torch.flip(feat_maps_supervision.squeeze(1),(2,)))))
            compensation_loss_f_2 = F.smooth_l1_loss(feat_maps, feat_maps_supervision, reduction='sum') / feat_maps.shape[0]
            compensation_loss_f += compensation_loss_f_2
            feat_maps = torch.flip(feat_maps,(2,))
            feat_maps_supervision = torch.flip(feat_maps_supervision,(2,))
            # feat_maps = F.relu(self.u_encoder.bn_decompress(self.u_encoder.com_decompresser(encoded_features_sync.squeeze(1))))
            # feat_maps_supervision = F.relu(self.u_encoder.bn_decompress(self.u_encoder.com_decompresser(feat_maps_supervision.squeeze(1))))
        else:
            feat_maps = encoded_features_sync.squeeze(1)

        feat_list = super().build_feature_list(batch_size, feat_maps)
        feat_list_supervision = super().build_feature_list(batch_size, feat_maps_supervision)
        
        local_com_mat = super().build_local_communication_matrix(
            feat_list
        )  # [2 5 512 16 16] [batch, agent, channel, height, width]
        # local_com_mat_update = super().build_local_communication_matrix(
        #     feat_list
        # )  # to avoid the inplace operation
        
        local_com_mat_supervision = super().build_local_communication_matrix(
            feat_list_supervision
        )       


        local_com_mat_update, save_agent_weight_list = self.fusion(batch_size,num_agent_tensor,local_com_mat,trans_matrices,size,center_agent, LA_Flag=True, encoded_layers=encoded_layers)
        
        local_com_mat_update_supervision, _ = self.fusion(batch_size,num_agent_tensor,local_com_mat_supervision,trans_matrices,size,center_agent, LA_Flag=True)
        
        compensation_loss_h = F.smooth_l1_loss(local_com_mat_update, local_com_mat_update_supervision, reduction='sum') / local_com_mat_update.shape[0] / 16
        # local_com_mat_update[0] = encoded_layers[3][5]
        # weighted feature maps is passed to decoder
        # feat_fuse_mat = super().agents_to_batch(local_com_mat)
        encoded_layers_center = []
        for layer in encoded_layers:
            layer_center = layer.view(Batch, his_k, layer.shape[1], layer.shape[2], layer.shape[3])
            encoded_layers_center.append(layer_center[center_agent*batch_size:(center_agent+1)*batch_size,-1])
        # feat_fuse_mat = torch.flip(local_com_mat_update.unsqueeze(0),(2,))[:,0]
        feat_fuse_mat = super().agents_to_batch(local_com_mat_update.unsqueeze(1),agent_num=1)
        # feat_fuse_mat = encoded_layers[3][5].unsqueeze
        del encoded_layers_center[self.layer+1]
        decoded_layers = super().get_decoded_layers(
            encoded_layers_center, feat_fuse_mat, batch_size
        )
        x = decoded_layers[0]

        cls_preds, loc_preds, result = super().get_cls_loc_result(x) 
        result['compensation_loss_f'] = compensation_loss_f
        result['compensation_loss_h'] = compensation_loss_h
        if self.kd_flag == 1:
            return (result, *decoded_layers, feat_fuse_mat)
        else:
            return result
            # return result, save_agent_weight_list

    def test_trans(self, bevs, batch, num_agent, trans_matrices, center_agent=1):
        from PIL import Image
        import numpy as np
        # self.neighbor_feat_list = list()
        bevs_temp_list = super().build_feature_list(2, bevs)
        bevs_temp = torch.cat(bevs_temp_list,1).permute(0,1,4,2,3)

        bevs_tran = torch.zeros((num_agent, bevs_temp.shape[2], bevs_temp.shape[3], bevs_temp.shape[4]))
        for i in range(num_agent):
            bevs_tran[i]=self.trans_to_somewhere(bevs_temp[batch,i], trans_matrices[batch][i])
            pc_pic = bevs_tran[i].permute(1,2,0).sum(2)
            pic_arr = np.array(pc_pic.detach().cpu())
            pic_arr = np.where(pic_arr>2, 2, pic_arr)
            pic_arr = pic_arr / pic_arr.max() * 255
            pc_im = Image.fromarray(pic_arr)
            pc_im = pc_im.convert('L')
            pc_im.save('./verify/{}.jpg'.format(str(i)))

        for i in range(num_agent):
            pc_pic = bevs_temp[batch][i].permute(1,2,0).sum(2)
            pic_arr = np.array(pc_pic.detach().cpu())
            pic_arr = np.where(pic_arr>2, 2, pic_arr)
            pic_arr = pic_arr / pic_arr.max() * 255
            pc_im = Image.fromarray(pic_arr)
            pc_im = pc_im.convert('L')
            pc_im.save('./verify/raw_{}.jpg'.format(str(i)))

        pc_pic = np.zeros((256,256))
        for i in range(num_agent):
            pc_pic = pc_pic + np.array(bevs_tran[i].permute(1,2,0).sum(2).detach().cpu())
        pc_pic = np.where(pic_arr>4, 4, pic_arr)
        pc_pic = pc_pic / pc_pic.max() * 255
        
        pc_im = Image.fromarray(pc_pic)
        pc_im = pc_im.convert('L')
        pc_im.save('./verify/all.jpg'.format(str(i)))

    
    def test_trans_newest(self, bevs, batch, num_agent, trans_matrices, center_agent=1):
        from PIL import Image
        import numpy as np
        bevs = bevs.permute(0,3,1,2)
        bevs = torch.flip(bevs,(2,))
        k = bevs.shape[0]
        bevs_tran = torch.zeros(bevs.shape)
              
        for i in range(k):
            if i < k-1:
                bevs_tran[i] = self.trans_to_somewhere(bevs[i], trans_matrices[i])
            else:
                bevs_tran[i] = bevs[i]
            pc_pic = bevs_tran[i].sum(0)
            pic_arr = np.array(pc_pic.detach().cpu())
            pic_arr = np.where(pic_arr>2, 2, pic_arr)
            pic_arr = pic_arr / pic_arr.max() * 255
            pc_im = Image.fromarray(pic_arr)
            pc_im = pc_im.convert('L')
            pc_im.save('./verify/{}.jpg'.format(str(i)))

        for i in range(k):
            pc_pic = bevs[i].sum(0)
            pic_arr = np.array(pc_pic.detach().cpu())
            pic_arr = np.where(pic_arr>2, 2, pic_arr)
            pic_arr = pic_arr / pic_arr.max() * 255
            pc_im = Image.fromarray(pic_arr)
            pc_im = pc_im.convert('L')
            pc_im.save('./verify/raw_{}.jpg'.format(str(i)))

        pc_pic = np.zeros((256,256))
        for i in range(k):
            pc_pic = pc_pic + np.array(bevs_tran[i].sum(0).detach().cpu())
        pc_pic = np.where(pic_arr>6, 6, pic_arr)
        pc_pic = pc_pic / pc_pic.max() * 255
        
        pc_im = Image.fromarray(pc_pic)
        pc_im = pc_im.convert('L')
        pc_im.save('./verify/all.jpg'.format(str(i)))

    
    def trans_to_somewhere(self, nb_agent, tfm_ji):
        nb_agent = nb_agent.unsqueeze(0)
        # nb_agent = torch.unsqueeze(local_com_mat[b, j], 0)

        # tfm_ji = trans_matrices[b, j, agent_idx]
        size = nb_agent.size()
        M = (
            torch.hstack((tfm_ji[:2, :2], -tfm_ji[:2, 3:4])).float().unsqueeze(0)
        )  # [1,2,3]

        mask = torch.tensor([[[1, 1, 4 / 128], [1, 1, 4 / 128]]], device=M.device)
        

        M *= mask

        grid = F.affine_grid(M, size=torch.Size(size))
        warp_feat = F.grid_sample(nb_agent, grid).squeeze()
        
        return warp_feat

    def fusion(self, batch_size, num_agent_tensor, local_com_mat, trans_matrices, size, center_agent = 1 ,LA_Flag = True,encoded_layers=0):
        device = local_com_mat.device
        local_com_mat = local_com_mat.view(batch_size,local_com_mat.shape[1],local_com_mat.shape[-3],local_com_mat.shape[-2],local_com_mat.shape[-1])
        local_com_mat_update = local_com_mat[:,center_agent].clone()
        size = local_com_mat_update[0].unsqueeze(0).size()
        save_agent_weight_list = list()
        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            # for i in range(num_agent):
            i = center_agent
            tg_agent = local_com_mat[b, i]
            all_warp = trans_matrices[b]  # transformation [2 5 5 4 4]

            self.neighbor_feat_list = list()
            self.neighbor_feat_list.append(tg_agent)

            if super().outage():
                agent_wise_weight_feat = self.neighbor_feat_list[0]
            else:
                super().build_neighbors_feature_list(
                    b,
                    i,
                    all_warp,
                    num_agent,
                    local_com_mat,
                    device,
                    size,
                    trans_matrices,
                    LA_Flag = LA_Flag
                )

                # agent-wise weighted fusion
                tmp_agent_weight_list = list()
                sum_weight = 0
                nb_len = len(self.neighbor_feat_list)
                for k in range(nb_len):
                    cat_feat = torch.cat(
                        [tg_agent, self.neighbor_feat_list[k]], dim=0
                    )
                    cat_feat = cat_feat.unsqueeze(0)
                    agent_weight = torch.squeeze(
                        self.pixel_weighted_fusion(cat_feat)
                    )
                    tmp_agent_weight_list.append(torch.exp(agent_weight))
                    sum_weight = sum_weight + torch.exp(agent_weight)

                agent_weight_list = list()
                for k in range(nb_len):
                    agent_weight = torch.div(tmp_agent_weight_list[k], sum_weight)
                    agent_weight.expand([256, -1, -1])
                    agent_weight_list.append(agent_weight)

                agent_wise_weight_feat = 0
                for k in range(nb_len):
                    agent_wise_weight_feat = (
                        agent_wise_weight_feat
                        + agent_weight_list[k] * self.neighbor_feat_list[k]
                    )

            # feature update
            local_com_mat_update[b] = agent_wise_weight_feat

            save_agent_weight_list.append(agent_weight_list)
        return local_com_mat_update, save_agent_weight_list
        # weighted feature maps is passed to decoder
        # feat_fuse_mat = super().agents_to_batch(local_com_mat)


class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel * 2, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1



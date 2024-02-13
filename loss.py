import sys

import torch
import numpy as np
import torch.nn.functional as F
from collections import namedtuple
from torchvision import models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss

class ProxyMetricLoss(nn.Module):
    # proxy based metric loss that tries to position proxies and push the embeddings so that the embeddings of anomalies
    # are close to positive proxies and the embeddings of normal samples are close to negative proxies. The
    def __init__(self):
        super(ProxyMetricLoss, self).__init__()
        chns = [128, 256, 512, 512, 512]
        self.positive_proxies1 = torch.nn.Parameter(torch.zeros(1,chns[0],10),requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies1, a=2.2361)
        self.positive_proxies2 = torch.nn.Parameter(torch.zeros(1,chns[1],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies2, a=2.2361)
        self.positive_proxies3 = torch.nn.Parameter(torch.zeros(1,chns[2],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies3, a=2.2361)
        self.positive_proxies4 = torch.nn.Parameter(torch.zeros(1,chns[3],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies4, a=2.2361)
        self.positive_proxies5 = torch.nn.Parameter(torch.zeros(1,chns[4],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies5, a=2.2361)

        self.positive_proxies = [self.positive_proxies1, self.positive_proxies2, self.positive_proxies3, self.positive_proxies4, self.positive_proxies5]
        self.negative_proxies1 = torch.nn.Parameter(torch.zeros(1,chns[0],10), requires_grad=True)
        torch.nn.init.normal_(self.negative_proxies1, 0, 0.2)
        torch.nn.init.kaiming_uniform_(self.negative_proxies1, a=2.2361)
        self.negative_proxies2 = torch.nn.Parameter(torch.zeros(1,chns[1],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies2, a=2.2361)
        self.negative_proxies3 = torch.nn.Parameter(torch.zeros(1,chns[2],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies3, a=2.2361)
        self.negative_proxies4 = torch.nn.Parameter(torch.zeros(1,chns[3],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies4, a=2.2361)
        self.negative_proxies5 = torch.nn.Parameter(torch.zeros(1,chns[4],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies5, a=2.2361)

        self.negative_proxies = [self.negative_proxies1, self.negative_proxies2, self.negative_proxies3, self.negative_proxies4, self.negative_proxies5]

        self.anomaly_thresholds = [0.4,0.3,0.2,0.1,0.1]
    def proxy_sim_reg_loss(self, pos_prox_l2, neg_prox_l2):
        # we want to make sure that negative and positive proxies are far away from each other. -- and maybe even each other?
        # get the l2 circle distance between negative and positive proxies and maximize
        neg_prox_l2_T = neg_prox_l2.transpose(1,2)
        pos_prox_l2_T = pos_prox_l2.transpose(1,2)
        # get b x c x c -- similarities
        Fcos = neg_prox_l2_T.matmul(pos_prox_l2)

        F_l2 = 2 - 2*Fcos
        #F_l2 = (2 - 2*Fcos) * (1.0 - torch.eye(10))
        Fcos_pospos = pos_prox_l2_T.matmul(pos_prox_l2)
        Fcos_negneg = neg_prox_l2_T.matmul(neg_prox_l2)
        #F_l2pos = 2 - 2 * Fcos_pospos
        #F_l2neg = 2 - 2 * Fcos_negneg
        reg_loss_pos = torch.mean(Fcos_pospos)
        reg_loss_neg = torch.mean(Fcos_negneg)
        reg_loss = torch.mean(Fcos)
        return reg_loss + reg_loss_pos + reg_loss_neg


    def forward(self, x, t, ds, ind):
        negative_proxies = self.negative_proxies[ind]
        positive_proxies = self.positive_proxies[ind]
        #t_ds = torch.nn.functional.max_pool2d(t,ds)
        t_ds_m = torch.nn.functional.avg_pool2d(t,ds)
        # 0.15 works worse than  0.3 probably since it gets confused as to what constitutes an anomaly since 0.15 is very low
        # and it's hard to distiguish between 90% normal example with a little transparent dot in the middle.. especially
        # in classes like tile..
        t_ds = torch.where(t_ds_m >= 0.3, torch.ones_like(t_ds_m), torch.zeros_like(t_ds_m))
        # TODO add a weight to the contribution of each sample. Since it isn't really obvious how much a sample
        # that for example only covers 20% of the regions, should count. Hard to directly classify.. Maybe just ignore
        # the regions between 0.15 and 0.3. ? Maybe just se the t_ds threshold to 0.1 and multiply each loss contribution
        # by the portion of the image covered
        t_keep = torch.where((t_ds_m < 0.3) & (t_ds_m > 0.15), torch.zeros_like(t_ds_m), torch.ones_like(t_ds_m))


        # print("----------------")
        # print(x.shape)
        # reshape to b x c x w*h
        x_flat = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3] ))
        # l2 normalize embedding vectors
        div = (2 * (torch.sum(x_flat**2, dim=1, keepdim=True)+1e-12)**0.5)
        Fl2 = x_flat / div

        div_pos_prox = (2 * (torch.sum(positive_proxies**2, dim=1, keepdim=True)+1e-12)**0.5)
        pos_prox_l2 = positive_proxies / div_pos_prox

        div_neg_prox = (2 * (torch.sum(negative_proxies ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        neg_prox_l2 = negative_proxies / div_neg_prox

        # transpose to b x w*h x c
        Fl2_T = Fl2.transpose(1,2)
        # multiply b x w*h x c   times b x c x P
        # to get cosine similarity between all embeddings n^2
        # print(Fl2_T.shape)
        # print(Fl2.shape)
        Fcos_pos = Fl2_T.matmul(pos_prox_l2)
        Fcos_neg = Fl2_T.matmul(neg_prox_l2)
        # print(Fcos.shape)
        # print(torch.max(Fcos))
        # print(torch.min(Fcos))
        # get l2 dist since L2dist = ||A||^2 + ||B||^2 - 2 AT B
        # since A and B have been normalized and ||A||^2 = 1, its L2 = 2 - 2 AT B
        # and cos sim = ATB / ||A||^2 ||B||^2 = ATB
        # L2 = 2 - 2 cos sim = 2 (1 - cos sim)
        # of shape b x w*h x P
        Fl2dist_pos = 2*(1-Fcos_pos)
        Fl2dist_neg = 2*(1-Fcos_neg)
        # print(torch.max(Fl2dist))
        # print(torch.min(Fl2dist))

        # get mask targets per pixel in form b x 1 x w*h
        t_flat = torch.reshape(t_ds, (x.shape[0], t_ds.shape[1], t_ds.shape[2]* t_ds.shape[3]))
        t_flat_keep = torch.reshape(t_keep, (x.shape[0], t_keep.shape[1], t_keep.shape[2]* t_keep.shape[3]))
        # transpose to b x w*h x 1
        t_flat_T = t_flat.transpose(1, 2)
        t_flat_keep_T = t_flat_keep.transpose(1, 2)
        # b x w*h * P
        # dist from positive points to positive proxies - min should be small
        masked_dist_pos_pos = Fl2dist_pos * t_flat_T
        # dist from positive points to negative proxies - min should be large
        masked_dist_pos_neg = Fl2dist_neg * t_flat_T
        # dist from negative points to positive proxies - min should be large
        masked_dist_neg_pos = Fl2dist_pos * (1-t_flat_T)
        # dist from negative points to negative proxies - min should be small
        masked_dist_neg_neg = Fl2dist_neg * (1-t_flat_T)


        # find closest positive and negative proxies for each positive or negative point and balance it according to the amount of
        # the region covered by the anomaly
        md_pp_min = torch.min(masked_dist_pos_pos, dim=2, keepdim=True)[0]
        md_pn_min = torch.min(masked_dist_pos_neg, dim=2, keepdim=True)[0]
        md_np_min = torch.min(masked_dist_neg_pos, dim=2, keepdim=True)[0]
        md_nn_min = torch.min(masked_dist_neg_neg, dim=2, keepdim=True)[0]

        #mean over all positive points - how close is the avg. pos. point to it's nearest pos. proxy - should be close
        #md_pp_mean = torch.sum(md_pp_min*t_flat_keep_T) / (torch.sum(t_flat_T*t_flat_keep_T)+1e-12)
        md_pp_mean = torch.sum(md_pp_min) / (torch.sum(t_flat_T)+1e-12)
        #mean over all positive points - how close is the avg. pos. point to it's nearest neg. proxy - should be far
        #md_pn_mean = torch.sum(md_pn_min*t_flat_keep_T) / (torch.sum(t_flat_T*t_flat_keep_T)+1e-12)
        md_pn_mean = torch.sum(md_pn_min) / (torch.sum(t_flat_T)+1e-12)
        #mean over all negative points - how close is the avg. neg. point to it's nearest pos. proxy - should be far
        #md_np_mean = torch.sum(md_np_min*t_flat_keep_T) / (torch.sum((1-t_flat_T)*t_flat_keep_T)+1e-12)
        md_np_mean = torch.sum(md_np_min) / (torch.sum((1-t_flat_T))+1e-12)
        #mean over all negative points - how close is the avg. neg. point to it's nearest neg. proxy - should be close
        #md_nn_mean = torch.sum(md_nn_min*t_flat_keep_T) / (torch.sum((1-t_flat_T)*t_flat_keep_T)+1e-12)
        md_nn_mean = torch.sum(md_nn_min) / (torch.sum((1-t_flat_T))+1e-12)

        reg_loss = self.proxy_sim_reg_loss(pos_prox_l2, neg_prox_l2)

        loss = md_pp_mean + md_nn_mean - md_np_mean - md_pn_mean + 0.1 * reg_loss
        return loss

class ProxyMetricNormalizeLoss(nn.Module):
    # proxy based metric loss that tries to position proxies to represent normal features and tries to push the
    # anomalous features to be mapped closer to the normal features by pushing them to the proxies
    def __init__(self):
        super(ProxyMetricNormalizeLoss, self).__init__()
        self.positive_proxies = torch.nn.Parameter(torch.zeros(1,512,100),requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies, a=2.2361)
        div_p = ((torch.sum(self.positive_proxies.data ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        self.positive_proxies.data = self.positive_proxies.data / div_p
        self.global_representation = torch.nn.Parameter(torch.zeros(1,1,100),requires_grad=False)

    def nearest_proxy_feature_map(self, x):
        sigma = 0.01
        la=5.0
        gamma = 1.0 / 0.1
        positive_proxies = self.positive_proxies
        # reshape to b x c x w*h
        x_flat = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3] ))
        # l2 normalize embedding vectors
        div = ((torch.sum(x_flat ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        Fl2 = x_flat / div
        # transpose to b x w*h x c
        Fl2_T = Fl2.transpose(1,2)

        div_pos_prox = ((torch.sum(positive_proxies**2, dim=1, keepdim=True)+1e-12)**0.5)
        pos_prox_l2 = positive_proxies / div_pos_prox

        # b x  w*h x c,  1 x c x K -> b x w*h x K
        Fcos_pos = Fl2_T.matmul(pos_prox_l2)

        # softmax for each positive and negative proxy
        # b x wh x K
        softmax_sim_pos = torch.softmax(gamma*Fcos_pos, dim=2)
        # b x wh
        softmax_sim_ind = torch.argmax(softmax_sim_pos, dim=2)
        # b x c x wh
        nearest_proxy_map = pos_prox_l2[0,:,softmax_sim_ind]
        nearest_proxy_map = torch.transpose(nearest_proxy_map, 0, 1)
        nearest_proxy_map = torch.transpose(nearest_proxy_map, 1, 2)
        diff_map = torch.sum(nearest_proxy_map * Fl2_T, dim=2)
        diff_map = torch.reshape(diff_map,(x.shape[2],x.shape[3]))
        return diff_map



        # sum over the similarities times the softmax of the similarities.. get a general similarity score for how
        # similar a certain embedding is to the class (positive- anomaly, or negative - normal).
        # b x wh x 1
        #Sim_pos = torch.sum(softmax_sim_pos * Fcos_pos, dim=2, keepdim=True)


    def get_global_rep_error(self, x):
        gamma = 10.0

        pos_proxy_d = self.positive_proxies.detach()
        div_pos_prox = ((torch.sum(pos_proxy_d**2, dim=1, keepdim=True)+1e-12)**0.5)
        pos_prox_l2 = pos_proxy_d / div_pos_prox

        x_flat = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        # l2 normalize embedding vectors
        div = ((torch.sum(x_flat ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        Fl2 = x_flat / div
        # transpose to b x w*h x c
        Fl2_T = Fl2.transpose(1, 2)
        # b x  w*h x c,  1 x c x K -> b x w*h x K
        Fcos_pos = Fl2_T.matmul(pos_prox_l2)

        # softmax for each positive proxy
        softmax_sim_pos = torch.softmax(gamma * (Fcos_pos), dim=2)
        global_rep_mean = torch.mean(softmax_sim_pos, dim=1, keepdim=True)
        global_rep_error = torch.mean(torch.abs(global_rep_mean - self.global_representation))
        return global_rep_error



    def global_rep_reg(self,Fl2_T, has_anomaly):
        gamma = 1.0 / 0.1
        pos_proxy_d = self.positive_proxies.detach()
        div_pos_prox = ((torch.sum(pos_proxy_d**2, dim=1, keepdim=True)+1e-12)**0.5)
        pos_prox_l2 = pos_proxy_d / div_pos_prox
        Fcos_pos = Fl2_T.matmul(pos_prox_l2)
        # b x wh x K
        softmax_sim_pos = torch.softmax(gamma*(Fcos_pos), dim=2)

        # b x wh
        softmax_sim_ind = torch.argmax(softmax_sim_pos, dim=2)
        # b x wh x F
        #nearest_proxy_map = pos_proxy_d[0,:,softmax_sim_ind]
        # b x 1 x F
        #global_mean_feat = torch.mean(nearest_proxy_map, dim=1, keepdim=True)
        # 1 x 1 x F
        #global_mean_feat = torch.mean(global_mean_feat, dim=0, keepdim=True)


        # b x 1 x K
        global_rep_mean = torch.mean(softmax_sim_pos, dim=1, keepdim=True)
        has_anomaly_t = torch.unsqueeze(has_anomaly,dim=2)
        global_rep_mean = global_rep_mean * (1.0 - has_anomaly_t)
        global_rep_mean = torch.sum(global_rep_mean,dim=0, keepdim=True) / (torch.sum((1.0-has_anomaly_t))+1e-12)
        global_rep_error = torch.mean(torch.abs(global_rep_mean - self.global_representation))
        if torch.sum(has_anomaly_t) > 0:
            self.global_representation.data = 0.99 * self.global_representation.data + 0.01 * global_rep_mean
        return global_rep_error

    def global_error_mask(self, x, Fl2_T):
        gamma = 1.0 / 0.1
        pos_proxy_d = self.positive_proxies.detach()
        div_pos_prox = ((torch.sum(pos_proxy_d**2, dim=1, keepdim=True)+1e-12)**0.5)
        pos_prox_l2 = pos_proxy_d / div_pos_prox
        Fcos_pos = Fl2_T.matmul(pos_prox_l2)
        # b x wh x K
        softmax_sim_pos = torch.softmax(gamma*(Fcos_pos), dim=2)
        global_rep_mean = torch.mean(softmax_sim_pos, dim=[1])
        # b, 1, K
        global_mean_error = torch.abs(self.global_representation.data - global_rep_mean)
        # b, 1, 1, K
        global_mean_error = torch.unsqueeze(global_mean_error, dim=1)
        softmax_mask = torch.reshape(softmax_sim_pos, (x.shape[0], x.shape[2], x.shape[3], softmax_sim_pos.shape[2]))
        # b, H, W, K
        softmax_mask = softmax_mask * global_mean_error
        # b, H, W
        error_mask = torch.max(global_mean_error, dim=3)
        return error_mask


    def proxy_sim_neg_compare(self, x,x_gt, t_flat_T):
        gamma = 1.0 / 0.1

        pos_proxy_d = self.positive_proxies.detach()
        div_pos_prox = ((torch.sum(pos_proxy_d**2, dim=1, keepdim=True)+1e-12)**0.5)
        pos_prox_l2 = pos_proxy_d / div_pos_prox

        # reshape to b x c x w*h
        x_gt_flat = torch.reshape(x_gt, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3] ))
        # l2 normalize embedding vectors
        div_gt = ((torch.sum(x_gt_flat ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        Fl2_gt = x_gt_flat / div_gt
        # transpose to b x w*h x c
        Fl2_gt_T = Fl2_gt.transpose(1,2)

        # reshape to b x c x w*h
        x_flat = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3] ))
        # l2 normalize embedding vectors
        div = ((torch.sum(x_flat ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        Fl2 = x_flat / div
        # transpose to b x w*h x c
        Fl2_T = Fl2.transpose(1,2)

        # b x 1 x w*h
        cos_sim_gt = torch.sum(Fl2 * Fl2_gt, dim=1, keepdim=True)
        t_flat = t_flat_T.transpose(1, 2)
        loss_sim_gt = torch.sum((1.0-cos_sim_gt) * t_flat) / (torch.sum(t_flat)+1e-12)


        # # b x  w*h x c,  1 x c x K -> b x w*h x K
        # Fcos_pos = Fl2_T.matmul(pos_prox_l2)
        # F_gt_cos_pos = Fl2_gt_T.matmul(pos_prox_l2)
        #
        # # softmax for each positive proxy
        # softmax_sim_pos = torch.softmax(gamma*(Fcos_pos), dim=2)
        # softmax_sim_gt_pos = torch.softmax(gamma*(F_gt_cos_pos), dim=2)
        # # sum over the similarities times the softmax of the similarities.. get a general similarity score for how
        # # similar a certain embedding is to the class (positive- anomaly, or negative - normal).
        # # b x wh x 1
        # sim_pos_f = softmax_sim_pos * Fcos_pos
        # sim_pos_f_gt = softmax_sim_gt_pos * F_gt_cos_pos
        # sim_pos_f_dif = torch.abs(sim_pos_f - sim_pos_f_gt)
        # sim_pos_l1_dif = torch.sum(sim_pos_f_dif,dim=2,keepdim=True)
        # loss_pos_gt = torch.sum((sim_pos_l1_dif) * t_flat_T) / (torch.sum(t_flat_T) + 1e-12)

        return loss_sim_gt

    def proxy_sim_neg_map(self, x, t_flat_T):
        gamma = 1.0 / 0.1

        pos_proxy_d = self.positive_proxies.detach()
        div_pos_prox = ((torch.sum(pos_proxy_d**2, dim=1, keepdim=True)+1e-12)**0.5)
        pos_prox_l2 = pos_proxy_d / div_pos_prox

        # reshape to b x c x w*h
        x_flat = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3] ))
        # l2 normalize embedding vectors
        div = ((torch.sum(x_flat ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        Fl2 = x_flat / div
        # transpose to b x w*h x c
        Fl2_T = Fl2.transpose(1,2)
        # b x  w*h x c,  1 x c x K -> b x w*h x K
        Fcos_pos = Fl2_T.matmul(pos_prox_l2)

        # softmax for each positive proxy
        softmax_sim_pos = torch.softmax(gamma*(Fcos_pos), dim=2)
        # sum over the similarities times the softmax of the similarities.. get a general similarity score for how
        # similar a certain embedding is to the class (positive- anomaly, or negative - normal).
        # b x wh x 1
        Sim_pos = torch.sum(softmax_sim_pos * Fcos_pos, dim=2, keepdim=True)
        # only take into acount the anomalous features - minimize the distance but do not affect the proxies
        loss_pos = torch.sum((1.0 - Sim_pos) * t_flat_T) / (torch.sum(t_flat_T)+1e-12)
        return loss_pos


    def proxy_sim_reg_loss(self, pos_prox_l2):
        # we want to make sure that positive proxies are unrelated as to cover the maximum diversity of features
        pos_prox_l2_T = pos_prox_l2.transpose(1,2)
        # get b x c x c -- similarities
        Fcos = pos_prox_l2_T.matmul(pos_prox_l2)

        #F_l2 = 2 - 2*Fcos
        #F_l2 = (2 - 2*Fcos) * (1.0 - torch.eye(10))
        #F_l2pos = 2 - 2 * Fcos_pospos
        #F_l2neg = 2 - 2 * Fcos_negneg
        reg_loss = torch.mean(Fcos*(1-torch.eye(Fcos.shape[0]).cuda()))
        return reg_loss

    def forward(self, x,x_gt, t, ds, has_anomaly):
        #default settings in their code.. they say nothing about lambda in the paper but 20 is the default in their code..
        sigma = 0.01
        la=5.0
        gamma = 1.0 / 0.1
        positive_proxies = self.positive_proxies
        #t_ds = torch.nn.functional.max_pool2d(t,ds)
        t_ds_m = torch.nn.functional.avg_pool2d(t,ds)
        t_ds = torch.where(t_ds_m > 0.02, torch.ones_like(t_ds_m), torch.zeros_like(t_ds_m))
        # b x wh x 1
        t_flat = torch.reshape(t_ds, (x.shape[0], t_ds.shape[1], t_ds.shape[2]* t_ds.shape[3]))
        t_flat_T = t_flat.transpose(1, 2)

        # reshape to b x c x w*h
        x_flat = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3] ))
        # l2 normalize embedding vectors
        div = ((torch.sum(x_flat ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        Fl2 = x_flat / div
        # transpose to b x w*h x c
        Fl2_T = Fl2.transpose(1,2)

        div_pos_prox = ((torch.sum(positive_proxies**2, dim=1, keepdim=True)+1e-12)**0.5)
        pos_prox_l2 = positive_proxies / div_pos_prox

        # b x  w*h x c,  1 x c x K -> b x w*h x K
        Fcos_pos = Fl2_T.matmul(pos_prox_l2)

        # softmax for each positive and negative proxy
        softmax_sim_pos = torch.softmax(gamma*Fcos_pos, dim=2)
        # sum over the similarities times the softmax of the similarities.. get a general similarity score for how
        # similar a certain embedding is to the class (positive- anomaly, or negative - normal).
        # b x wh x 1
        Sim_pos = torch.sum(softmax_sim_pos * Fcos_pos, dim=2, keepdim=True)
        # only take into acount the normal features - minimize the distance
        loss_pos = torch.sum((1.0 - Sim_pos ) * (1.0-t_flat_T)) / (torch.sum((1.0-t_flat_T))+1e-12)
        loss_reg = self.proxy_sim_reg_loss(pos_prox_l2)
        loss_global = self.global_rep_reg(Fl2_T,has_anomaly)
        #loss_negative = self.proxy_sim_neg_map(x, t_flat_T)
        loss_negative = self.proxy_sim_neg_compare(x,x_gt, t_flat_T)
        loss = loss_pos + loss_negative + 0.1 * loss_reg + loss_global

        return loss






class ProxyMemoryFakeFeatLoss(nn.Module):
    # proxy based metric loss that tries to position proxies to represent normal features and tries to push the
    # anomalous features to be mapped closer to the normal features by pushing them to the proxies
    def __init__(self, feat_dim=512):
        super(ProxyMemoryFakeFeatLoss, self).__init__()
        self.positive_proxies = torch.nn.Parameter(torch.zeros(1,feat_dim,100),requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies, a=2.2361)
        self.min_bounds = None
        self.max_bounds = None

    def generate_aug_feat(self,x, p=0.2):
        anomaly_mask = (torch.rand(x.shape[0],1,x.shape[2],x.shape[3]) > 1.0-p).float().cuda()
        min_bounds = torch.unsqueeze(self.min_bounds,2)
        min_bounds = torch.unsqueeze(min_bounds,3)
        max_bounds = torch.unsqueeze(self.max_bounds,2)
        max_bounds = torch.unsqueeze(max_bounds,3)

        random_tensor = torch.rand(x.shape[0],512,x.shape[2],x.shape[3]).cuda()
        random_interpolation = min_bounds * random_tensor + max_bounds * (1.0 - random_tensor)

        final_feat = anomaly_mask * random_interpolation + x * (1.0 - anomaly_mask)

        return final_feat


    def get_proxy_bounds(self):
        pos_prox = self.positive_proxies.detach()
        min_feat,_ = torch.min(pos_prox, dim=2)
        max_feat,_ = torch.max(pos_prox, dim=2)
        self.min_bounds = min_feat * 0.8
        self.max_bounds = max_feat * 1.2

    def nearest_proxy_feature_map_dif(self, x):
        sigma = 0.01
        la=5.0
        gamma = 1.0 / 0.1
        positive_proxies = self.positive_proxies.detach()
        # reshape to b x c x w*h
        x_flat = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3] ))
        # l2 normalize embedding vectors
        div = ((torch.sum(x_flat ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        Fl2 = x_flat / div
        # transpose to b x w*h x c
        Fl2_T = Fl2.transpose(1,2)

        div_pos_prox = ((torch.sum(positive_proxies**2, dim=1, keepdim=True)+1e-12)**0.5)
        pos_prox_l2 = positive_proxies / div_pos_prox

        # b x  w*h x c,  1 x c x K -> b x w*h x K
        Fcos_pos = Fl2_T.matmul(pos_prox_l2)

        # softmax for each positive and negative proxy
        # b x wh x K
        softmax_sim_pos = torch.softmax(gamma*Fcos_pos, dim=2)
        # b x wh
        softmax_sim_ind = torch.argmax(softmax_sim_pos, dim=2)
        # b x c x wh
        nearest_proxy_map = pos_prox_l2[0,:,softmax_sim_ind]
        nearest_proxy_map = torch.transpose(nearest_proxy_map, 0, 1)
        nearest_proxy_map = torch.transpose(nearest_proxy_map, 1, 2)
        diff_map = torch.sum(nearest_proxy_map * Fl2_T, dim=2)
        diff_map = torch.reshape(diff_map,(x.shape[0],1,x.shape[2],x.shape[3]))

        return diff_map

    def proxy_sim_reg_loss(self, pos_prox_l2):
        # we want to make sure that positive proxies are unrelated as to cover the maximum diversity of features
        pos_prox_l2_T = pos_prox_l2.transpose(1,2)
        # get b x c x c -- similarities
        Fcos = pos_prox_l2_T.matmul(pos_prox_l2)

        #F_l2 = 2 - 2*Fcos
        #F_l2 = (2 - 2*Fcos) * (1.0 - torch.eye(10))
        #F_l2pos = 2 - 2 * Fcos_pospos
        #F_l2neg = 2 - 2 * Fcos_negneg
        reg_loss = torch.mean(Fcos*(1-torch.eye(Fcos.shape[0]).cuda()))
        return reg_loss



    def forward(self, x, ds):
        #default settings in their code.. they say nothing about lambda in the paper but 20 is the default in their code..
        sigma = 0.01
        la=5.0
        gamma = 1.0 / 0.1
        positive_proxies = self.positive_proxies
        #t_ds = torch.nn.functional.max_pool2d(t,ds)
        # b x wh x 1

        # reshape to b x c x w*h
        x_flat = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3] ))
        # l2 normalize embedding vectors
        div = ((torch.sum(x_flat ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        Fl2 = x_flat / div
        # transpose to b x w*h x c
        Fl2_T = Fl2.transpose(1,2)

        div_pos_prox = ((torch.sum(positive_proxies**2, dim=1, keepdim=True)+1e-12)**0.5)
        pos_prox_l2 = positive_proxies / div_pos_prox

        # b x  w*h x c,  1 x c x K -> b x w*h x K
        Fcos_pos = Fl2_T.matmul(pos_prox_l2)

        # softmax for each positive and negative proxy
        softmax_sim_pos = torch.softmax(gamma*Fcos_pos, dim=2)
        # sum over the similarities times the softmax of the similarities.. get a general similarity score for how
        # similar a certain embedding is to the class (positive- anomaly, or negative - normal).
        # b x wh x 1
        Sim_pos = torch.sum(softmax_sim_pos * Fcos_pos, dim=2, keepdim=True)
        # only take into acount the normal features - minimize the distance
        loss_pos = torch.mean(1.0 - Sim_pos )
        loss_reg = self.proxy_sim_reg_loss(pos_prox_l2)
        #loss_negative = self.proxy_sim_neg_map(x, t_flat_T)
        loss = loss_pos + 0.1 * loss_reg

        return loss







class ProxySoftmaxMetricLoss(nn.Module):
    # proxy based metric loss that tries to position proxies and push the embeddings so that the embeddings of anomalies
    # are close to positive proxies and the embeddings of normal samples are close to negative proxies. The
    def __init__(self):
        super(ProxySoftmaxMetricLoss, self).__init__()
        chns = [128, 256, 512, 512, 512]
        self.positive_proxies1 = torch.nn.Parameter(torch.zeros(1,chns[0],10),requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies1, a=2.2361)
        self.positive_proxies2 = torch.nn.Parameter(torch.zeros(1,chns[1],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies2, a=2.2361)
        self.positive_proxies3 = torch.nn.Parameter(torch.zeros(1,chns[2],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies3, a=2.2361)
        self.positive_proxies4 = torch.nn.Parameter(torch.zeros(1,chns[3],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies4, a=2.2361)
        self.positive_proxies5 = torch.nn.Parameter(torch.zeros(1,chns[4],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies5, a=2.2361)

        self.positive_proxies = [self.positive_proxies1, self.positive_proxies2, self.positive_proxies3, self.positive_proxies4, self.positive_proxies5]
        self.negative_proxies1 = torch.nn.Parameter(torch.zeros(1,chns[0],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies1, a=2.2361)
        self.negative_proxies2 = torch.nn.Parameter(torch.zeros(1,chns[1],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies2, a=2.2361)
        self.negative_proxies3 = torch.nn.Parameter(torch.zeros(1,chns[2],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies3, a=2.2361)
        self.negative_proxies4 = torch.nn.Parameter(torch.zeros(1,chns[3],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies4, a=2.2361)
        self.negative_proxies5 = torch.nn.Parameter(torch.zeros(1,chns[4],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies5, a=2.2361)

        self.negative_proxies = [self.negative_proxies1, self.negative_proxies2, self.negative_proxies3, self.negative_proxies4, self.negative_proxies5]

    def proxy_sim_reg_loss(self, pos_prox_l2, neg_prox_l2):
        # we want to make sure that negative and positive proxies are far away from each other. -- and maybe even each other?
        # get the l2 circle distance between negative and positive proxies and maximize
        neg_prox_l2_T = neg_prox_l2.transpose(1,2)
        pos_prox_l2_T = pos_prox_l2.transpose(1,2)
        # get b x c x c -- similarities
        Fcos = neg_prox_l2_T.matmul(pos_prox_l2)

        #F_l2 = 2 - 2*Fcos
        #F_l2 = (2 - 2*Fcos) * (1.0 - torch.eye(10))
        Fcos_pospos = pos_prox_l2_T.matmul(pos_prox_l2)
        Fcos_negneg = neg_prox_l2_T.matmul(neg_prox_l2)
        #F_l2pos = 2 - 2 * Fcos_pospos
        #F_l2neg = 2 - 2 * Fcos_negneg
        reg_loss_pos = torch.mean(Fcos_pospos)
        reg_loss_neg = torch.mean(Fcos_negneg)
        reg_loss = torch.mean(Fcos)
        return reg_loss + reg_loss_pos + reg_loss_neg

    def forward(self, x, t, ds, ind):
        #default settings in their code.. they say nothing about lambda in the paper but 20 is the default in their code..
        sigma = 0.01
        la=5.0
        gamma = 1.0 / 0.1
        negative_proxies = self.negative_proxies[ind]
        positive_proxies = self.positive_proxies[ind]
        #t_ds = torch.nn.functional.max_pool2d(t,ds)
        t_ds_m = torch.nn.functional.avg_pool2d(t,ds)
        t_ds = torch.where(t_ds_m > 0.3, torch.ones_like(t_ds_m), torch.zeros_like(t_ds_m))
        # b x wh x 1
        t_flat = torch.reshape(t_ds, (x.shape[0], t_ds.shape[1], t_ds.shape[2]* t_ds.shape[3]))
        t_flat_T = t_flat.transpose(1, 2)

        # reshape to b x c x w*h
        x_flat = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3] ))
        # l2 normalize embedding vectors
        div = ((torch.sum(x_flat ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        Fl2 = x_flat / div
        # transpose to b x w*h x c
        Fl2_T = Fl2.transpose(1,2)

        div_pos_prox = ((torch.sum(positive_proxies**2, dim=1, keepdim=True)+1e-12)**0.5)
        pos_prox_l2 = positive_proxies / div_pos_prox

        div_neg_prox = ((torch.sum(negative_proxies ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        neg_prox_l2 = negative_proxies / div_neg_prox


        # b x  w*h x c,  1 x c x K -> b x w*h x K
        Fcos_pos = Fl2_T.matmul(pos_prox_l2)
        Fcos_neg = Fl2_T.matmul(neg_prox_l2)

        # softmax for each positive and negative proxy
        softmax_sim_pos = torch.softmax(gamma*Fcos_pos, dim=2)
        softmax_sim_neg = torch.softmax(gamma*Fcos_neg, dim=2)



        # sum over the similarities times the softmax of the similarities.. get a general similarity score for how
        # similar a certain embedding is to the class (positive- anomaly, or negative - normal).
        # b x wh x 1
        Sim_pos = torch.sum(softmax_sim_pos * Fcos_pos, dim=2, keepdim=True)
        Sim_neg = torch.sum(softmax_sim_neg * Fcos_neg, dim=2, keepdim=True)

        # normalize so that the softmax calculation is more stable - we use the max of the sum of sim po and sim neg
        # but any value could be used as long as the e^x doesn't over and underfit
        Sim_pos_norm = Sim_pos - (Sim_pos+Sim_neg)/2.0
        Sim_neg_norm = Sim_neg - (Sim_pos+Sim_neg)/2.0


        # b x wh x 1
        #Sim_pos_pos = torch.exp(la*(Sim_pos-sigma))
        Sim_pos_pos = torch.exp(la*(Sim_pos_norm-sigma))
        #Sim_pos_neg = torch.exp(la*(Sim_neg))
        Sim_pos_neg = torch.exp(la*(Sim_neg_norm))
        #Sim_neg_pos = torch.exp(la*(Sim_pos))
        Sim_neg_pos = torch.exp(la*(Sim_pos_norm))
        #Sim_neg_neg = torch.exp(la*(Sim_neg-sigma))
        Sim_neg_neg = torch.exp(la*(Sim_neg_norm-sigma))

        loss_pos = -torch.log(Sim_pos_pos/(Sim_pos_pos + Sim_pos_neg+1e-12)) * t_flat_T
        loss_neg = -torch.log(Sim_neg_neg/(Sim_neg_neg + Sim_neg_pos+1e-12)) * (1.0 - t_flat_T)
        loss_sum_pos = torch.sum(loss_pos) / (torch.sum(t_flat_T)+1e-12)
        loss_sum_neg = torch.sum(loss_neg) / (torch.sum(1-t_flat_T)+1e-12)
        #loss_sum_pos = torch.mean(loss_pos)
        #loss_sum_neg = torch.mean(loss_neg)
        loss_reg = self.proxy_sim_reg_loss(pos_prox_l2, neg_prox_l2)
        loss = loss_sum_pos + loss_sum_neg + 0.1 * loss_reg

        return loss



class ProxyAnchorMetricLoss(nn.Module):
    # proxy based metric loss that tries to position proxies and push the embeddings so that the embeddings of anomalies
    # are close to positive proxies and the embeddings of normal samples are close to negative proxies. The
    def __init__(self):
        super(ProxyAnchorMetricLoss, self).__init__()
        chns = [128, 256, 512, 512, 512]
        self.positive_proxies1 = torch.nn.Parameter(torch.zeros(1,chns[0],10),requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies1, a=2.2361)
        self.positive_proxies2 = torch.nn.Parameter(torch.zeros(1,chns[1],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies2, a=2.2361)
        self.positive_proxies3 = torch.nn.Parameter(torch.zeros(1,chns[2],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies3, a=2.2361)
        self.positive_proxies4 = torch.nn.Parameter(torch.zeros(1,chns[3],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies4, a=2.2361)
        self.positive_proxies5 = torch.nn.Parameter(torch.zeros(1,chns[4],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.positive_proxies5, a=2.2361)

        self.positive_proxies = [self.positive_proxies1, self.positive_proxies2, self.positive_proxies3, self.positive_proxies4, self.positive_proxies5]
        self.negative_proxies1 = torch.nn.Parameter(torch.zeros(1,chns[0],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies1, a=2.2361)
        self.negative_proxies2 = torch.nn.Parameter(torch.zeros(1,chns[1],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies2, a=2.2361)
        self.negative_proxies3 = torch.nn.Parameter(torch.zeros(1,chns[2],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies3, a=2.2361)
        self.negative_proxies4 = torch.nn.Parameter(torch.zeros(1,chns[3],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies4, a=2.2361)
        self.negative_proxies5 = torch.nn.Parameter(torch.zeros(1,chns[4],10), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.negative_proxies5, a=2.2361)

        self.negative_proxies = [self.negative_proxies1, self.negative_proxies2, self.negative_proxies3, self.negative_proxies4, self.negative_proxies5]

    def proxy_sim_reg_loss(self, pos_prox_l2, neg_prox_l2):
        # we want to make sure that negative and positive proxies are far away from each other. -- and maybe even each other?
        # get the l2 circle distance between negative and positive proxies and maximize
        neg_prox_l2_T = neg_prox_l2.transpose(1,2)
        pos_prox_l2_T = pos_prox_l2.transpose(1,2)
        # get b x c x c -- similarities
        Fcos = neg_prox_l2_T.matmul(pos_prox_l2)

        #F_l2 = 2 - 2*Fcos
        #F_l2 = (2 - 2*Fcos) * (1.0 - torch.eye(10))
        Fcos_pospos = pos_prox_l2_T.matmul(pos_prox_l2)
        Fcos_negneg = neg_prox_l2_T.matmul(neg_prox_l2)
        #F_l2pos = 2 - 2 * Fcos_pospos
        #F_l2neg = 2 - 2 * Fcos_negneg
        reg_loss_pos = torch.mean(Fcos_pospos)
        reg_loss_neg = torch.mean(Fcos_negneg)
        reg_loss = torch.mean(Fcos)
        return reg_loss + reg_loss_pos + reg_loss_neg

    def forward(self, x, t, ds, ind):
        #default settings in their code.. they say nothing about lambda in the paper but 20 is the default in their code..
        sigma = 0.01
        la=2.0
        gamma = 1.0 / 0.1
        negative_proxies = self.negative_proxies[ind]
        positive_proxies = self.positive_proxies[ind]
        #t_ds = torch.nn.functional.max_pool2d(t,ds)
        t_ds_m = torch.nn.functional.avg_pool2d(t,ds)
        t_ds = torch.where(t_ds_m > 0.3, torch.ones_like(t_ds_m), torch.zeros_like(t_ds_m))
        # b x wh x 1
        t_flat = torch.reshape(t_ds, (x.shape[0], t_ds.shape[1], t_ds.shape[2]* t_ds.shape[3]))
        t_flat_T = t_flat.transpose(1, 2)

        # reshape to b x c x w*h
        x_flat = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3] ))
        # l2 normalize embedding vectors
        div = ((torch.sum(x_flat ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        Fl2 = x_flat / div
        # transpose to b x w*h x c
        Fl2_T = Fl2.transpose(1,2)

        div_pos_prox = ((torch.sum(positive_proxies**2, dim=1, keepdim=True)+1e-12)**0.5)
        pos_prox_l2 = positive_proxies / div_pos_prox

        div_neg_prox = ((torch.sum(negative_proxies ** 2, dim=1, keepdim=True) + 1e-12) ** 0.5)
        neg_prox_l2 = negative_proxies / div_neg_prox


        # b x  w*h x c,  1 x c x K -> b x w*h x K
        Fcos_pos = Fl2_T.matmul(pos_prox_l2)
        Fcos_neg = Fl2_T.matmul(neg_prox_l2)

        # b x w*h x K --> 1
        exp_sim_pos = torch.sum(torch.log(1+torch.sum(torch.exp(-la*(Fcos_pos-sigma)) * t_flat_T, dim=1))) / 10.0
        exp_sim_neg = torch.sum(torch.log(1+torch.sum(torch.exp(-la*(Fcos_pos-sigma)) * t_flat_T, dim=1))) / (torch.sum(t_flat_T))
        exp_sim_neg = torch.exp(-la*(Fcos_neg-sigma)) * (1.0 - t_flat_T)


        loss = exp_sim_neg
        return loss

class MetricLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, t, ds):

        t_ds = torch.nn.functional.avg_pool2d(t,ds)
        t_ds = torch.where(t_ds > 0.3, torch.ones_like(t_ds), torch.zeros_like(t_ds))

        # print("----------------")
        # print(x.shape)
        # reshape to b x c x w*h
        x_flat = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3] ))
        # l2 normalize embedding vectors
        div = (2 * (torch.sum(x_flat**2, dim=1, keepdim=True)+1e-12)**0.5)
        Fl2 = x_flat / div

        # transpose to b x w*h x c
        Fl2_T = Fl2.transpose(1,2)
        # multiply b x c x w*h  times b x w*h x c
        # to get cosine similarity between all embeddings n^2
        # print(Fl2_T.shape)
        # print(Fl2.shape)
        Fcos = Fl2_T.matmul(Fl2)
        # print(Fcos.shape)
        # print(torch.max(Fcos))
        # print(torch.min(Fcos))
        # get l2 dist since L2dist = ||A||^2 + ||B||^2 - 2 AT B
        # since A and B have been normalized and ||A||^2 = 1, its L2 = 2 - 2 AT B
        # and cos sim = ATB / ||A||^2 ||B||^2 = ATB
        # L2 = 2 - 2 cos sim = 2 (1 - cos sim)
        Fl2dist = 2*(1-Fcos)
        # print(torch.max(Fl2dist))
        # print(torch.min(Fl2dist))

        # get mask targets per pixel in form b x 1 x w*h
        t_flat = torch.reshape(t_ds, (x.shape[0], t_ds.shape[1], t_ds.shape[2]* t_ds.shape[3]))
        # transpose to b x w*h x 1
        t_flat_T = t_flat.transpose(1, 2)
        # multiply b x 1 x w*h times b x w*h x 1 -- we get 1 at locations where both are 1 for each channel
        Mp = t_flat_T.matmul(t_flat)
        # multiply b x 1 x w*h times b x w*h x 1 -- we get 1 at locations where both are 0
        Mn = (1-t_flat_T).matmul(1-t_flat)
        Md = (1- Mp - Mn)

        # Md, Mn, Mp are very biased, Md is large, Mn also large since there are a lot of normal samples, Mp is small.. weight Mp higher?
        # optimise so anomalies (Mp) are far away from each other, normal samples are close (Mn), and different samples are far (Md)
        mdm = torch.sum(Md*Fl2dist)/ (torch.sum(Md)+1e-12)
        mnm = torch.sum(Mn*Fl2dist)/ (torch.sum(Mn)+1e-12)
        mpm = torch.sum(Mp*Fl2dist)/ (torch.sum(Mp)+1e-12)
        loss = mnm - mdm
        return loss



        #convB_reshaped = convB_reshaped.transpose(1,2)


        #convB_reshaped = torch.reshape(convB,(convB.shape[0],convB.shape[1],convB.shape[2]*convB.shape[3]))
        #convB_reshaped = convB_reshaped.transpose(1,2)
        #convC_reshaped = torch.reshape(convC,(convC.shape[0],convC.shape[1],convC.shape[2]*convC.shape[3]))
        #convD_reshaped = torch.reshape(convD,(convD.shape[0],convD.shape[1],convD.shape[2]*convD.shape[3]))
        #BC_mul = convC_reshaped.matmul(convB_reshaped)



class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #self.encoder = torch.hub.load('pytorch/vision:v0.5.0', 'resnet34')
        self.resnet34 = resnet34(pretrained=True)

    def forward(self,x,y):
        c1x = self.resnet34.conv1(x)
        bn1x = self.resnet34.bn1(c1x)
        rel1x = self.resnet34.relu(bn1x)
        mp1x = self.resnet34.maxpool(rel1x)
        l1x = self.resnet34.layer1(mp1x)
        l2x = self.resnet34.layer2(l1x)
        l3x = self.resnet34.layer3(l2x)
        l4x = self.resnet34.layer4(l3x)

        c1y = self.resnet34.conv1(y)
        bn1y = self.resnet34.bn1(c1y)
        rel1y = self.resnet34.relu(bn1y)
        mp1y = self.resnet34.maxpool(rel1y)
        l1y = self.resnet34.layer1(mp1y)
        l2y = self.resnet34.layer2(l1y)
        l3y = self.resnet34.layer3(l2y)
        l4y = self.resnet34.layer4(l3y)

        l1x_div = torch.sqrt(torch.sum(l1x**2,dim=1, keepdim=True)+1e-12)
        l2x_div = torch.sqrt(torch.sum(l2x**2,dim=1, keepdim=True)+1e-12)
        l3x_div = torch.sqrt(torch.sum(l3x**2,dim=1, keepdim=True)+1e-12)
        l4x_div = torch.sqrt(torch.sum(l4x**2,dim=1, keepdim=True)+1e-12)
        l1y_div = torch.sqrt(torch.sum(l1y**2,dim=1, keepdim=True)+1e-12)
        l2y_div = torch.sqrt(torch.sum(l2y**2,dim=1, keepdim=True)+1e-12)
        l3y_div = torch.sqrt(torch.sum(l3y**2,dim=1, keepdim=True)+1e-12)
        l4y_div = torch.sqrt(torch.sum(l4y**2,dim=1, keepdim=True)+1e-12)
        l1y = l1y/l1y_div
        l2y = l2y/l2y_div
        l3y = l3y/l3y_div
        l4y = l4y/l4y_div
        l1x = l1x/l1x_div
        l2x = l2x/l2x_div
        l3x = l3x/l3x_div
        l4x = l4x/l4x_div

        l1dif = l1y-l1x
        l2dif = l2y-l2x
        l3dif = l3y-l3x
        l4dif = l4y-l4x
        l1dif_l2 = torch.sqrt(torch.sum(l1dif**2,dim=1, keepdim=True)+1e-12)
        l2dif_l2 = torch.sqrt(torch.sum(l2dif**2,dim=1, keepdim=True)+1e-12)
        l3dif_l2 = torch.sqrt(torch.sum(l3dif**2,dim=1, keepdim=True)+1e-12)
        l4dif_l2 = torch.sqrt(torch.sum(l4dif**2,dim=1, keepdim=True)+1e-12)


        loss = (torch.mean(l1dif_l2) + torch.mean(l2dif_l2) + torch.mean(l3dif_l2) + torch.mean(l4dif_l2)) / 4.0
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, pred, target):
        # both inputs from 0 to 1
        loss = 1.0 - 2 * torch.sum(target * pred) / (torch.sum(target ** 2) + torch.sum(pred ** 2) + 1e-6)
        return loss

def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1606.04797.pdf
        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class BinTopPercentLoss(nn.Module):
    def __init__(self,t):
        super(BinTopPercentLoss, self).__init__()
        self.t = t
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logit, target):
        bce_loss_anom = self.ce_loss(logit, target)
        num_pixels = torch.sum(torch.ones_like(target)) * self.t / 100.0
        ce_loss_lin = bce_loss_anom.view((-1, ))
        loss, _ = torch.topk(ce_loss_lin, num_pixels, sorted=False)
        loss = torch.mean(loss)
        return loss

class BinTopKLoss(nn.Module):
    def __init__(self,t):
        super(BinTopKLoss, self).__init__()
        self.t = t
        pass

    def forward(self, logit, target):
        bg_score = logit[:,:1,:,:]
        anomaly_score = logit[:,1:,:,:]
        target_bg = 1 - target

        bg_score_mask = torch.where(bg_score < self.t, torch.ones_like(bg_score), torch.zeros_like(bg_score)) * target_bg
        anom_score_mask = torch.where(anomaly_score < self.t, torch.ones_like(anomaly_score), torch.zeros_like(anomaly_score)) * target
        anom_loss = torch.log(anomaly_score) * anom_score_mask
        bg_loss = torch.log(bg_score) * bg_score_mask
        loss = - torch.sum(bg_loss + anom_loss) / torch.sum(anom_score_mask + bg_score_mask)
        return loss

class FocalLossMasked(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLossMasked, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target, mask):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        loss = loss * mask

        if self.size_average:
            loss = torch.sum(loss) / (torch.sum(mask)+1e-12)
            #loss = loss.mean()
        #else:
        #    loss = loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        #else:
        #    loss = loss.sum()
        return loss

def msgmsd(r,d):
    t_r = r.clone()
    t_d = d.clone()
    gms_lst = []
    for i in range(3):
        gms_t = gms(t_r,t_d)
        gmsm = torch.mean(gms_t)
        gmsd_t = (gms_t-gmsm)**2
        t_r = torch.nn.functional.avg_pool2d(t_r,(2,2))
        t_d = torch.nn.functional.avg_pool2d(t_d,(2,2))
        up_gms = 1.0-torch.nn.functional.upsample(gmsd_t, scale_factor=2**i)
        gms_lst.append(up_gms)
    return gms_lst
def msgms(r,d):
    t_r = r.clone()
    t_d = d.clone()
    gms_lst = []
    for i in range(3):
        gms_t = gms(t_r,t_d)
        t_r = torch.nn.functional.avg_pool2d(t_r,(2,2))
        #t_r = torch.nn.functional.max_pool2d(t_r,(2,2))
        t_d = torch.nn.functional.avg_pool2d(t_d,(2,2))
        #t_d = torch.nn.functional.max_pool2d(t_d,(2,2))
        up_gms = torch.nn.functional.upsample(gms_t, scale_factor=2**i)
        gms_lst.append(up_gms)
    return gms_lst

def gms(r,d):
    np_filter_x = np.array([[[[1/3.0, 0, -1/3.0],[1/3.0, 0, -1/3.0],[1/3.0, 0, -1/3.0]]]])
    np_filter_x = np.repeat(np_filter_x,repeats=r.shape[1], axis=1)
    np_filter_y = np.array([[[[1 / 3.0, 1 / 3.0, 1 / 3.0], [0,0,0],[-1 / 3.0, -1 / 3.0, -1 / 3.0]]]])
    np_filter_y = np.repeat(np_filter_y,repeats=r.shape[1], axis=1)

    hx = torch.from_numpy(np_filter_x).float().cuda()
    hy = torch.from_numpy(np_filter_y).float().cuda()
    rcx = torch.nn.functional.conv2d(r,hx,padding=1)
    rcy = torch.nn.functional.conv2d(r,hy,padding=1)
    dcx = torch.nn.functional.conv2d(d,hx,padding=1)
    dcy = torch.nn.functional.conv2d(d,hy,padding=1)

    mr = (rcx**2 + rcy**2 + 0.00001)**0.5
    md = (dcx**2 + dcy**2 + 0.00001)**0.5
    c = 0.0026
    gms = (2*mr*md+c)/(mr**2+md**2+c)
    return gms

def gmsd(r,d):
    gms_t = gms(r,d)
    gmsm = torch.mean(gms_t)
    gmsd = (torch.mean((gms_t-gmsm)**2)+0.0000001)**0.5
    return gmsd

class MSGMSLoss(torch.nn.modules.loss._Loss):
    def __init__(self, mask=None):
        super(MSGMSLoss, self).__init__()
        self.mask = mask
    def forward(self, r,d):
        gms_t = torch.mean(gms(r, d))
        r1 = torch.nn.functional.avg_pool2d(r, (2, 2))
        d1 = torch.nn.functional.avg_pool2d(d, (2, 2))
        gms_t1 = torch.mean(gms(r1, d1))
        r2 = torch.nn.functional.avg_pool2d(r1, (2, 2))
        d2 = torch.nn.functional.avg_pool2d(d1, (2, 2))
        gms_t2 = torch.mean(gms(r2, d2))
        r3 = torch.nn.functional.avg_pool2d(r2, (2, 2))
        d3 = torch.nn.functional.avg_pool2d(d2, (2, 2))
        gms_t3 = torch.mean(gms(r3, d3))
        gms_fin = (gms_t+gms_t1+gms_t2+gms_t3)/4.0
        return 1.0-gms_fin

class GMSLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super(GMSLoss, self).__init__()
    def forward(self, r,d):
        gms_t = gms(r,d)
        return 1.0-torch.mean(gms_t)


class MaskedMSELoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    def forward(self, input, target, mask):
        mse_map = (input - target)**2
        mse_map = F.mse_loss(input,target,reduce=False)
        mse_map = mse_map * mask
        mse_loss = torch.sum(mse_map) / torch.sum(mask)
        return mse_loss


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out



class PerceptualLossNorm(torch.nn.modules.loss._Loss):
    def __init__(self):
        super(PerceptualLossNorm, self).__init__()
        self.vgg = Vgg16(requires_grad=False).cuda()

    def forward(self, input, target):
        features_y = self.vgg(input)
        features_x = self.vgg(target)
        l1x = features_x.relu2_2
        l1y = features_y.relu2_2
        l2x = features_x.relu1_2
        l2y = features_y.relu1_2
        l3x = features_x.relu3_3
        l3y = features_y.relu3_3
        l4x = features_x.relu4_3
        l4y = features_y.relu4_3
        l1x = l1x / torch.sqrt(torch.sum(l1x**2, dim=1, keepdim=True)+1e-12)
        l1y = l1y / torch.sqrt(torch.sum(l1y**2, dim=1, keepdim=True)+1e-12)
        l2x = l2x / torch.sqrt(torch.sum(l2x**2, dim=1, keepdim=True)+1e-12)
        l2y = l2y / torch.sqrt(torch.sum(l2y**2, dim=1, keepdim=True)+1e-12)
        l3x = l3x / torch.sqrt(torch.sum(l3x**2, dim=1, keepdim=True)+1e-12)
        l3y = l3y / torch.sqrt(torch.sum(l3y**2, dim=1, keepdim=True)+1e-12)
        l4x = l4x / torch.sqrt(torch.sum(l4x**2, dim=1, keepdim=True)+1e-12)
        l4y = l4y / torch.sqrt(torch.sum(l4y**2, dim=1, keepdim=True)+1e-12)

        content_loss1 = F.mse_loss(l1x, l1y)
        content_loss2 = F.mse_loss(l2x, l2y)
        content_loss3 = F.mse_loss(l3x, l3y)
        content_loss4 = F.mse_loss(l4x, l4y)
        content_loss = (content_loss1 + content_loss2 + content_loss3 + content_loss4)/4.0
        return content_loss

class PerceptualLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = Vgg16(requires_grad=False).cuda()

    def forward(self, input, target):
        features_y = self.vgg(input)
        features_x = self.vgg(target)
        content_loss1 = F.mse_loss(features_y.relu2_2, features_x.relu2_2)
        content_loss2 = F.mse_loss(features_y.relu1_2, features_x.relu1_2)
        content_loss3 = F.mse_loss(features_y.relu3_3, features_x.relu3_3)
        content_loss4 = F.mse_loss(features_y.relu4_3, features_x.relu4_3)
        content_loss = content_loss1 + content_loss2 + content_loss3 + content_loss4
        return content_loss

class PerceptualDistance(torch.nn.Module):
    def __init__(self):
        super(PerceptualDistance, self).__init__()
        self.vgg = Vgg16(requires_grad=False).cuda()

    def forward(self, input, target):
        features_y = self.vgg(input)
        features_x = self.vgg(target)
        fx_r12 = features_x.relu1_2
        fx_r22 = features_x.relu2_2
        fx_r33 = features_x.relu3_3
        fx_r43 = features_x.relu4_3
        fy_r12 = features_y.relu1_2
        fy_r22 = features_y.relu2_2
        fy_r33 = features_y.relu3_3
        fy_r43 = features_y.relu4_3
        ufxr12 = fx_r12
        ufxr22 = torch.nn.functional.upsample(fx_r22,scale_factor=2)
        ufxr33 = torch.nn.functional.upsample(fx_r33,scale_factor=4)
        ufxr43 = torch.nn.functional.upsample(fx_r43,scale_factor=8)
        ufyr12 = fy_r12
        ufyr22 = torch.nn.functional.upsample(fy_r22,scale_factor=2)
        ufyr33 = torch.nn.functional.upsample(fy_r33,scale_factor=4)
        ufyr43 = torch.nn.functional.upsample(fy_r43,scale_factor=8)
        d12 = torch.mean((ufxr12 - ufyr12)**2,dim=1,keepdim=True)
        d22 = torch.mean((ufxr22 - ufyr22)**2,dim=1,keepdim=True)
        d33 = torch.mean((ufxr33 - ufyr33)**2,dim=1,keepdim=True)
        d43 = torch.mean((ufxr43 - ufyr43)**2,dim=1,keepdim=True)

        return d12,d22,d33,d43


def deep_SVDD_loss(x, center, R, mu=0.1):
    # x - b,c,h,w
    # center - c
    # mu hyperparam - default 0.1
    # b,1,h,w - sum by channel dim
    distance = torch.sum((x - center) ** 2, dim=1)
    # b,1,h,w
    scores = distance - R ** 2
    # 1
    loss = R ** 2 + (1 / mu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss


def IID_loss_whole(x1_outs, x2_outs_inv, lamb=1.0, EPS=1e-16):
    bn, k, h, w = x1_outs.shape
    x1_outs = x1_outs.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
    x2_outs_inv = x2_outs_inv.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w

    # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1
    p_i_j = torch.conv2d(x1_outs, weight=x2_outs_inv, padding=(0,
                                                               0))

    # do expectation over each shift location in the T_side_dense *
    # T_side_dense box
    T_side_dense = 1

    # T x T x k x k
    p_i_j = p_i_j.permute(2, 3, 0, 1)
    p_i_j = p_i_j / (p_i_j.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True) + EPS)  # norm

    # symmetrise, transpose the k x k part
    p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0

    # T x T x k x k
    p_i_mat = p_i_j.sum(dim=2, keepdim=True).repeat(1, 1, k, 1)
    p_j_mat = p_i_j.sum(dim=3, keepdim=True).repeat(1, 1, 1, k)

    # for log stability; tiny values cancelled out by mult with p_i_j anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_i_j = torch.where(p_i_j < EPS, EPS * torch.ones_like(p_i_j), p_i_j)
    p_i_mat = torch.where(p_i_mat < EPS, EPS * torch.ones_like(p_i_mat), p_i_mat)
    p_j_mat = torch.where(p_j_mat < EPS, EPS * torch.ones_like(p_j_mat), p_j_mat)

    # maximise information
    loss = (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) - lamb * torch.log(p_j_mat))).sum() / (
                T_side_dense * T_side_dense)

    return loss


def perform_affine_tf(data, tf_matrices):
    # expects 4D tensor, we preserve gradients if there are any

    n_i, k, h, w = data.shape
    n_i2, r, c = tf_matrices.shape
    assert (n_i == n_i2)
    assert (r == 2 and c == 3)

    grid = F.affine_grid(tf_matrices, data.shape)  # output should be same size
    data_tf = F.grid_sample(data, grid,
                            padding_mode="zeros")  # this can ONLY do bilinear

    return data_tf


def random_translation_multiple(data, half_side_min, half_side_max):
    n, c, h, w = data.shape

    # pad last 2, i.e. spatial, dimensions, equally in all directions
    data = torch.nn.functional.pad(data,
                                   (half_side_max, half_side_max, half_side_max, half_side_max),
                                   "constant", 0)
    assert (data.shape[2:] == (2 * half_side_max + h, 2 * half_side_max + w))

    # random x, y displacement
    t = np.random.randint(half_side_min, half_side_max + 1, size=(2,))
    polarities = np.random.choice([-1, 1], size=(2,), replace=True)
    t *= polarities

    # -x, -y in orig img frame is now -x+half_side_max, -y+half_side_max in new
    t += half_side_max

    data = data[:, :, t[1]:(t[1] + h), t[0]:(t[0] + w)]
    assert (data.shape[2:] == (h, w))

    return data


def IID_segmentation_loss(x1_outs, x2_outs_inv, lamb=1.0,
                          half_T_side_dense=None,
                          half_T_side_sparse_min=None,
                          half_T_side_sparse_max=None, EPS=1e-16):
    assert (x1_outs.requires_grad)
    # assert (x2_outs.requires_grad)
    # assert (not all_affine2_to_1.requires_grad)

    # assert (x1_outs.shape == x2_outs.shape)
    #
    # # bring x2 back into x1's spatial frame
    # x2_outs_inv = perform_affine_tf(x2_outs, all_affine2_to_1)

    if (half_T_side_sparse_min != 0) or (half_T_side_sparse_max != 0):
        x2_outs_inv = random_translation_multiple(x2_outs_inv,
                                                  half_side_min=half_T_side_sparse_min,
                                                  half_side_max=half_T_side_sparse_max)

    # zero out all irrelevant patches - I DON'T USE SUCH AUGMENTATIONS WHERE MASK IS NECESSARY (shear, rorate by random angle...)
    # bn, k, h, w = x1_outs.shape
    # all_mask_img1 = all_mask_img1.view(bn, 1, h, w)  # mult, already float32
    # x1_outs = x1_outs * all_mask_img1  # broadcasts
    # x2_outs_inv = x2_outs_inv * all_mask_img1

    # sum over everything except classes, by convolving x1_outs with x2_outs_inv
    # which is symmetric, so doesn't matter which one is the filter
    x1_outs = x1_outs.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
    x2_outs_inv = x2_outs_inv.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w

    # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1
    p_i_j = F.conv2d(x1_outs, weight=x2_outs_inv, padding=(half_T_side_dense, half_T_side_dense))
    p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)  # k, k

    # normalise, use sum, not bn * h * w * T_side * T_side, because we use a mask
    # also, some pixels did not have a completely unmasked box neighbourhood,
    # but it's fine - just less samples from that pixel
    current_norm = float(p_i_j.sum())
    p_i_j = p_i_j / current_norm

    # symmetrise
    p_i_j = (p_i_j + p_i_j.t()) / 2.

    # compute marginals
    p_i_mat = p_i_j.sum(dim=1).unsqueeze(1)  # k, 1
    p_j_mat = p_i_j.sum(dim=0).unsqueeze(0)  # 1, k

    # for log stability; tiny values cancelled out by mult with p_i_j anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_i_j = torch.where(p_i_j < EPS, EPS * torch.ones_like(p_i_j), p_i_j)
    p_i_mat = torch.where(p_i_mat < EPS, EPS * torch.ones_like(p_i_mat), p_i_mat)
    p_j_mat = torch.where(p_j_mat < EPS, EPS * torch.ones_like(p_j_mat), p_j_mat)
    T_side_dense = half_T_side_dense * 2 + 1
    # maximise information
    loss = (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) -
                      lamb * torch.log(p_j_mat))).sum() / (
                   T_side_dense * T_side_dense)

    # for analysis only
    # loss_no_lamb = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) -
    #                           torch.log(p_j_mat))).sum()

    return loss


def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=1e-16):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j = torch.where(p_i_j < EPS, EPS * torch.ones_like(p_i_j), p_i_j)
    p_i = torch.where(p_i < EPS, EPS * torch.ones_like(p_i), p_i)
    p_j = torch.where(p_j < EPS, EPS * torch.ones_like(p_j), p_j)
    # p_i_j[(p_i_j < EPS).data] = EPS
    # p_j[(p_j < EPS).data] = EPS
    # p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))

    loss = loss.sum()

    # loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
    #                           - torch.log(p_j) \
    #                           - torch.log(p_i))
    #
    # loss_no_lamb = loss_no_lamb.sum()

    return loss


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j

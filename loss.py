
import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score 

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
from typing import Callable, List

def gather_features(
    mixture_features,
    track_features,
    mixture_features_mlp=None, 
    track_features_mlp=None,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
    mlp_loss=False):
    
    # We gather tensors from all gpus
    if gather_with_grad:
        all_mixture_features = torch.cat(torch.distributed.nn.all_gather(mixture_features), dim=0)
        all_track_features = torch.cat(torch.distributed.nn.all_gather(track_features), dim=0)
        if mlp_loss:
            all_mixture_features_mlp = torch.cat(torch.distributed.nn.all_gather(mixture_features_mlp), dim=0)
            all_track_features_mlp = torch.cat(torch.distributed.nn.all_gather(track_features_mlp), dim=0)
    else:
        gathered_mixture_features = [torch.zeros_like(mixture_features) for _ in range(world_size)]
        gathered_track_features = [torch.zeros_like(track_features) for _ in range(world_size)]
        dist.all_gather(gathered_mixture_features, mixture_features)
        dist.all_gather(gathered_track_features, track_features)
        if mlp_loss:
            gathered_mixture_features_mlp = [torch.zeros_like(mixture_features_mlp) for _ in range(world_size)]
            gathered_track_features_mlp = [torch.zeros_like(track_features_mlp) for _ in range(world_size)]
            dist.all_gather(gathered_mixture_features_mlp, mixture_features_mlp)
            dist.all_gather(gathered_track_features_mlp, track_features_mlp)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_mixture_features[rank] = mixture_features
            gathered_track_features[rank] = track_features
            if mlp_loss:
                gathered_mixture_features_mlp[rank] = mixture_features_mlp
                gathered_track_features_mlp[rank] = track_features_mlp

        all_mixture_features = torch.cat(gathered_mixture_features, dim=0)
        all_track_features = torch.cat(gathered_track_features, dim=0)
        if mlp_loss:
            all_mixture_features_mlp = torch.cat(gathered_mixture_features, dim=0)
            all_track_features_mlp = torch.cat(gathered_track_features, dim=0)
    if mlp_loss:
        return all_mixture_features, all_track_features, all_mixture_features_mlp, all_track_features_mlp
    else:
        return all_mixture_features, all_track_features
    
class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        mlp_loss=False,
        weight_loss_kappa=0,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.mlp_loss = mlp_loss
        self.weighted_loss = bool(weight_loss_kappa!=0)
        self.weight_loss_kappa = weight_loss_kappa
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
    
    def forward(self, mixture_features, track_features, logit_scale_m, logit_scale_t=None, mixture_features_mlp=None, track_features_mlp=None):
        device = mixture_features.device
        if self.mlp_loss:
            if self.world_size > 1:
                all_mixture_features, all_track_features, all_mixture_features_mlp, all_track_features_mlp = gather_features(
                    mixture_features=mixture_features,track_features=track_features,
                    mixture_features_mlp=mixture_features_mlp,track_features_mlp=track_features_mlp,
                    local_loss=self.local_loss,gather_with_grad=self.gather_with_grad,
                    rank=self.rank,world_size=self.world_size,use_horovod=self.use_horovod,
                    mlp_loss=self.mlp_loss
                )
                if self.local_loss:
                    m_logits_per_mixture = logit_scale_m * mixture_features @ all_track_features_mlp.T
                    m_logits_per_track = logit_scale_m * track_features_mlp @ all_mixture_features.T
                    t_logits_per_mixture = logit_scale_t * mixture_features_mlp @ all_track_features.T
                    t_logits_per_track = logit_scale_t * track_features @ all_mixture_features_mlp.T
                else:
                    m_logits_per_mixture = logit_scale_m * all_mixture_features @ all_track_features_mlp.T
                    m_logits_per_track = m_logits_per_mixture.T
                    t_logits_per_mixture = logit_scale_t * all_mixture_features_mlp @ all_track_features.T
                    t_logits_per_track = t_logits_per_mixture.T
            else:
                m_logits_per_mixture = logit_scale_m * mixture_features @ track_features_mlp.T
                m_logits_per_track = logit_scale_m * track_features_mlp @ mixture_features.T
                t_logits_per_mixture = logit_scale_t * mixture_features_mlp @ track_features.T
                t_logits_per_track = logit_scale_t * track_features @ mixture_features_mlp.T

            # calculated ground-truth and cache if enabled
            num_logits = m_logits_per_mixture.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]

            if not self.weighted_loss:
                total_loss = (
                    F.cross_entropy(m_logits_per_mixture, labels) +
                    F.cross_entropy(m_logits_per_track, labels) + 
                    F.cross_entropy(t_logits_per_mixture, labels) +
                    F.cross_entropy(t_logits_per_track, labels) 
                    ) / 4
            else:
                mixture_weight = (mixture_features@mixture_features.T).detach()
                mixture_weight = (torch.exp(torch.sum(mixture_weight, axis=1)/(self.weight_loss_kappa*len(mixture_weight)))).detach()
                track_weight = (track_features@track_features.T).detach()
                track_weight = (torch.exp(torch.sum(track_weight, axis=1)/(self.weight_loss_kappa*len(track_features)))).detach()
                total_loss = (
                    F.cross_entropy(m_logits_per_mixture, labels, weight=mixture_weight) +
                    F.cross_entropy(m_logits_per_track, labels, weight=mixture_weight) + 
                    F.cross_entropy(t_logits_per_mixture, labels, weight=track_weight) +
                    F.cross_entropy(t_logits_per_track, labels, weight=track_weight) 
                    ) / 4
        else:
            if self.world_size > 1:
                all_mixture_features, all_track_features = gather_features(
                    mixture_features=mixture_features,track_features=track_features,
                    local_loss=self.local_loss,gather_with_grad=self.gather_with_grad,
                    rank=self.rank,world_size=self.world_size,use_horovod=self.use_horovod,
                    mlp_loss=self.mlp_loss
                )

                if self.local_loss:
                    logits_per_mixture = logit_scale_m * mixture_features @ all_track_features.T
                    logits_per_track = logit_scale_m * track_features @ all_mixture_features.T
                else:
                    logits_per_mixture = logit_scale_m * all_mixture_features @ all_track_features.T
                    logits_per_track = logits_per_mixture.T
            else:
                logits_per_mixture = logit_scale_m * mixture_features @ track_features.T
                logits_per_track = logit_scale_m * track_features @ mixture_features.T

            # calculated ground-truth and cache if enabled
            num_logits = logits_per_mixture.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]
            if not self.weighted_loss:
                total_loss = (
                    F.cross_entropy(logits_per_mixture, labels) +
                    F.cross_entropy(logits_per_track, labels)
                    ) / 2
            else:
                mixture_weight = (all_mixture_features@all_mixture_features.T).detach()
                mixture_weight = (torch.exp(torch.sum(mixture_weight, axis=1)/(self.weight_loss_kappa*len(all_mixture_features)))).detach()
                track_weight = (all_track_features@all_track_features.T).detach()
                track_weight = (torch.exp(torch.sum(track_weight, axis=1)/(self.weight_loss_kappa*len(all_track_features)))).detach()
                total_loss = (
                    F.cross_entropy(logits_per_mixture, labels, weight=track_weight) +
                    F.cross_entropy(logits_per_track, labels, weight=mixture_weight)
                    ) / 2
        return total_loss

class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        mlp_loss=False,
        weight_loss_kappa=0,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.mlp_loss = mlp_loss
        self.weighted_loss = bool(weight_loss_kappa!=0)
        self.weight_loss_kappa = weight_loss_kappa
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
    
    def forward(self, mixture_features, track_features, logit_scale_m, logit_scale_t=None, mixture_features_mlp=None, track_features_mlp=None):
        device = mixture_features.device
        if self.mlp_loss:
            if self.world_size > 1:
                all_mixture_features, all_track_features, all_mixture_features_mlp, all_track_features_mlp = gather_features(
                    mixture_features=mixture_features,track_features=track_features,
                    mixture_features_mlp=mixture_features_mlp,track_features_mlp=track_features_mlp,
                    local_loss=self.local_loss,gather_with_grad=self.gather_with_grad,
                    rank=self.rank,world_size=self.world_size,use_horovod=self.use_horovod,
                    mlp_loss=self.mlp_loss
                )
                if self.local_loss:
                    m_logits_per_mixture = logit_scale_m * mixture_features @ all_track_features_mlp.T
                    m_logits_per_track = logit_scale_m * track_features_mlp @ all_mixture_features.T
                    t_logits_per_mixture = logit_scale_t * mixture_features_mlp @ all_track_features.T
                    t_logits_per_track = logit_scale_t * track_features @ all_mixture_features_mlp.T
                else:
                    m_logits_per_mixture = logit_scale_m * all_mixture_features @ all_track_features_mlp.T
                    m_logits_per_track = m_logits_per_mixture.T
                    t_logits_per_mixture = logit_scale_t * all_mixture_features_mlp @ all_track_features.T
                    t_logits_per_track = t_logits_per_mixture.T
            else:
                m_logits_per_mixture = logit_scale_m * mixture_features @ track_features_mlp.T
                m_logits_per_track = logit_scale_m * track_features_mlp @ mixture_features.T
                t_logits_per_mixture = logit_scale_t * mixture_features_mlp @ track_features.T
                t_logits_per_track = logit_scale_t * track_features @ mixture_features_mlp.T

            # calculated ground-truth and cache if enabled
            num_logits = m_logits_per_mixture.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]

            if not self.weighted_loss:
                total_loss = (
                    F.cross_entropy(m_logits_per_mixture, labels) +
                    F.cross_entropy(m_logits_per_track, labels) + 
                    F.cross_entropy(t_logits_per_mixture, labels) +
                    F.cross_entropy(t_logits_per_track, labels) 
                    ) / 4
            else:
                mixture_weight = (mixture_features@mixture_features.T).detach()
                mixture_weight = (torch.exp(torch.sum(mixture_weight, axis=1)/(self.weight_loss_kappa*len(mixture_weight)))).detach()
                track_weight = (track_features@track_features.T).detach()
                track_weight = (torch.exp(torch.sum(track_weight, axis=1)/(self.weight_loss_kappa*len(track_features)))).detach()
                total_loss = (
                    F.cross_entropy(m_logits_per_mixture, labels, weight=mixture_weight) +
                    F.cross_entropy(m_logits_per_track, labels, weight=mixture_weight) + 
                    F.cross_entropy(t_logits_per_mixture, labels, weight=track_weight) +
                    F.cross_entropy(t_logits_per_track, labels, weight=track_weight) 
                    ) / 4
        else:
            if self.world_size > 1:
                all_mixture_features, all_track_features = gather_features(
                    mixture_features=mixture_features,track_features=track_features,
                    local_loss=self.local_loss,gather_with_grad=self.gather_with_grad,
                    rank=self.rank,world_size=self.world_size,use_horovod=self.use_horovod,
                    mlp_loss=self.mlp_loss
                )

                if self.local_loss:
                    logits_per_mixture = logit_scale_m * mixture_features @ all_track_features.T
                    logits_per_track = logit_scale_m * track_features @ all_mixture_features.T
                else:
                    logits_per_mixture = logit_scale_m * all_mixture_features @ all_track_features.T
                    logits_per_track = logits_per_mixture.T
            else:
                logits_per_mixture = logit_scale_m * mixture_features @ track_features.T
                logits_per_track = logit_scale_m * track_features @ mixture_features.T

            # calculated ground-truth and cache if enabled
            num_logits = logits_per_mixture.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]
            if not self.weighted_loss:
                total_loss = (
                    F.cross_entropy(logits_per_mixture, labels) +
                    F.cross_entropy(logits_per_track, labels)
                    ) / 2
            else:
                mixture_weight = (all_mixture_features@all_mixture_features.T).detach()
                mixture_weight = (torch.exp(torch.sum(mixture_weight, axis=1)/(self.weight_loss_kappa*len(all_mixture_features)))).detach()
                track_weight = (all_track_features@all_track_features.T).detach()
                track_weight = (torch.exp(torch.sum(track_weight, axis=1)/(self.weight_loss_kappa*len(all_track_features)))).detach()
                total_loss = (
                    F.cross_entropy(logits_per_mixture, labels, weight=track_weight) +
                    F.cross_entropy(logits_per_track, labels, weight=mixture_weight)
                    ) / 2
        return total_loss

def lp_gather_features(
        pred,
        target,
        world_size=1,
        use_horovod=False):
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        with torch.no_grad():
            all_preds = hvd.allgather(pred)
            all_targets = hvd.allgath(target)
    else:
        gathered_preds = [torch.zeros_like(pred) for _ in range(world_size)]
        gathered_targets = [torch.zeros_like(target) for _ in range(world_size)]

        dist.all_gather(gathered_preds, pred)
        dist.all_gather(gathered_targets, target)
        all_preds = torch.cat(gathered_preds, dim=0)
        all_targets = torch.cat(gathered_targets, dim=0)

    return all_preds, all_targets

def get_map(pred, target):
    pred = torch.sigmoid(pred).numpy()
    target = target.numpy()
    return np.mean(average_precision_score(target, pred, average=None))

def get_acc(pred, target):
    pred = torch.argmax(pred,1).numpy()
    target = torch.argmax(target,1).numpy()
    return accuracy_score(target, pred)

def get_mauc(pred, target):
    pred = torch.sigmoid(pred).numpy()
    target = target.numpy()
    return np.mean(roc_auc_score(target, pred, average=None))

class LPMetrics(object):
    def __init__(self, metric_names = ['map','acc','mauc']):
        self.metrics = []
        for name in metric_names:
            self.metrics.append(self.get_metric(name))
        self.metric_names = metric_names

    def get_metric(self,name):
        if name == 'map':
            return get_map
        elif name == 'acc':
            return get_acc
        elif name == 'mauc':
            return get_mauc
        else:
            raise ValueError(f'the metric should be at least one of [map, acc, mauc]')

    def evaluate_mertics(self, pred, target):
        metric_dict = {}
        for i in range(len(self.metric_names)):
            metric_dict[self.metric_names[i]] = self.metrics[i](pred, target)
        return metric_dict

def calc_celoss(pred, target):
    target = torch.argmax(target, 1).long()
    return nn.CrossEntropyLoss()(pred, target)

class LPLoss(nn.Module):

    def __init__(self, loss_name):
        super().__init__()
        if loss_name == 'bce':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif loss_name == 'ce':
            self.loss_func = calc_celoss
        elif loss_name == 'mse':
            self.loss_func = nn.MSELoss()
        else:
            raise ValueError(f'the loss func should be at least one of [bce, ce, mse]')

    def forward(self, pred, target):
        loss = self.loss_func(pred, target)
        return loss
        
from distributed import GatherLayer
class NT_Xent(nn.Module): # Normalizee Temperature-scaled Cross-Entropy loss
    def __init__(self, batch_size, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        #self.temperature = temperature
        self.world_size = world_size
        self.temperature = 0.1
        self.beta = 1
        self.tau_plus = 0.01
        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j, logit_scale_m, logit_scale_t=None, mixture_features_mlp=None, track_features_mlp=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size # 256 

        if self.world_size > 1:
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
        z = torch.cat((z_i, z_j), dim=0) # [256, 128]

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
    
class SimCLRLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        mlp_loss=False,
        weight_loss_kappa=0,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.mlp_loss = mlp_loss
        self.weighted_loss = bool(weight_loss_kappa!=0)
        self.weight_loss_kappa = weight_loss_kappa
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        
    def forward(self, mixture_features, track_features, logit_scale_m, logit_scale_t=None, mixture_features_mlp=None, track_features_mlp=None):
        device = mixture_features.device
        self.criterian = NT_Xent(mixture_features.shape[0], self.world_size).to(device)
        total_loss = self.criterian(mixture_features, track_features, logit_scale_m)
        return total_loss


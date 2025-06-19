import json
import logging
import math
import os
import time
from contextlib import suppress
import random 
import julius 
import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from loss import SimCLRLoss
from distributed import is_master
from fx_chain.constants import ALL_PROCESSORS
from train_utils import *
import torch.distributed as dist

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model

def _core_norm(
    args, 
    batch, 
    device, 
    num_of_mixed_stems, 
    fx_aug_chain, 
    fx_order = None, 
    eval_mode = False
):
    # get stems and availability mask from dataloader 
    stems = batch['audio'] 
    stems = stems.squeeze(0) 

    if not eval_mode:
        # pick n+1 number of tracks for mixing 
        total_num_of_stems = stems.shape[1]
        if args.distributed and dist.is_initialized():
            if dist.get_rank() == 0: # to make sure the entire batch is same instrumentation 
                selected_stems = torch.randperm(total_num_of_stems, device=device)[:num_of_mixed_stems+1]
            else:
                selected_stems = torch.zeros(num_of_mixed_stems+1, dtype=torch.long, device=device)
            dist.broadcast(selected_stems, src=0)
        else:
            selected_stems = torch.randperm(total_num_of_stems, device=device)[:num_of_mixed_stems+1]

        selected_stems = selected_stems.cpu()
        stems = stems[:, selected_stems]
        
        # Only process the selected stems
        number_of_stems = stems.shape[1]
        stems = stems.to(device)  
    else:
        # pick n+1 number of tracks for mixing 
        total_num_of_stems = stems.shape[1] 
        selected_stems = torch.randperm(total_num_of_stems)[:num_of_mixed_stems+1]
        stems = stems[:, selected_stems]
        number_of_stems = stems.shape[1]
        stems = stems.to(device)  # [batch_size, num_of_mixed_stems+1, stereo, seq_length]
    
    # Split stems into A, B parts first
    split_length = stems.shape[-1] // 2
    stems_a = stems[..., :split_length] # [batch_size, num_of_mixed_stems+1, stereo, seq_length//2]
    stems_b = stems[..., split_length:] # [batch_size, num_of_mixed_stems+1, stereo, seq_length//2]

    batch_size = stems_a.shape[0]
    # randomly pick different query stem for each batch
    query_stem_ids = torch.stack(
        [torch.randperm(number_of_stems)[:1] for _ in range(batch_size)]
    )  # [batch_size, 1]

    # apply fx to all stems
    stems_a = stems_a.view(-1, stems_a.shape[-2], stems_a.shape[-1])
    stems_b = stems_b.view(-1, stems_b.shape[-2], stems_b.shape[-1])
    
    
    if fx_order is None:
        processors_order = ALL_PROCESSORS.copy()
        random.shuffle(processors_order)
        processed_stems_a, nn_param, activate = fx_aug_chain(stems_a, None, None, processors_order)
        processed_stems_b, _, _ = fx_aug_chain(stems_b, nn_param, activate, processors_order)
    else:
        must_use = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        activate = torch.bernoulli(must_use.expand(stems_a.shape[0], -1))
        processed_stems_a, nn_param, activate = fx_aug_chain(stems_a, None, activate, fx_order)
        processed_stems_b, _, _ = fx_aug_chain(stems_b, nn_param, activate, fx_order)

    # loudness normalize of each stem before summing as the mixture 
    # LOUDNESS_RANGE = (-18, -14)
    random_loudness_values = -18 + 4 * torch.rand(processed_stems_a.shape[0]) 
    processed_stems_a = loudness_normalize(processed_stems_a, random_loudness_values)
    processed_stems_b = loudness_normalize(processed_stems_b, random_loudness_values)


    processed_stems_a = processed_stems_a.view(stems_a.shape[0]//number_of_stems, 
                                            number_of_stems, 
                                            processed_stems_a.shape[-2], 
                                            processed_stems_a.shape[-1])

    processed_stems_b = processed_stems_b.view(stems_b.shape[0]//number_of_stems, 
                                            number_of_stems, 
                                            processed_stems_b.shape[-2], 
                                            processed_stems_b.shape[-1])
    

    batch_indices = torch.arange(processed_stems_a.shape[0])
    stem_a = processed_stems_a[batch_indices, query_stem_ids.squeeze(1)]
    query_stem = stem_a.clone()
    query_stem = loudness_normalize(query_stem, None)

    # Sum to get mixtures
    mixture_a = processed_stems_a.sum(dim=1)
    mixture_b = processed_stems_b.sum(dim=1)

    # final loudness normalize  
    mixture_a = loudness_normalize(mixture_a, None)
    mixture_b = loudness_normalize(mixture_b, None)


    return mixture_a, mixture_b, stem_a, julius.resample_frac(query_stem, int(44100), int(48000))

def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer = None, fx_aug_chain = None):
    num_stems_to_mix = args.num_mixes
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    model.train()

    # Reset indices at the start of each epoch
    data["train"].dataloader.dataset.reset_indices()
    
    # loss 
    loss = SimCLRLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
        mlp_loss=args.clap_mlploss,
        weight_loss_kappa=args.kappa,
    )
    
    dataloader, sampler = data["train"].dataloader, data["train"].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches # how many batches in dataloader
    samples_per_epoch = dataloader.num_samples   #  how many data in dataloader 
    
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # interface
    loss_mixtures_m = {n: AverageMeter() for n in range(num_stems_to_mix)}  
    loss_stems_m = {n: AverageMeter() for n in range(num_stems_to_mix)}  

    total_steps = num_batches_per_epoch * num_stems_to_mix 
    global_step = epoch * total_steps  
    
    for i, batch in enumerate(dataloader):
        base_step = num_batches_per_epoch * epoch + i
        current_iter = i + 1  
        for n in range(num_stems_to_mix): # random mix 1, 2, 3, 4 tracks
            step = base_step * num_stems_to_mix + n
            global_step += 1  
            current_track_count = n + 1 
            if isinstance(scheduler, dict):
                for s in scheduler.values():
                    s(step)
            else:
                scheduler(step)
            
            global_sc, local_sc = get_loss_weights(step, args.loss_sch)
            mixture_a, mixture_b, stem_a, query_stem = _core_norm(args, batch, device, n, fx_aug_chain, eval_mode=False) 
            
            data_time_m.update(time.time() - end)
            if isinstance(optimizer, dict):
                for o_ in optimizer.values():
                    o_.zero_grad()
            else:
                optimizer.zero_grad()

            
            with autocast():
                (
                    mixture_a_features, # global
                    mixture_b_features, # global 
                    stem_a_features, # local
                    extracted_stem_features, # local
                    mixture_a_features_mlp,
                    mixture_b_features_mlp,
                    logit_scale_m,
                    logit_scale_t,
                ) = model(mixture_a, mixture_b, stem_a, query_stem, device=device)
                
                loss_mixtures = loss(
                    mixture_a_features,
                    mixture_b_features,
                    logit_scale_m
                ) 
                loss_stems = loss(
                    mixture_features=stem_a_features,
                    track_features=extracted_stem_features,
                    logit_scale_m=logit_scale_m
                ) 
                total_loss = global_sc * loss_mixtures + local_sc * loss_stems
                

            if isinstance(optimizer, dict):
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    for o_ in optimizer.values():
                        if args.horovod:
                            o_.synchronize()
                            scaler.unscale_(o_)
                            with o_.skip_synchronize():
                                scaler.step(o_)
                        else:
                            scaler.step(o_)
                    scaler.update()
                else:
                    total_loss.backward()
                    for o_ in optimizer.values():
                        o_.step()
            else:
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    if args.horovod:
                        optimizer.synchronize()
                        scaler.unscale_(optimizer)
                        with optimizer.skip_synchronize():
                            scaler.step(optimizer)
                    else:
                        scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

            with torch.no_grad():
                unwrap_model(model).logit_scale_m.clamp_(0, math.log(100))
                if args.clap_mlploss:
                    unwrap_model(model).logit_scale_t.clamp_(0, math.log(100))

            loss_mixtures_m[n].update(loss_mixtures.item(), mixture_a.shape[0])
            loss_stems_m[n].update(loss_stems.item(), mixture_a.shape[0])
            batch_time_m.update(time.time() - end)
            end = time.time()
            if is_master(args) and ((i) % 1 == 0 or current_iter == num_batches_per_epoch):
                log_data = {
                    "epoch": epoch,
                    "global_step": global_step,
                    f"loss/mixtures_w_{current_track_count}_stems": loss_mixtures_m[n].val,
                    f"loss/mixtures_w_{current_track_count}_stems_avg": loss_mixtures_m[n].avg,
                    "training/learning_rate": optimizer.param_groups[0]["lr"] if not isinstance(optimizer, dict) 
                                        else [o_.param_groups[0]["lr"] for o_ in optimizer.values()],
                }
                log_data[f"loss/extracted_stems_w_{current_track_count}_mixtures"] = loss_stems_m[n].val
                log_data[f"loss/extracted_stems_w_{current_track_count}_mixtures_avg"] = loss_stems_m[n].avg
                

                if args.clap_mlploss:
                    log_data.update({
                        "training/logit_scale_track": logit_scale_t.item(),
                    })

                logging_msg = (
                    f"Epoch: {epoch}/{args.epochs} | "
                    f"Batch: {current_iter}/{num_batches_per_epoch} | "
                    f"Global Step: {global_step}/{args.epochs * total_steps} | "
                    f"Tracks: {current_track_count}/4 | "
                    f"Mixture Loss: {loss_mixtures_m[n].val:.4f} (avg: {loss_mixtures_m[n].avg:.4f}) | "
                    f"Stems Loss: {loss_stems_m[n].val:.4f} (avg: {loss_stems_m[n].avg:.4f}) | "
                    f"LR: {log_data['training/learning_rate']:.6f} | "
                    f"Batch Time: {batch_time_m.avg:.3f}s"
                )

                logging.info(logging_msg)
                
                if args.wandb:
                    wandb.log(log_data)
            
            if n == 3:  
                batch_time_m.reset()
                data_time_m.reset()

# > ======== Retrieval  
def evaluate(model, data, epoch, args, fx_aug_chain, tb_writer=None):
    num_random_stems_to_mix = args.num_mixes
    metrics = {}
    if not args.parallel_eval:
        if not is_master(args):
            return metrics
    device = torch.device(args.device)
    model.eval()
    if is_master(args):
        print('Evaluating...')
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress

    if "val" in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)): # how many epochs to evaluate
        dataloader = data["val"].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples
        total_batches = len(dataloader)
        eval_info = {}
        eval_info = {str(n): {
            "cumulative_loss": 0.0,
            "num_samples": 0,
            "mixture_a_features": [],
            "mixture_b_features": [],
            "stem_a_features": [],
            "extracted_stem_features": [],
        } for n in range(1, num_random_stems_to_mix+1)}

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                for number_of_stems in range(num_random_stems_to_mix):
                    mixture_a, mixture_b, stem_a, query_stem = _core_norm(
                        args, batch, device, number_of_stems, fx_aug_chain, eval_mode=True
                    ) 
                    with autocast():
                        (
                            mixture_a_features, 
                            mixture_b_features,
                            stem_a_features, 
                            extracted_stem_features, 
                            mixture_a_features_mlp,
                            mixture_b_features_mlp,
                            logit_scale_m,
                            logit_scale_t,
                        ) = model(mixture_a, mixture_b, stem_a, query_stem, device=device)
                        if is_master(args):
                            num_samples += mixture_a_features.shape[0]
                            if is_master(args):
                                eval_info[str(number_of_stems+1)]["mixture_a_features"].append(mixture_a_features.cpu())
                                eval_info[str(number_of_stems+1)]["mixture_b_features"].append(mixture_b_features.cpu())
                                eval_info[str(number_of_stems+1)]["stem_a_features"].append(stem_a_features.cpu())
                                eval_info[str(number_of_stems+1)]["extracted_stem_features"].append(extracted_stem_features.cpu())      
                                eval_info[str(number_of_stems+1)]["num_samples"] += mixture_a_features.shape[0]
                    if is_master(args) and (i % 1) == 0:  
                        logging.info(
                            f"Eval Epoch: {epoch} [{number_of_stems+1} tracks] | "
                            f"Batch: {i+1}/{total_batches} | "
                            f"Overall Progress: {(i * num_random_stems_to_mix + number_of_stems + 1)}/{(total_batches * num_random_stems_to_mix)}"
                        )
            #### 
            if is_master(args):
                val_metrics_per_dataset = {}
                for n in eval_info.keys():
                    metrics_single_dataset = get_metrics(
                        mixture_a_features=torch.cat(eval_info[n]["mixture_a_features"]),
                        mixture_b_features=torch.cat(eval_info[n]["mixture_b_features"]),
                        logit_scale_m=logit_scale_m.cpu(),
                        mlp_loss=args.clap_mlploss
                    )
                    if len(eval_info[n]["stem_a_features"]) > 0 and len(eval_info[n]["extracted_stem_features"]) > 0:
                        metrics_single_dataset_2 = get_metrics_stem(
                            stem_features=torch.cat(eval_info[n]["stem_a_features"]),
                            extracted_stem_features=torch.cat(eval_info[n]["extracted_stem_features"]),
                            logit_scale_m=logit_scale_m.cpu(),
                            mlp_loss=args.clap_mlploss
                        )
                        metrics_single_dataset.update(metrics_single_dataset_2)
                    val_metrics_per_dataset[n] = {
                        n + "/" + k: v for k, v in metrics_single_dataset.items()
                    }
                    metrics.update(val_metrics_per_dataset[n])
                    if "epoch" not in metrics.keys():
                        metrics.update({"epoch": epoch})
    
    if is_master(args):
        if not metrics:
            return metrics

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\n".join(
                [
                    "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in m.items()])
                    for m in val_metrics_per_dataset.values()
                ]
            )
        )

        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/{name}", val, epoch)

            with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

        if args.wandb:
            assert wandb is not None, "Please install wandb."
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, "epoch": epoch})

        return metrics
    else:
        return metrics
    
def get_metrics(
        mixture_a_features,
        mixture_b_features,
        logit_scale_m,
        mixture_a_features_mlp=None,
        mixture_b_features_mlp=None,
        logit_scale_t=None,
        mlp_loss=False):
    
    metrics = {}
    logits_per_mixture_a = (logit_scale_m * mixture_a_features @ mixture_b_features.t()).detach().cpu()
    logits_per_mixture_b = logits_per_mixture_a.t().detach().cpu()

    labels = torch.arange(mixture_a_features.shape[0]).long()
    # Change the loss from two terms into four terms with 2x2 combined CE loss
    total_loss = (F.cross_entropy(logits_per_mixture_a, labels) + F.cross_entropy(logits_per_mixture_b, labels)) / 2

    metrics[f"cumulative_loss"] = total_loss.item()
    metrics[f"num_samples"] = mixture_a_features.shape[0]

    logits = {"mixture_to_mixture": logits_per_mixture_a}

    ground_truth = torch.arange(len(mixture_b_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]  # (yusong) this line is slow because it uses single thread
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
        # map@10
        metrics[f"{name}_mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

    return metrics

def get_metrics_stem(
        stem_features,
        extracted_stem_features,
        logit_scale_m,
        mixture_a_features_mlp=None,
        mixture_b_features_mlp=None,
        logit_scale_t=None,
        mlp_loss=False):
    metrics = {}
    logits_per_stem_a = (logit_scale_m * stem_features @ extracted_stem_features.t()).detach().cpu()
    logits_per_stem_b = logits_per_stem_a.t().detach().cpu()

    labels = torch.arange(stem_features.shape[0]).long()
    # Change the loss from two terms into four terms with 2x2 combined CE loss
    total_loss = (F.cross_entropy(logits_per_stem_a, labels) + F.cross_entropy(logits_per_stem_b, labels)) / 2

    metrics[f"stem_cumulative_loss"] = total_loss.item()
    metrics[f"stem_num_samples"] = stem_features.shape[0]

    logits = {"stem_to_extracted_stem": logits_per_stem_a}

    ground_truth = torch.arange(len(extracted_stem_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]  # (yusong) this line is slow because it uses single thread
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
        # map@10
        metrics[f"{name}_mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

    return metrics

# # > ======= Evaluate on single fx 
def evaluate_single_fx(model, data, epoch, args, fx_aug_chain, tb_writer=None):
    metrics = {}
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    model.eval()
    
    dataloader, sampler = data["val"].dataloader, data["val"].sampler
    total_batches = len(dataloader)
    total_effects = len(ALL_PROCESSORS)
    eval_info = {fx: {
        "cumulative_loss": 0.0,
        "num_samples": 0,
        "mixture_a_features": [],
        "mixture_b_features": []
    } for fx in ALL_PROCESSORS}
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            for single_fx in ALL_PROCESSORS:
                fx_order = [single_fx]
                mixture_a, mixture_b, stem_a, query_stem = _core_norm(args, batch, device, 1, fx_aug_chain, fx_order, eval_mode=True)
                with autocast():
                    (
                        mixture_a_features, # global
                        mixture_b_features, # global 
                        stem_a_features, # local
                        extracted_stem_features, # local
                        mixture_a_features_mlp,
                        mixture_b_features_mlp,
                        logit_scale_m,
                        logit_scale_t,
                    ) = model(mixture_a, mixture_b, stem_a, query_stem, device=device)
                    if is_master(args):
                        eval_info[single_fx]["mixture_a_features"].append(mixture_a_features.cpu())
                        eval_info[single_fx]["mixture_b_features"].append(mixture_b_features.cpu())
                        eval_info[single_fx]["num_samples"] += mixture_a_features.shape[0]
                
                if is_master(args) and (i % 1) == 0:
                    current_effect_idx = ALL_PROCESSORS.index(single_fx)
                    overall_progress = (i * total_effects + current_effect_idx + 1) / (total_batches * total_effects)
                    logging.info(
                        f"Eval Epoch: {epoch} [Audio effect: {single_fx}] | "
                        f"Batch: {i+1}/{total_batches} | "
                        f"Overall Progress: {(i * total_effects + current_effect_idx + 1)}/{(total_batches * total_effects)}"
                    )
    
    cumulative_losses = {}
    if is_master(args):
        for fx in ALL_PROCESSORS:
            if eval_info[fx]["num_samples"] > 0:
                metrics_single_fx = get_metrics(
                    mixture_a_features=torch.cat(eval_info[fx]["mixture_a_features"]),
                    mixture_b_features=torch.cat(eval_info[fx]["mixture_b_features"]),
                    logit_scale_m=logit_scale_m.cpu(),
                    mlp_loss=args.clap_mlploss
                )
                metrics.update({f"{fx}/{k}": v for k, v in metrics_single_fx.items()})
                cumulative_losses[fx] = metrics_single_fx["cumulative_loss"]
        metrics["epoch"] = epoch

        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/{name}", val, epoch)

        if args.wandb:
            assert wandb is not None, "Please install wandb."
            for name, val in metrics.items():   
                wandb.log({f"val/{name}": val, "epoch": epoch})
            
            wandb.log({
                "Cumulative_Losses": {fx: loss for fx, loss in cumulative_losses.items()},
                "epoch": epoch
            })
        return metrics, cumulative_losses
    else:
        return metrics, cumulative_losses
    


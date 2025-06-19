import logging
import os
import random
from datetime import datetime
import copy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try: # i won't use this
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try: # i won't use this
    import horovod.torch as hvd
except ImportError:
    hvd = None

from params import parse_args
from logger import setup_logging
from distributed import is_master, init_distributed_device, world_info_from_env
from scheduler import cosine_lr
from models.factory import create_model
from data import get_data
from models.utils import get_optimizer
from fx_aug import Random_FX_Chain
from train import evaluate, evaluate_single_fx, train_one_epoch

import warnings
warnings.filterwarnings("ignore", message="Grid size.*will likely result in GPU under-utilization due to low occupancy.")
warnings.simplefilter(action='ignore', category=FutureWarning)

def is_pretrained_params(n):
    return (
        n.startswith("transformer")
        or n in ["positional_embedding", "text_projection"]
        or n.startswith("token_embedding")
        or n.startswith("ln_final")
        or n.startswith("logit_scale_t")
    )

def maintain_ckpts(args, startidx, all_idx_len):
    for i in reversed(range(startidx, all_idx_len)):
        if os.path.exists(os.path.join(args.checkpoint_path, f"epoch_top_{i}.pt")):
            os.rename(
                os.path.join(args.checkpoint_path, f"epoch_top_{i}.pt"),
                os.path.join(args.checkpoint_path, f"epoch_top_{i+1}.pt"),
            )
    if os.path.exists(
        os.path.join(args.checkpoint_path, f"epoch_top_{all_idx_len}.pt")
    ):
        os.remove(os.path.join(args.checkpoint_path, f"epoch_top_{all_idx_len}.pt"))
    return

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def copy_codebase(args):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(
        current_code_path, new_code_path, ignore=ignore_patterns("log", "logs", "wandb")
    )
    print("Done copying code.")
    return 1

def main():
    args = parse_args()
    
    # > ============================================================================= < 
    # set up seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # # get the name of the experiments
    if args.name is None:
        args.name = "-".join(
            [
                datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
            ]
        )
    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1
    
    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)
    # fully initialize distributed device environment
    device = init_distributed_device(args)
    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
    if is_master(args):
        args.tensorboard_path = (
            os.path.join(args.logs, args.name, "tensorboard")
            if args.tensorboard
            else ""
        )
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ""
        args.checkpoint_path = ""

    if args.copy_codebase:
        copy_codebase(args)
    
    assert args.precision in ["amp", "fp16", "fp32"]
    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use fp32 mixed-precision instead of FP16 and AMP in this model. "
            "They will cause NaN loss and NaN gradients. "
            "FP16 and AMP support needs further verification and tuning, especially for train."
        )

    if args.horovod:
        logging.info(
            f"Running in horovod mode with multiple processes / nodes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    elif args.distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")
    # > ============================================================================= < 
    
    # create model & random audio effect chain 
    model, model_cfg = create_model(args) ######## 
    model = model.to(device)
    print('model: ', model)
    
    # # will do the fx probility scheduling 
    FX_AUG_CHAIN = Random_FX_Chain(model_cfg['mixture_cfg']['sample_rate'], device)
    
    # > No need to modify, just some initial setup <
    # > ============================================================================= < 
    if args.horovod:
        with torch.no_grad():
            for param in model.parameters():
                param.set_(param.contiguous())
    
    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
    
    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args["static_graph"] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True, **ddp_args
        )
    # > ============================================================================= < 

    # # get training data 
    data = get_data(
        args, 
        sample_rate = model_cfg['mixture_cfg']['sample_rate'], 
        win_len = model_cfg['mixture_cfg']['clip_samples'],
        hop_len = model_cfg['mixture_cfg']['clip_samples'] // 10,
    )
    assert len(data), "At least one train or eval dataset must be specified."
    
    # ctx = 0
    # print('> len(dataset): ', data['train'].dataloader.num_samples)
    # train_data = data["train"]
    # for batch in train_data.dataloader:
    #     audio = batch['audio']
    #     print('> audio: ', audio.shape)
    #     break
    
    # > No need to modify, just some initial setup <
    # > ============================================================================= < 
    exclude = (
        lambda n, p: p.ndim < 2
        or "bn" in n
        or "ln" in n
        or "bias" in n
        or "logit_scale" in n
    )
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())

    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad
    ]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    # set wd-related params to 0 if use adam optimizer
    if args.optimizer == "adam":
        args.wd = 0
        args.wd_pretrained = 0
        args.wd_new = 0
    
    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        total_steps = data["train"].dataloader.num_batches * args.num_mixes * args.epochs 

        optimizer = get_optimizer(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            momentum=args.momentum,
            optimizer_name=args.optimizer,
        )

        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters()
            )
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    scaler = GradScaler() if args.precision == "amp" else None

    # # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            if "epoch" in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
                    sd = {k[len("module.") :]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and "scaler" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler"])
                logging.info(
                    f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})"
                )
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(
                    f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})"
                )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = False
    
    # # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    # > Comment out wandb for now < 
    if args.wandb and is_master(args):
        assert wandb is not None, "Please install wandb."
        logging.debug("Starting wandb.")
        args.train_sz = data["train"].dataloader.num_samples 
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            entity=args.wandb_entity, 
            project=args.wandb_project,
            notes=args.wandb_notes,
            name=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log="all")
        wandb.save(params_file)
        logging.debug("Finished loading wandb.")
    # # > ============================================================================= < 
    
    
    if "train" not in data:
        evaluate(model, data, start_epoch, args, FX_AUG_CHAIN, writer)
        return
    elif start_epoch == 0 and "val" in data and not args.no_eval:
        evaluate(model, data, 0,  args,  FX_AUG_CHAIN, writer)
        evaluate_single_fx(model, data, 0, args, FX_AUG_CHAIN, writer)
    
    
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f"Start epoch {epoch}")

        # Initialize fx_aug_chain
        train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer, FX_AUG_CHAIN)
        completed_epoch = epoch + 1
        torch.cuda.empty_cache()
        # # Evaluate the model 
        if not args.no_eval and (completed_epoch % args.val_frequency == 0):
            metrics = evaluate(model, data, completed_epoch, args, FX_AUG_CHAIN, writer)
            _, cumulative_losses = evaluate_single_fx(model, data, completed_epoch, args, FX_AUG_CHAIN, writer)

            # fx probability scheduling
            if not cumulative_losses: 
                pass 
            else:   
                max_loss = max(cumulative_losses.values())
                min_loss = min(cumulative_losses.values())
                new_fx_prob = {}
                for fx in cumulative_losses: 
                    highest_prob = 1.0 
                    lowest_prob = 0.1 
                    cur_val = cumulative_losses[fx] 
                    new_fx_prob[fx] = (cur_val - min_loss) / (max_loss - min_loss) * (highest_prob - lowest_prob) + lowest_prob
                FX_AUG_CHAIN.update_fx_prob(new_fx_prob)
                if args.wandb:
                    assert wandb is not None, "Please install wandb."
                    wandb.log({'New_FX_Prob': new_fx_prob, "epoch": completed_epoch})
            
        # Saving checkpoints.
        if args.save_logs:
            if args.split_opt:
                opt_dict = {
                    k + "_" + "optimizer": v.state_dict() for k, v in optimizer.items()
                }
            else:
                opt_dict = {"optimizer": optimizer.state_dict()}
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
            }
            checkpoint_dict.update(opt_dict)
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )

    if args.wandb and is_master(args):
        wandb.finish()
    
    
if __name__ == "__main__":
    main()
    print('> ============ TEST FINISHED ============ < ')
# Copyright (c) Facebook, Inc. and its affiliates.

import psutil
import argparse
import itertools
import datetime
import math
import numpy as np
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import datasets
from iterators import EpochBatchIterator
from models import CombinedSpatiotemporalModel, JumpCNFSpatiotemporalModel, SelfAttentiveCNFSpatiotemporalModel, JumpGMMSpatiotemporalModel
from models.spatial import GaussianMixtureSpatialModel, IndependentCNF, JumpCNF, SelfAttentiveCNF
from models.spatial.cnf import TimeVariableCNF
from models.temporal import HomogeneousPoissonPointProcess, HawkesPointProcess, SelfCorrectingPointProcess, NeuralPointProcess
from models.temporal.neural import ACTFNS as TPP_ACTFNS
from models.temporal.neural import TimeVariableODE
import toy_datasets
import utils
from viz_dataset import load_data, MAPS


torch.backends.cudnn.benchmark = True


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))


def cleanup():
    dist.destroy_process_group()


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return learning_rate * decayed


def learning_rate_schedule(global_step, warmup_steps, base_learning_rate, train_steps):
    warmup_steps = int(round(warmup_steps))
    scaled_lr = base_learning_rate
    if warmup_steps:
        learning_rate = global_step / warmup_steps * scaled_lr
    else:
        learning_rate = scaled_lr

    if global_step < warmup_steps:
        learning_rate = learning_rate
    else:
        learning_rate = cosine_decay(scaled_lr, global_step - warmup_steps, train_steps - warmup_steps)
    return learning_rate


def set_learning_rate(optimizer, lr):
    for i, group in enumerate(optimizer.param_groups):
        group['lr'] = lr


def cast(tensor, device):
    return tensor.float().to(device)


def get_t0_t1(data):
    if data == "citibike":
        return torch.tensor([0.0]), torch.tensor([24.0])
    elif data == "covid_nj_cases":
        return torch.tensor([0.0]), torch.tensor([7.0])
    elif data == "earthquakes_jp":
        return torch.tensor([0.0]), torch.tensor([30.0])
    elif data == "pinwheel":
        return torch.tensor([0.0]), torch.tensor([toy_datasets.END_TIME])
    elif data == "gmm":
        return torch.tensor([0.0]), torch.tensor([toy_datasets.END_TIME])
    elif data == "fmri":
        return torch.tensor([0.0]), torch.tensor([10.0])
    else:
        raise ValueError(f"Unknown dataset {data}")


def get_dim(data):
    if data == "gmm":
        return 1
    elif data == "fmri":
        return 3
    else:
        return 2


def validate(model, test_loader, t0, t1, device):

    model.eval()

    space_loglik_meter = utils.AverageMeter()
    time_loglik_meter = utils.AverageMeter()

    with torch.no_grad():
        for batch in test_loader:
            event_times, spatial_locations, input_mask = map(lambda x: cast(x, device), batch)
            num_events = input_mask.sum()
            space_loglik, time_loglik = model(event_times, spatial_locations, input_mask, t0, t1)
            space_loglik = space_loglik.sum() / num_events
            time_loglik = time_loglik.sum() / num_events

            space_loglik_meter.update(space_loglik.item(), num_events)
            time_loglik_meter.update(time_loglik.item(), num_events)

    model.train()

    return space_loglik_meter.avg, time_loglik_meter.avg


def main(rank, world_size, args, savepath):
    setup(rank, world_size, args.port)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    logger = utils.get_logger(os.path.join(savepath, "logs"))

    try:
        _main(rank, world_size, args, savepath, logger)
    except:
        import traceback
        logger.error(traceback.format_exc())
        raise

    cleanup()


def to_numpy(x):
    if torch.is_tensor(x):
        return x.cpu().detach().numpy()
    return [to_numpy(x_i) for x_i in x]


def _main(rank, world_size, args, savepath, logger):

    if rank == 0:
        logger.info(args)
        logger.info(f"Saving to {savepath}")
        tb_writer = SummaryWriter(os.path.join(savepath, "tb_logdir"))

    device = torch.device(f'cuda:{rank:d}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        if device.type == 'cuda':
            logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
        else:
            logger.info('WARNING: Using device {}'.format(device))

    t0, t1 = map(lambda x: cast(x, device), get_t0_t1(args.data))

    train_set = load_data(args.data, split="train")
    val_set = load_data(args.data, split="val")
    test_set = load_data(args.data, split="test")

    train_epoch_iter = EpochBatchIterator(
        dataset=train_set,
        collate_fn=datasets.spatiotemporal_events_collate_fn,
        batch_sampler=train_set.batch_by_size(args.max_events),
        seed=args.seed + rank,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.test_bsz,
        shuffle=False,
        collate_fn=datasets.spatiotemporal_events_collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_bsz,
        shuffle=False,
        collate_fn=datasets.spatiotemporal_events_collate_fn,
    )

    if rank == 0:
        logger.info(f"{len(train_set)} training examples, {len(val_set)} val examples, {len(test_set)} test examples")

    x_dim = get_dim(args.data)

    if args.model == "jumpcnf" and args.tpp == "neural":
        model = JumpCNFSpatiotemporalModel(dim=x_dim,
                                           hidden_dims=list(map(int, args.hdims.split("-"))),
                                           tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
                                           actfn=args.actfn,
                                           tpp_cond=args.tpp_cond,
                                           tpp_style=args.tpp_style,
                                           tpp_actfn=args.tpp_actfn,
                                           share_hidden=args.share_hidden,
                                           solve_reverse=args.solve_reverse,
                                           tol=args.tol,
                                           otreg_strength=args.otreg_strength,
                                           tpp_otreg_strength=args.tpp_otreg_strength,
                                           layer_type=args.layer_type,
                                           ).to(device)
    elif args.model == "attncnf" and args.tpp == "neural":
        model = SelfAttentiveCNFSpatiotemporalModel(dim=x_dim,
                                                    hidden_dims=list(map(int, args.hdims.split("-"))),
                                                    tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
                                                    actfn=args.actfn,
                                                    tpp_cond=args.tpp_cond,
                                                    tpp_style=args.tpp_style,
                                                    tpp_actfn=args.tpp_actfn,
                                                    share_hidden=args.share_hidden,
                                                    solve_reverse=args.solve_reverse,
                                                    l2_attn=args.l2_attn,
                                                    tol=args.tol,
                                                    otreg_strength=args.otreg_strength,
                                                    tpp_otreg_strength=args.tpp_otreg_strength,
                                                    layer_type=args.layer_type,
                                                    lowvar_trace=not args.naive_hutch,
                                                    ).to(device)
    elif args.model == "cond_gmm" and args.tpp == "neural":
        model = JumpGMMSpatiotemporalModel(dim=x_dim,
                                           hidden_dims=list(map(int, args.hdims.split("-"))),
                                           tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
                                           actfn=args.actfn,
                                           tpp_cond=args.tpp_cond,
                                           tpp_style=args.tpp_style,
                                           tpp_actfn=args.tpp_actfn,
                                           share_hidden=args.share_hidden,
                                           tol=args.tol,
                                           tpp_otreg_strength=args.tpp_otreg_strength,
                                           ).to(device)
    else:
        # Mix and match between spatial and temporal models.
        if args.tpp == "poisson":
            tpp_model = HomogeneousPoissonPointProcess()
        elif args.tpp == "hawkes":
            tpp_model = HawkesPointProcess()
        elif args.tpp == "correcting":
            tpp_model = SelfCorrectingPointProcess()
        elif args.tpp == "neural":
            tpp_hidden_dims = list(map(int, args.tpp_hdims.split("-")))
            tpp_model = NeuralPointProcess(
                cond_dim=x_dim, hidden_dims=tpp_hidden_dims, cond=args.tpp_cond, style=args.tpp_style, actfn=args.tpp_actfn,
                otreg_strength=args.tpp_otreg_strength, tol=args.tol)
        else:
            raise ValueError(f"Invalid tpp model {args.tpp}")

        if args.model == "gmm":
            model = CombinedSpatiotemporalModel(GaussianMixtureSpatialModel(), tpp_model).to(device)
        elif args.model == "cnf":
            model = CombinedSpatiotemporalModel(
                IndependentCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                               layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength,
                               squash_time=True),
                tpp_model).to(device)
        elif args.model == "tvcnf":
            model = CombinedSpatiotemporalModel(
                IndependentCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                               layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength),
                tpp_model).to(device)
        elif args.model == "jumpcnf":
            model = CombinedSpatiotemporalModel(
                JumpCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                        layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength),
                tpp_model).to(device)
        elif args.model == "attncnf":
            model = CombinedSpatiotemporalModel(
                SelfAttentiveCNF(dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                                 layer_type=args.layer_type, actfn=args.actfn, l2_attn=args.l2_attn, tol=args.tol, otreg_strength=args.otreg_strength),
                tpp_model).to(device)
        else:
            raise ValueError(f"Invalid model {args.model}")

    params = []
    attn_params = []
    for name, p in model.named_parameters():
        if "self_attns" in name:
            attn_params.append(p)
        else:
            params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": params},
        {"params": attn_params}
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    if rank == 0:
        ema = utils.ExponentialMovingAverage(model)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        logger.info(model)

    begin_itr = 0
    checkpt_path = os.path.join(savepath, "model.pth")
    if os.path.exists(checkpt_path):
        # Restart from checkpoint if run is a restart.
        if rank == 0:
            logger.info(f"Resuming checkpoint from {checkpt_path}")
        checkpt = torch.load(checkpt_path, "cpu")
        model.module.load_state_dict(checkpt["state_dict"])
        optimizer.load_state_dict(checkpt["optim_state_dict"])
        begin_itr = checkpt["itr"] + 1

    elif args.resume:
        # Check the resume flag if run is new.
        if rank == 0:
            logger.info(f"Resuming model from {args.resume}")
        checkpt = torch.load(args.resume, "cpu")
        model.module.load_state_dict(checkpt["state_dict"])
        optimizer.load_state_dict(checkpt["optim_state_dict"])
        begin_itr = checkpt["itr"] + 1

    space_loglik_meter = utils.RunningAverageMeter(0.98)
    time_loglik_meter = utils.RunningAverageMeter(0.98)
    gradnorm_meter = utils.RunningAverageMeter(0.98)

    model.train()
    start_time = time.time()
    iteration_counter = itertools.count(begin_itr)
    begin_epoch = begin_itr // len(train_epoch_iter)
    for epoch in range(begin_epoch, math.ceil(args.num_iterations / len(train_epoch_iter))):
        batch_iter = train_epoch_iter.next_epoch_itr(shuffle=True)
        for batch in batch_iter:
            itr = next(iteration_counter)

            optimizer.zero_grad()

            event_times, spatial_locations, input_mask = map(lambda x: cast(x, device), batch)
            N, T = input_mask.shape
            num_events = input_mask.sum()

            if num_events == 0:
                raise RuntimeError("Got batch with no observations.")

            space_loglik, time_loglik = model(event_times, spatial_locations, input_mask, t0, t1)

            space_loglik = space_loglik.sum() / num_events
            time_loglik = time_loglik.sum() / num_events
            loglik = time_loglik + space_loglik

            space_loglik_meter.update(space_loglik.item())
            time_loglik_meter.update(time_loglik.item())

            loss = loglik.mul(-1.0).mean()
            loss.backward()

            # Set learning rate
            total_itrs = math.ceil(args.num_iterations / len(train_epoch_iter)) * len(train_epoch_iter)
            lr = learning_rate_schedule(itr, args.warmup_itrs, args.lr, total_itrs)
            set_learning_rate(optimizer, lr)

            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=args.gradclip).item()
            gradnorm_meter.update(grad_norm)

            optimizer.step()

            if rank == 0:
                if itr > 0.8 * args.num_iterations:
                    ema.apply()
                else:
                    ema.apply(decay=0.0)

            if rank == 0:
                tb_writer.add_scalar("train/lr", lr, itr)
                tb_writer.add_scalar("train/temporal_loss", time_loglik.item(), itr)
                tb_writer.add_scalar("train/spatial_loss", space_loglik.item(), itr)
                tb_writer.add_scalar("train/grad_norm", grad_norm, itr)

            if itr % args.logfreq == 0:
                elapsed_time = time.time() - start_time

                # Average NFE across devices.
                nfe = 0
                for m in model.modules():
                    if isinstance(m, TimeVariableCNF) or isinstance(m, TimeVariableODE):
                        nfe += m.nfe
                nfe = torch.tensor(nfe).to(device)
                dist.all_reduce(nfe, op=dist.ReduceOp.SUM)
                nfe = nfe // world_size

                # Sum memory usage across devices.
                mem = torch.tensor(memory_usage_psutil()).float().to(device)
                dist.all_reduce(mem, op=dist.ReduceOp.SUM)

                if rank == 0:
                    logger.info(
                        f"Iter {itr} | Epoch {epoch} | LR {lr:.5f} | Time {elapsed_time:.1f}"
                        f" | Temporal {time_loglik_meter.val:.4f}({time_loglik_meter.avg:.4f})"
                        f" | Spatial {space_loglik_meter.val:.4f}({space_loglik_meter.avg:.4f})"
                        f" | GradNorm {gradnorm_meter.val:.2f}({gradnorm_meter.avg:.2f})"
                        f" | NFE {nfe.item()}"
                        f" | Mem {mem.item():.2f} MB")

                    tb_writer.add_scalar("train/nfe", nfe, itr)
                    tb_writer.add_scalar("train/time_per_itr", elapsed_time / args.logfreq, itr)

                start_time = time.time()

            if rank == 0 and itr % args.testfreq == 0:
                # ema.swap()
                val_space_loglik, val_time_loglik = validate(model, val_loader, t0, t1, device)
                test_space_loglik, test_time_loglik = validate(model, test_loader, t0, t1, device)
                # ema.swap()
                logger.info(
                    f"[Test] Iter {itr} | Val Temporal {val_time_loglik:.4f} | Val Spatial {val_space_loglik:.4f}"
                    f" | Test Temporal {test_time_loglik:.4f} | Test Spatial {test_space_loglik:.4f}")

                tb_writer.add_scalar("val/temporal_loss", val_time_loglik, itr)
                tb_writer.add_scalar("val/spatial_loss", val_space_loglik, itr)

                tb_writer.add_scalar("test/temporal_loss", test_time_loglik, itr)
                tb_writer.add_scalar("test/spatial_loss", test_space_loglik, itr)

                torch.save({
                    "itr": itr,
                    "state_dict": model.module.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "ema_parmas": ema.ema_params,
                }, checkpt_path)

                start_time = time.time()

    if rank == 0:
        tb_writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, choices=MAPS.keys(), default="earthquakes_jp")

    parser.add_argument("--model", type=str, choices=["cond_gmm", "gmm", "cnf", "tvcnf", "jumpcnf", "attncnf"], default="gmm")
    parser.add_argument("--tpp", type=str, choices=["poisson", "hawkes", "correcting", "neural"], default="poisson")
    parser.add_argument("--actfn", type=str, default="swish")
    parser.add_argument("--tpp_actfn", type=str, choices=TPP_ACTFNS.keys(), default="softplus")
    parser.add_argument("--hdims", type=str, default="64-64-64")
    parser.add_argument("--layer_type", type=str, choices=["concat", "concatsquash"], default="concat")
    parser.add_argument("--tpp_hdims", type=str, default="32-32")
    parser.add_argument("--tpp_nocond", action="store_false", dest='tpp_cond')
    parser.add_argument("--tpp_style", type=str, choices=["split", "simple", "gru"], default="gru")
    parser.add_argument("--no_share_hidden", action="store_false", dest='share_hidden')
    parser.add_argument("--solve_reverse", action="store_true")
    parser.add_argument("--l2_attn", action="store_true")
    parser.add_argument("--naive_hutch", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--otreg_strength", type=float, default=1e-4)
    parser.add_argument("--tpp_otreg_strength", type=float, default=1e-4)

    parser.add_argument("--warmup_itrs", type=int, default=0)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gradclip", type=float, default=0)
    parser.add_argument("--max_events", type=int, default=4000)
    parser.add_argument("--test_bsz", type=int, default=32)

    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logfreq", type=int, default=10)
    parser.add_argument("--testfreq", type=int, default=100)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.port is None:
        args.port = int(np.random.randint(10000, 20000))

    if args.experiment_id is None:
        args.experiment_id = time.strftime("%Y%m%d_%H%M%S")

    experiment_name = f"{args.model}"
    if args.model in ["cnf", "tvcnf", "jumpcnf", "attncnf"]:
        experiment_name += f"{args.hdims}"
        experiment_name += f"_{args.layer_type}"
        experiment_name += f"_{args.actfn}"
        experiment_name += f"_ot{args.otreg_strength}"

    if args.model == "attncnf":
        if args.l2_attn:
            experiment_name += "_l2attn"
        if args.naive_hutch:
            experiment_name += "_naivehutch"

    if args.model in ["cnf", "tvcnf", "jumpcnf", "attncnf"]:
        experiment_name += f"_tol{args.tol}"

    experiment_name += f"_{args.tpp}"
    if args.tpp in ["neural"]:
        experiment_name += f"{args.tpp_hdims}"
        experiment_name += f"{args.tpp_style}"
        experiment_name += f"_{args.tpp_actfn}"
        experiment_name += f"_ot{args.tpp_otreg_strength}"
        if args.tpp_cond:
            experiment_name += "_cond"
    if args.share_hidden and args.model in ["jumpcnf", "attncnf"] and args.tpp == "neural":
        experiment_name += "_sharehidden"
    if args.solve_reverse and args.model == "jumpcnf" and args.tpp == "neural":
        experiment_name += "_rev"
    experiment_name += f"_lr{args.lr}"
    experiment_name += f"_gc{args.gradclip}"
    experiment_name += f"_bsz{args.max_events}x{args.ngpus}_wd{args.weight_decay}_s{args.seed}"
    experiment_name += f"_{args.experiment_id}"
    savepath = os.path.join(args.experiment_dir, experiment_name)

    # Top-level logger for logging exceptions into the log file.
    utils.makedirs(savepath)
    logger = utils.get_logger(os.path.join(savepath, "logs"))

    if args.gradclip == 0:
        args.gradclip = 1e10

    try:
        mp.set_start_method("forkserver")
        mp.spawn(main,
                 args=(args.ngpus, args, savepath),
                 nprocs=args.ngpus,
                 join=True)
    except Exception:
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

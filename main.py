import sys
import os
import argparse
from shutil import rmtree

from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from models.model import EventDetection
from dataset import VideoDataSet, Collator
from loss_function import bmn_loss_func, attention_mining_loss_func, get_mask
from post_processing import PostProcessor, getDatasetDict
from utils import ProposalGenerator

from eval_anet import evaluate_proposals as anet_evaluate_prop
from eval_thumos import evaluate_proposals as thumos_evaluate_prop

from eval_det_anet import evaluate_detections as anet_evaluate_det
from eval_det_thumos import evaluate_detections as thumos_evaluate_det

from config.defaults import get_cfg

sys.dont_write_bytecode = True


class Solver:
    def __init__(self, cfg, local_rank=0):
        self.cfg = cfg
        self.local_rank = local_rank

        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        torch.manual_seed(0)
        torch.cuda.set_device(local_rank)

        self.model = EventDetection(cfg).cuda()
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank
        )

        if cfg.MODE not in ['train', 'training']:  # TODO: add condition for resume feature.
            checkpoint = torch.load(cfg.TEST.CHECKPOINT_PATH)
            print('Loaded model at epoch %d.' % checkpoint['epoch'])
            self.model.load_state_dict(checkpoint['state_dict'])

        exp_id = max([0] + [int(run.split('_')[-1]) for run in os.listdir(self.cfg.TRAIN.LOG_DIR)]) + 1
        self.log_dir = os.path.join(self.cfg.TRAIN.LOG_DIR, 'run_' + str(exp_id))

        if self.local_rank == 0:
            if not os.path.isdir(os.path.join(self.log_dir, 'checkpoints')):
                os.makedirs(os.path.join(self.log_dir, 'checkpoints'))

        if cfg.MODE in ['train', 'training']:
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
            #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
            self.train_collator = Collator(cfg, 'train')
        self.test_collator = Collator(cfg, 'test')

        self.temporal_dim = cfg.DATA.TEMPORAL_DIM
        self.max_duration = cfg.DATA.MAX_DURATION

        self.evaluate_func = None
        if cfg.DATASET == 'anet':
            if cfg.EVAL_TYPE == 'proposal':
                self.evaluate_func = anet_evaluate_prop
            elif cfg.EVAL_TYPE == 'detection':
                self.evaluate_func = anet_evaluate_det
        elif cfg.DATASET == 'thumos':
            if cfg.EVAL_TYPE == 'proposal':
                self.evaluate_func = thumos_evaluate_prop
            elif cfg.EVAL_TYPE == 'detection':
                self.evaluate_func = thumos_evaluate_det
        if self.evaluate_func is None:
            print('Evaluation function [{}] of dataset [{}] is not implemented'.format(cfg.EVAL_TYPE, cfg.DATASET))

    def train_epoch(self, data_loader, bm_mask, epoch, writer):
        cfg = self.cfg
        self.model.train()
        self.optimizer.zero_grad()
        loss_names = [
            'BMN Loss', 'TemLoss', 'PemLoss Regression', 'PemLoss Classification',
            'Attention Loss', 'AttnPemLoss', 'AttnTemLoss'
        ]
        epoch_losses = [0] * 7
        period_losses = [0] * 7
        last_period_size = len(data_loader) % cfg.TRAIN.STEP_PERIOD
        last_period_start = cfg.TRAIN.STEP_PERIOD * (len(data_loader) // cfg.TRAIN.STEP_PERIOD)

        for n_iter, (featmaps, agent_boxes, label_confidence, label_start, label_end) in enumerate(tqdm(data_loader)):
            featmaps = [x.cuda() for x in featmaps] if cfg.USE_ENV else None

            label_start = label_start.cuda()
            label_end = label_end.cuda()
            label_confidence = label_confidence.cuda()

            env_outputs, act_outputs = self.model(featmaps, agent_boxes)
            #act_outputs = self.model(featmaps, agent_boxes)

            confidence_map, start, end = act_outputs
            losses = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask)
            period_size = cfg.TRAIN.STEP_PERIOD if n_iter < last_period_start else last_period_size

            confidence_map, start, end = env_outputs
            mining_losses = attention_mining_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask)
            period_size = cfg.TRAIN.STEP_PERIOD if n_iter < last_period_start else last_period_size

            total_loss = (losses[0] + mining_losses[0]) / period_size
            total_loss.backward()

            losses = losses + mining_losses
            losses = [l / cfg.TRAIN.STEP_PERIOD for l in losses]
            period_losses = [l + pl for l, pl in zip(losses, period_losses)]

            if (n_iter + 1) % cfg.TRAIN.STEP_PERIOD != 0 and n_iter != (len(data_loader) - 1):
                continue

            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_losses = [el + pl for el, pl in zip(epoch_losses, period_losses)]

            write_step = epoch * len(data_loader) + n_iter
            for i, loss_name in enumerate(loss_names):
                loss = period_losses[i]
                dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
                if self.local_rank == 0:
                    writer.add_scalar(loss_name, loss / len(self.cfg.GPU_IDS), write_step)
            period_losses = [0] * 7

        print(
            "BMN training loss(epoch %d): tem_loss: %.03f, pem reg_loss: %.03f, pem cls_loss: %.03f, total_loss: %.03f" % (
                epoch, epoch_losses[1] / (n_iter + 1),
                epoch_losses[2] / (n_iter + 1),
                epoch_losses[3] / (n_iter + 1),
                epoch_losses[0] / (n_iter + 1)))

    def train(self, n_epochs):
        if self.local_rank == 0:
            writer = SummaryWriter(self.log_dir)
            checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        else:
            writer = None

        train_data = VideoDataSet(self.cfg, split=self.cfg.TRAIN.SPLIT)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=self.cfg.TRAIN.BATCH_SIZE, num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True, drop_last=True, collate_fn=self.train_collator)

        val_data = VideoDataSet(self.cfg, split=self.cfg.VAL.SPLIT)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data,
            shuffle=False)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            sampler=val_sampler,
            batch_size=self.cfg.VAL.BATCH_SIZE, num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True, drop_last=False, collate_fn=self.test_collator)

        bm_mask = get_mask(self.temporal_dim, self.max_duration).cuda()
        scores = []
        for epoch in range(n_epochs):
            #print('Current LR: {}'.format(self.scheduler.get_last_lr()[0]))
            self.train_epoch(train_loader, bm_mask, epoch, writer)
            #self.scheduler.step()
            score = self.evaluate(val_loader, self.cfg.VAL.SPLIT)

            dist.barrier()
            if score is None:  # Not master process, keep training
                continue

            state = {
                'epoch': epoch + 1,
                'score': score,
                'state_dict': self.model.state_dict()
            }
            if len(scores) == 0 or score > max(scores):
                torch.save(state, os.path.join(checkpoint_dir, "best_{}.pth".format(self.cfg.EVAL_SCORE)))
            torch.save(state, os.path.join(checkpoint_dir, "model_{}.pth".format(epoch + 1)))

            writer.add_scalar(self.cfg.EVAL_SCORE, score, epoch)
            scores.append(score)

    def evaluate(self, data_loader=None, split=None):
        self.inference(data_loader, split, self.cfg.VAL.BATCH_SIZE)

        dist.barrier()
        # Only evaluate on a single (master) gpu:
        if self.local_rank == 0:
            score = self.evaluate_func(self.cfg)  # AUC if dataset=anet, AR@100 if dataset=thumos
            return score

    def inference(self, data_loader=None, split=None, batch_size=None):
        if self.local_rank == 0:
            if os.path.isdir('results/outputs/'):
                rmtree('results/outputs/')
            os.makedirs('results/outputs/')
        dist.barrier()

        annotations = getDatasetDict(self.cfg.DATA.ANNOTATION_FILE, split) if self.cfg.DATASET == 'thumos' else None
        self.prop_gen = ProposalGenerator(self.temporal_dim, self.max_duration, annotations)
        self.post_processing = PostProcessor(self.cfg, split)
        if data_loader is None:
            data_loader = torch.utils.data.DataLoader(
                VideoDataSet(self.cfg, split=split),
                batch_size=batch_size, shuffle=False, num_workers=self.cfg.NUM_WORKERS,
                pin_memory=True, drop_last=False, collate_fn=self.test_collator)

        col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_score", "score"]
        self.model.eval()
        with torch.no_grad():
            for video_names, featmaps, agent_boxes in tqdm(data_loader):
                featmaps = [x.cuda() for x in featmaps] if self.cfg.USE_ENV else None

                _, act_outputs = self.model(featmaps, agent_boxes)
                confidence_map, start_map, end_map = act_outputs
                confidence_map = confidence_map.cpu().numpy()
                start_map = start_map.cpu().numpy()
                end_map = end_map.cpu().numpy()

                batch_props = self.prop_gen(start_map, end_map, confidence_map, video_names)
                for video_name, new_props in zip(video_names, batch_props):
                    new_df = pd.DataFrame(new_props, columns=col_name)
                    new_df.to_feather("./results/outputs/" + video_name + ".feather")
        self.post_processing()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_rank',
        type=int,
        help='For GPU distributed run. (Automatically assigned by torch.distributed.launch)'
    )
    parser.add_argument(
        '--cfg-file',
        default=None,
        type=str,
        help='Path to YAML config file.'
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER
    )
    return parser.parse_args()


def main(args):
    cfg = get_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    solver = Solver(cfg, args.local_rank)

    if cfg.MODE in ["train", "training"]:
        solver.train(cfg.TRAIN.NUM_EPOCHS)
    elif cfg.MODE in ['validate', 'validation']:
        solver.evaluate(split=cfg.VAL.SPLIT)
    elif cfg.MODE in ['test', 'testing']:
        solver.inference(split=cfg.TEST.SPLIT, batch_size=cfg.TEST.BATCH_SIZE)


if __name__ == '__main__':
    args = get_args()
    main(args)

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.node_dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds, test_single_slice

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/node_data_thyroid', help='Name of Experiment')
parser.add_argument('--exp', type=str, # _GrabCut_BoxLoss_logits plLoss
                    default='node_data_thyroid/Fully_Supervised_GPU1', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=50,
                    help='labeled data')
parser.add_argument('--gpu', type=str, default='1',
                    help='gpu id')
parser.add_argument('--grabCut', type=bool, default=False,
                    help='if use GrabCut for recist label')
parser.add_argument('--poly', type=bool, default=False,
                    help='if use GrabCut for recist label')
parser.add_argument('--circle', type=bool, default=False,
                    help='if use GrabCut for recist label')
parser.add_argument('--rect', type=bool, default=False,
                    help='if use GrabCut for recist label')
parser.add_argument('--ellipse', type=bool, default=False,
                    help='if use GrabCut for recist label')
parser.add_argument('--BackEllipse', type=bool, default=False,
                    help='if use GrabCut for recist label')
parser.add_argument('--SAM', type=bool, default=False,
                    help='sam')
parser.add_argument('--ProjectionLoss', type=bool, default=False,
                    help='if use GrabCut for recist label')
args = parser.parse_args()


def projection_losses(gt_mask, pred_mask):
    def dice_loss(input, target):
        assert input.shape[-2:] == target.shape[-2:]
        input = input.view(input.size(0), -1).float()
        target = target.view(target.size(0), -1).float()

        d = (
            2 * torch.sum(input * target, dim=1)
        ) / (
            torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4
        )

        return 1 - d
    # 将类别维度上的概率值转换为二值掩码
    pred_mask_binary = torch.argmax(pred_mask, dim=1)
    # 计算每个像素在 x、y 维度上的最大值，然后对二值掩码进行投影
    pred_x_proj = pred_mask_binary.max(dim=1)[0]
    pred_y_proj = pred_mask_binary.max(dim=2)[0]
    # print("*******************************")
    # print(gt_mask.shape, pred_x_proj.shape)
    # print("*******************************")
    gt_x_proj = gt_mask.max(dim=1)[0]
    gt_y_proj = gt_mask.max(dim=2)[0]
    # 计算投影损失
    loss_proj = dice_loss(pred_x_proj, gt_x_proj) + dice_loss(pred_y_proj, gt_y_proj)
    loss_proj = loss_proj.mean()
    return loss_proj

def projection_losses_logits(gt_mask, pred_mask):

    def dice_loss(score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    pred_x_proj = pred_mask.max(dim=2)[0]
    pred_y_proj = pred_mask.max(dim=3)[0]
    gt_x_proj = gt_mask.max(dim=1)[0]
    gt_y_proj = gt_mask.max(dim=2)[0]
    loss_proj = dice_loss(pred_x_proj, gt_x_proj.unsqueeze(1)) + dice_loss(pred_y_proj, gt_y_proj.unsqueeze(1))
    loss_proj = loss_proj.mean()
    return loss_proj


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    # labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    # db_train = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, transform=transforms.Compose([
    #     RandomGenerator(args.patch_size)
    # ]))
    db_train = BaseDataSets(base_dir=args.root_path,split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            if args.ProjectionLoss:
                axis_label_batch = sampled_batch['axis_label']
                axis_label_batch = axis_label_batch.cuda()
                loss_box = projection_losses_logits(axis_label_batch, outputs_soft)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            if args.ProjectionLoss:
                loss = 0.5 * (loss_dice + loss_ce) + 0.5 * loss_box
            else:
                loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    # print(sampled_batch["idx"])
                    metric_i = test_single_slice(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

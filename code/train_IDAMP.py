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
import torch.nn.functional as Func
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.node_dataset_Ours import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds, test_single_CoTraining, test_single_slice

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/node_data_thyroid', help='Name of Experiment')
parser.add_argument('--exp', type=str, 
                    default='node_data_thyroid/Weak_Supervised_Ours', help='experiment_name')
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
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu id')
parser.add_argument('--conservative', type=bool, default=True,
                    help='if use GrabCut for recist label')
parser.add_argument('--radical', type=bool, default=True,
                    help='if use GrabCut for recist label')
parser.add_argument('--ProjectionLoss', type=bool, default=True,
                    help='if use GrabCut for recist label')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


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

def diceCoeff(pred, gt, eps=1e-5):
    N = gt.size(0)
    pred_flat = pred.view(N, -1).float()
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N

class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeff(y_pr, y_gt)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model2 = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    db_train = BaseDataSets(base_dir=args.root_path, conservative=args.conservative,radical=args.radical,split="train", transform=transforms.Compose([
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
    model2.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    w_con, w_rad = torch.FloatTensor([1, 3]).cuda(), torch.FloatTensor([3, 1]).cuda()
    ce_loss1 = CrossEntropyLoss(weight=w_con)
    ce_loss2 = CrossEntropyLoss(weight=w_rad)
    ce_loss = CrossEntropyLoss(reduction='none')
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch_radical, label_batch_conservative = sampled_batch['image'], sampled_batch['label_radical'], sampled_batch['label_conservative']
            volume_batch, label_batch_radical, label_batch_conservative = volume_batch.cuda(), label_batch_radical.cuda(), label_batch_conservative.cuda()
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            
            beta = random.random() + 1e-10
            pseudo_output = torch.argmax(
                 (beta * outputs_soft.detach() + (1.0-beta) * outputs_soft2.detach()), dim=1, keepdim=False)
            pseudo_output_soft = (beta * outputs_soft.detach() + (1.0-beta) * outputs_soft2.detach())
            pseudo_supervision1 = ce_loss(outputs, pseudo_output)
            pseudo_supervision2 = ce_loss(outputs2, pseudo_output)
            if args.ProjectionLoss:
                axis_label_batch = sampled_batch['axis_label']
                axis_label_batch = axis_label_batch.cuda()
                loss_box_1 = projection_losses_logits(axis_label_batch, outputs_soft)
                loss_box_2 = projection_losses_logits(axis_label_batch, outputs_soft2)
            loss1 = ce_loss1(outputs, label_batch_radical[:].long()) + dice_loss(outputs_soft, label_batch_radical.unsqueeze(1))
            loss2 = ce_loss2(outputs2, label_batch_conservative[:].long()) + dice_loss(outputs_soft2, label_batch_conservative.unsqueeze(1))
            
            shield_mask = torch.zeros_like(outputs)
            shield_mask = label_batch_radical - label_batch_conservative
            supv_loss = loss1 + loss2
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            if args.ProjectionLoss and iter_num <= 10000:
                loss_proj = loss_box_1 + loss_box_2
                supv_loss += 0.3 * loss_proj
            
            pseudo_supervision1 = (pseudo_supervision1 * shield_mask).sum() / (shield_mask.sum() + 1e-16) 
            pseudo_supervision2 = (pseudo_supervision2 * shield_mask).sum() / (shield_mask.sum() + 1e-16)
            model1_loss = consistency_weight * pseudo_supervision1
            model2_loss = consistency_weight * pseudo_supervision2
            loss = model1_loss + model2_loss + supv_loss
            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_1', loss1, iter_num)
            writer.add_scalar('info/loss_2', loss2, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss1: %f, loss2: %f' %
                (iter_num, loss.item(), loss1.item(), loss2.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                #labs = label_batch[1, ...].unsqueeze(0) * 50
                #writer.add_image('train/GroundTruth', labs, iter_num)

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
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    # print(sampled_batch["idx"])
                    metric_i = test_single_slice(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()
               

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

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

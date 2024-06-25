import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from scipy import stats
from tqdm import tqdm
import cv2
from PIL import Image
# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/node_data_ljy', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='node_data_ljy_SAM/SAM_CpsRampNew01_GPU2_fold1', help='experiment_name')
parser.add_argument('--exp1', type=str,
                    default='node_data_ljy/Weak_Supervised_GrabCut_MTCL_GPU2', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=50,
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jc = metric.binary.jc(pred, gt)
        recall = metric.binary.recall(pred, gt)
        precision = metric.binary.precision(pred, gt)
        return dice, hd95, asd, jc, recall, precision
        # return dice, hd95, asd, jc
    else:
        return 0, 0, 0, 0, 0, 0

def test_single_volume(case, net, FLAGS):
    case_temp = case.replace(".png", ".png")
    image = Image.open(FLAGS.root_path + "/imgs/{}".format(case_temp)).convert('L')
    image = np.array(image)/255.0
    label = Image.open(FLAGS.root_path + "/masks/{}".format(case))
    label = np.array(label)
    label[label>0]=1
    # h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    # image = h5f['image'][:]
    # label = h5f['label'][:]
    prediction = np.zeros_like(label)
    # for ind in range(image.shape[0]):
    slice = image[:, :]
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (256 / x, 256 / y), order=0)  # 将切片缩放到256
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        if FLAGS.model == "unet_urds":
            out_main, _, _, _ = net(input)
        else:
            out_main = net(input)
        out = torch.argmax(torch.softmax(
            out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / 256, y / 256), order=0) # 将预测结果缩放回原始大小
        prediction = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)

    return first_metric


def test_single_slice(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    # for ind in range(image.shape[0]):
    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(
            net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def Inference(FLAGS):
    with open(FLAGS.root_path + '/test_fold1.txt', 'r') as f:
        image_list = f.readlines()
    image_list = [item.replace('\n', '')
                                for item in image_list]
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    snapshot_path1 = "../model/{}_{}_labeled/{}".format(
        FLAGS.exp1, FLAGS.labeled_num, FLAGS.model)
    

    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    net1 = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model2.pth'.format(FLAGS.model))
    save_mode_path1 = os.path.join(
        snapshot_path1, '{}_best_model.pth'.format(FLAGS.model))
    # save_mode_path1 = os.path.join(
    #     snapshot_path1, 'iter_8800_dice_0.7473.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    net1.load_state_dict(torch.load(save_mode_path1))
    print("init weight from {}".format(save_mode_path1))
    net1.eval()

    first_metrics = []
    for case in tqdm(image_list):
        first_metric = test_single_volume(case, net, FLAGS)
        first_metrics.append(first_metric)
    first_metrics = np.array(first_metrics)
    first_metrics1 = []
    for case in tqdm(image_list):
        first_metric1 = test_single_volume(case, net1, FLAGS)
        first_metrics1.append(first_metric1)
    first_metrics1 = np.array(first_metrics1)
    n_metrics = first_metrics.shape[1]

    for i in range(n_metrics):
        t_statistic, p_value = stats.ttest_rel(first_metrics[:, i], first_metrics1[:, i])
        print(f"For metric {i + 1}:")
        print("T statistic:", t_statistic)
        print("P value:", p_value)
        print("\n")
    # print("*" * 10)
    # print(first_metrics1.shape)
    # print("*" * 10)
    # avg_metrics = np.mean(first_metrics, axis=0)
    # std_metrics = np.std(first_metrics, axis=0)
    # return avg_metrics, std_metrics


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)
    # avg_metrics, std_metrics = Inference(FLAGS)
    # print(avg_metrics)
    # print(std_metrics)
    # print((metric[0]+metric[1]+metric[2])/3)

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
from tqdm import tqdm
import cv2
from PIL import Image
# from networks.efficientunet import UNet
from networks.net_factory import net_factory
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/node_data_ljy', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='node_data_ljy_SAM/SAM_UTReverse_GPU1_fold1_Eight_EllipseLabel', help='experiment_name')
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
    else:
        return 0, 0, 0, 0, 0, 0

def test_single_volume(case, net, test_save_path, FLAGS):
    image = Image.open(FLAGS.root_path + "/imgs/{}".format(case)).convert('L')
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
    # print("---------------------")
    # print("case:",case)
    # # print("prediction.shape:", prediction.shape)
    # # print("label.shape:", label.shape)
    # print(np.unique(prediction))
    # print(np.unique(label))
    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    # second_metric = calculate_metric_percase(prediction == 2, label == 2)
    # third_metric = calculate_metric_percase(prediction == 3, label == 3)

    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.SetSpacing((1, 1, 10))
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.SetSpacing((1, 1, 10))
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    cv2.imwrite(test_save_path + case[:-4] + "_img.png", image.astype(np.float32)*255)
    cv2.imwrite(test_save_path + case[:-4] + "_label.png", label.astype(np.float32)*255)
    cv2.imwrite(test_save_path + case[:-4] + "_prediction.png", prediction.astype(np.float32)*255)
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
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    # save_mode_path = os.path.join(
    #     snapshot_path, 'iter_12200_dice_0.6772.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    # second_total = 0.0
    # third_total = 0.0
    # for case in tqdm(image_list):
    #     first_metric = test_single_volume(
    #         case, net, test_save_path, FLAGS)
    #     first_total += np.asarray(first_metric)
    #     # second_total += np.asarray(second_metric)
    #     # third_total += np.asarray(third_metric)
    # # avg_metric = [first_total / len(image_list)]
    # avg_metric = np.mean(first_total)
    # std_metric = np.std(first_total)
    # return avg_metric, std_metric
    # return avg_metric
    first_metrics = []
    for case in tqdm(image_list):
        first_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_metrics.append(first_metric)
    first_metrics = np.array(first_metrics)  # Convert list of list to 2D np.array

    avg_metrics = np.mean(first_metrics, axis=0)
    std_metrics = np.std(first_metrics, axis=0)
    return avg_metrics, std_metrics


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    avg_metric, std_metric = Inference(FLAGS)
    print(avg_metric)
    print(std_metric)
    # print((metric[0]+metric[1]+metric[2])/3)

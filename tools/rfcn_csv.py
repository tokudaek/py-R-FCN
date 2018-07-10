#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import matplotlib
matplotlib.use('Agg'); flag_agg = True #Deactivate X
import matplotlib.pyplot as plt

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import inspect
import time
import os

HOME = os.environ['HOME']

####################### PARAMS #############################
CONF_THRESH = 0.05
NMS_THRESH = 0.3

METHODS = ['faster_rcnn_end2end', 'faster_rcnn_alt_opt', 'fast_rcnn']

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
           #'car')

NETS = {'ResNet-101': ('ResNet-101',
                       'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                      'resnet50_rfcn_final.caffemodel')}
##########################################################

def main():
    args = parse_args()
    list_files = args.list_files
    in_dir = args.in_dir
    out_dir = args.out_dir
    caffemodel = args.caffemodel
    prototxt = args.prototxt

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if flag_agg:
        print "##########################################################"
        print "Redirecting X..."
        print "##########################################################"

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    print ("prototxt: " + prototxt)
    print ("caffemodel: " + caffemodel)

    if not os.path.isfile(caffemodel):
        raise IOError(caffemodel)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id

    print '\n\nLoading network {:s}...'.format(caffemodel)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print('Network loaded.')

    
    if list_files:
        with open(list_files) as fh:
            list_files = fh.read().splitlines()
    else:
        allfiles = os.listdir(in_dir)
        list_files = []
        for file_ in allfiles:
            if file_.endswith(".jpg"):
                list_files.append(file_)

    allfigstime = 0
    for im_name in list_files:
        if im_name:
            detectiontime = demo(net, in_dir, out_dir, im_name)
            allfigstime += detectiontime
    print('Time to perform detection in all images:{}s'.format(allfigstime))

def export_bboxes(im_file, outdir, image_name, class_name, dets, thresh=0.1):
    """Export to CSV"""

    inds = np.where(dets[:, -1] >= thresh)[0]
    outcsv = os.path.join(outdir, image_name.replace('.jpg', '.csv'))
    np.savetxt(outcsv, dets[inds], delimiter=',', fmt='%d,%d,%d,%d,%f')

def vis_detections(im, out_dir, image_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""

    inds = np.where(dets[:, -1] >= thresh)[0]

    out_file = os.path.join(out_dir, os.path.splitext(image_name)[0] +
                            class_name +  
                            os.path.splitext(image_name)[1])
    if len(inds) == 0:  #In case it does not have bboxes
        if not os.path.isfile(out_file):
            im = im[:, :, (2, 1, 0)]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(im, aspect='equal')
            plt.savefig(out_file)
            plt.close()
        return

    im = im[:, :, (2, 1, 0)] #BGR
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    #plt.draw()

    #print (os.path.join(out_dir, image_name))
    plt.savefig(out_file)
    plt.close()

def demo(net, in_dir, out_dir, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    #im_file = os.path.join(in_dir, image_name)
    im_file = image_name
    image_name = os.path.basename(im_file)
    #print(im_file)
    im = cv2.imread(os.path.join(in_dir, im_file))

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()

    #print(os.path.join(in_dir, im_file))
    #print(im)

    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    for cls_ind, cls in [(15, 'person')]:
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(im, out_dir, image_name, cls, dets, thresh=CONF_THRESH)
        export_bboxes(im_file, out_dir, image_name, cls, dets, thresh=CONF_THRESH)
    return timer.total_time

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Py-R-FCN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='ResNet-101')
    parser.add_argument('--caffemodel', dest='caffemodel', help='caffemodel',
                         default='data/rfcn_models/resnet101_rfcn_final.caffemodel')
    parser.add_argument('--prototxt', dest='prototxt', help='prototxt',
                         default='models/pascal_voc/ResNet-101/rfcn_end2end/test_agnostic.prototxt')
    parser.add_argument('--list_files', dest='list_files', help='list of files',
                        type=str)
    parser.add_argument('--in_dir', dest='in_dir', help='Path to the input',
                        type=str, required=True)
    parser.add_argument('--out_dir', dest='out_dir', help='Path to the output',
                        type=str, required=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()

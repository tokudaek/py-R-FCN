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

import psycopg2
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import utils
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import inspect
import time
import os
import sys
import logging
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
import myutils

HOME = os.environ['HOME']

####################### PARAMS #############################
CONF_THRESH = 0.25
NMS_THRESH = 0.3

METHODS = ['faster_rcnn_end2end', 'faster_rcnn_alt_opt', 'fast_rcnn']

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

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
    gpuid = args.gpu_id
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    dbjson = 'config/db.json'
    conn = myutils.db_connect(dbjson)
    conn.autocommit = False

    if flag_agg:
        print "##########################################################"
        print "Redirecting X..."
        print "##########################################################"

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    print ("prototxt: " + prototxt)
    print ("caffemodel: " + caffemodel)

    if not os.path.isfile(caffemodel):
        raise IOError(caffemodel)

    methodid = 7
    thresh = CONF_THRESH
    nms = NMS_THRESH
    ids, relpaths, rolls = myutils.db_get_nonprocessed_images(conn, methodid)
    run(ids, relpaths, rolls, in_dir, out_dir, caffemodel, prototxt, gpuid,
        methodid, _buffer, dbjson, thresh, nms, _delete)

##########################################################
def run(ids, relpaths, rolls, in_dir, out_dir, caffemodel, prototxt, gpuid,
        methodid, dbjson, _buffer, logfile, _thresh, _nms, _delete):

    conn = myutils.db_connect(dbjson)
    conn.autocommit = False

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    logging.basicConfig(filename=logfile, level=logging.DEBUG)
    #print ("prototxt: " + prototxt)
    #print ("caffemodel: " + caffemodel)
    logging.debug("prototxt: " + prototxt)
    logging.debug("caffemodel: " + caffemodel)

    if not os.path.isfile(caffemodel):
        raise IOError(caffemodel)

    caffe.set_mode_gpu()
    caffe.set_device(gpuid)
    cfg.GPU_ID = gpuid

    #print '\n\nLoading network {:s}...'.format(caffemodel)
    logging.debug ('\n\nLoading network {:s}...'.format(caffemodel))
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    #print('Network loaded.')
    logging.debug('Network loaded')

    #LENBUFFER = 20
    counter = 0

    for relpath, id, roll in zip(relpaths, ids, rolls):
        commit = counter % _buffer == 0
        try:
            added = demo(conn, net, in_dir, out_dir, relpath, id, roll,
                         methodid, _thresh, _nms, commit, _delete)
            logging.debug('{}: {}'.format(counter, id))
        except psycopg2.Error as e:
            logging.debug('Could not process ' + str(id))
            added = 0

        counter += added

    conn.close()
    return counter

def vis_detections(im, out_dir, image_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""

    inds = np.where(dets[:, -1] >= thresh)[0]

    if len(inds) == 0:  #In case it does not have bboxes
        out_file = os.path.join(out_dir, image_name)
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

    #print(image_name) #TODO: Get the id
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

    plt.savefig(os.path.join(out_dir, image_name))
    plt.close()

#########################################################
def detectandstore(conn, im, updown, id, class_name, dets, methodid, thresh):
    """Draw detected bounding boxes."""

    inds = np.where(dets[:, -1] >= thresh)[0]
    hh,ww,_ = im.shape
    c = [ww/2, hh/2, ww/2, hh/2]

    cur = conn.cursor()

    if len(inds) != 0:  #In case it has bboxes
        im = im[:, :, (2, 1, 0)] #BGR

        for i in inds:  #For each bbox of thisclass
            bbox = dets[i, :4]
            score = dets[i, -1]
            b = bbox

            query = '''INSERT INTO tek.Bbox ''' \
                ''' (imageid, prob, x_min, y_min, x_max, y_max, methodid, classid) ''' \
                ''' SELECT {},{},{},{},{},{},{},Class.id from tek.Class WHERE name='{}'; ''' \
            .format(id, score, b[0], b[1], b[2], b[3], methodid, class_name)
            #print(query)
            cur.execute(query)

#########################################################
def demo(conn, net, in_dir, out_dir, relpath, id, roll, methodid, _nms,
         _thresh, commit=True, remove=False):
    """Detect object classes in an image using pre-computed object proposals."""

    im_file = os.path.join(in_dir, relpath)
    im = cv2.imread(im_file)
    updown = False
    if (im is None): return 0

    if int(roll) > 0: # Rotate 180deg the image
        rows,cols, _ = im.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        im = cv2.warpAffine(im,M,(cols,rows))
        updown = True

    scores, boxes = im_detect(net, im)

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1

        #cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, _nms)
        dets = dets[keep, :]
        #vis_detections(im, out_dir, image_name, cls, dets, thresh=CONF_THRESH)
        detectandstore(conn, im, updown, id, cls, dets, methodid, _thresh)

    cur = conn.cursor()
    query = '''INSERT INTO tek.ImageMethod (imageid,methodid) VALUES ({},{});''' \
        .format(id, methodid)

    cur.execute(query)
    if commit: conn.commit()
    if remove:
        try: os.remove(os.path.join(in_dir, relpath))
        except: pass
    return 1

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
    parser.add_argument('--in_dir', dest='in_dir', help='Path to the output',
                        type=str, required=True)
    parser.add_argument('--out_dir', dest='out_dir', help='Path to the output',
                        type=str, required=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()

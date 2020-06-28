#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import resnet
import cv2
import numpy as np
import json
from collections import Counter
import requests
from flask import Flask, request, redirect, jsonify, url_for, abort, make_response
import sys
import os
import base64

# constants
load_size = [256,256,3]
crop_size = [224,224,3]
batch_size = 512
num_classes = [48, 12, 2, 2, 2, 2, 2, 2]
num_channels = 3
samples = 1
num_batches = -1
batch_size = 1
if num_batches==-1:
    if(samples%batch_size==0):
      num_batches= int(samples/batch_size)
    else:
      num_batches= int(samples/batch_size)+1
num_threads = 20
depth = 152
#ckpt_path = '/home/nourfayed/Flask_Server/phase2'  # Directory of the checkpoints
ckpt_path = '/root/phase2'

def _test_preprocess(reshaped_image, crop_size, num_channels):

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, crop_size[0], crop_size[1])
  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)
  # Set the shapes of tensors.
  float_image.set_shape([crop_size[0], crop_size[1], num_channels])

  return float_image

def decode_img(img_str):
            img_bytes = bytes(img_str, 'utf-8')
            img_buff = base64.b64decode(img_bytes)
            img_jpg = np.frombuffer(img_buff, dtype=np.uint8)
            img = cv2.imdecode(img_jpg, cv2.IMREAD_COLOR)
            return img  

def read_img(image,desired_width = 256, desired_height = 256):
  img = cv2.resize(image, (desired_width, desired_height))
  _, img = cv2.imencode('.jpg', img)
  img = img.tostring()
  return img     

def inference_model(image,input_img):
  img = read_img(input_img)
  top5guesses_id, top5conf, top3guesses_cn, top3conf, top1guesses_bh, top1conf = sess.run([top5ind_id, top5val_id, top3ind_cn, top3val_cn, top1ind_bh, top1val_bh], feed_dict = {image: img})
  activity = np.zeros(6)
  confidence = np.zeros(6)
  for i in range(0,6):
    activity[i] = top1guesses_bh[i][0][0]
    confidence[i] = top1conf[i][0][0]
  return activity, confidence 

app = Flask(__name__)

g = tf.Graph().as_default()
tf.device('/cpu:0')
# Get images and labels.
image = tf.placeholder(tf.string, name='input')
reshaped_image = tf.to_float(tf.image.decode_jpeg(image, channels = num_channels))
reshaped_image = tf.image.resize_images(reshaped_image, (load_size[0], load_size[1]))
reshaped_image = _test_preprocess(reshaped_image, crop_size, num_channels)
imgs = reshaped_image[None, ...]
# Performing computations on a GPU
tf.device('/gpu:0')
# Build a Graph that computes the logits predictions from the
# inference model.
logits = resnet.inference(imgs, depth, num_classes, 0.0, False)
top5_id = tf.nn.top_k(tf.nn.softmax(logits[0]), 5)
top5ind_id= top5_id.indices
top5val_id= top5_id.values
# Count
top3_cn = tf.nn.top_k(tf.nn.softmax(logits[1]), 3)
top3ind_cn= top3_cn.indices
top3val_cn= top3_cn.values
# Additional Attributes (e.g. description)
top1_bh= [None]*6
top1ind_bh= [None]*6
top1val_bh= [None]*6

for i in range(0,6):
  top1_bh[i]= tf.nn.top_k(tf.nn.softmax(logits[i+2]), 1)
  top1ind_bh[i]= top1_bh[i].indices
  top1val_bh[i]= top1_bh[i].values
  
saver = tf.train.Saver(tf.global_variables())  
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
ckpt = tf.train.get_checkpoint_state(ckpt_path)
print(ckpt_path)
print(ckpt)
if ckpt: #and ckpt.model_checkpoint_path:
# Restores from checkpoint
  saver.restore(sess, ckpt.model_checkpoint_path)
  print('pass')
else:
  print('error')  

@app.route('/model/api/v1.0/recognize', methods=['POST'])
def recognize_activity():
  if not request.json or not 'img' in request.json:
    abort(204)
  img = decode_img(request.json['img'])
  activity, confidence = inference_model(image,img)

  return make_response(jsonify({'Status: ': 'finished', 'activity': json.dumps(activity.tolist()), 'confidence': json.dumps(confidence.tolist())}), 200)   

  
if __name__ == '__main__':
            app.run(host='0.0.0.0')           

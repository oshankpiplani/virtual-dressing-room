from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import scipy.io as sio
import cv2
import argparse
from glob import glob
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
from PIL import Image
from utils.image_reade_inf import *
from utils.ops import  *
from utils.utils import *
from utils.model_pgn import *

def parse_args():
    """Parse command line arguments"""
    argp = argparse.ArgumentParser(description="Inference pipeline")
    argp.add_argument('--input', required=True, help='Path to the input image')
    argp.add_argument('--output', required=True, help='Path to save output segmentation')
    argp.add_argument('--checkpoint', type=str, help='Path to the checkpoint directory', 
                      default=r'C:\Users\singh\project\ml\preprocessing\segmentation\CIHP_PGN\checkpoint\CIHP_pgn')
    argp.add_argument('--val-id-file', type=str, help='Path to val_id.txt file',
                     default='./val_id.txt')
    argp.add_argument('--val-file', type=str, help='Path to val.txt file',
                     default='./val.txt')
    return argp.parse_args()

def main():
    """Create the model and start the evaluation process."""
    args = parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input image '{args.input}' does not exist!")
        return 1
    
    # Check val files
    if not os.path.exists(args.val_id_file):
        print(f"Warning: Val ID file '{args.val_id_file}' does not exist!")
        # Create a default val_id.txt with the input image name
        try:
            img_basename = os.path.basename(args.input)
            img_name_no_ext = os.path.splitext(img_basename)[0]
            
            with open(args.val_id_file, 'w') as f:
                f.write(f"{img_name_no_ext}\n")
            print(f"Created default val_id.txt with image name: {img_name_no_ext}")
        except Exception as e:
            print(f"Error creating val_id.txt: {e}")
            return 1
    
    if not os.path.exists(args.val_file):
        print(f"Warning: Val file '{args.val_file}' does not exist!")
        # Create a default val.txt file
        try:
            img_basename = os.path.basename(args.input)
            img_name_no_ext = os.path.splitext(img_basename)[0]
            
            with open(args.val_file, 'w') as f:
                f.write(f"/images/{img_name_no_ext}.jpg /labels/{img_name_no_ext}.png\n")
            print(f"Created default val.txt with image path: /images/{img_name_no_ext}.jpg")
        except Exception as e:
            print(f"Error creating val.txt: {e}")
            return 1
    
    # Make sure output directory exists
    output_dir = os.path.dirname(args.output)
    parsing_dir = os.path.join(output_dir, 'cihp_parsing_maps')
    edge_dir = os.path.join(output_dir, 'cihp_edge_maps')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(parsing_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)
    
    # Create a single-element image list
    image_list_inp = [args.input]
    
    # Constants
    N_CLASSES = 20
    NUM_STEPS = 1  # We're only processing one image
    RESTORE_FROM = args.checkpoint
    
    # Check if checkpoint directory exists
    print(f"Checkpoint directory: {RESTORE_FROM}")
    print(f"Checkpoint exists: {os.path.exists(RESTORE_FROM)}")
    
    # Create queue coordinator
    coord = tf.train.Coordinator()
    
    # Load reader
    with tf.name_scope("create_inputs"):
        reader = ImageReader(image_list_inp, None, False, False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
    
    image_batch = tf.stack([image, image_rev])
    h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
    image_batch050 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.50)), tf.to_int32(tf.multiply(w_orig, 0.50))]))
    image_batch075 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
    image_batch125 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.25)), tf.to_int32(tf.multiply(w_orig, 1.25))]))
    image_batch150 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.50)), tf.to_int32(tf.multiply(w_orig, 1.50))]))
    image_batch175 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.75)), tf.to_int32(tf.multiply(w_orig, 1.75))]))
         
    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_050 = PGNModel({'data': image_batch050}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_075 = PGNModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_125 = PGNModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_150 = PGNModel({'data': image_batch150}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_175 = PGNModel({'data': image_batch175}, is_training=False, n_classes=N_CLASSES)
    
    # parsing net
    parsing_out1_050 = net_050.layers['parsing_fc']
    parsing_out1_075 = net_075.layers['parsing_fc']
    parsing_out1_100 = net_100.layers['parsing_fc']
    parsing_out1_125 = net_125.layers['parsing_fc']
    parsing_out1_150 = net_150.layers['parsing_fc']
    parsing_out1_175 = net_175.layers['parsing_fc']

    parsing_out2_050 = net_050.layers['parsing_rf_fc']
    parsing_out2_075 = net_075.layers['parsing_rf_fc']
    parsing_out2_100 = net_100.layers['parsing_rf_fc']
    parsing_out2_125 = net_125.layers['parsing_rf_fc']
    parsing_out2_150 = net_150.layers['parsing_rf_fc']
    parsing_out2_175 = net_175.layers['parsing_rf_fc']

    # edge net
    edge_out2_100 = net_100.layers['edge_rf_fc']
    edge_out2_125 = net_125.layers['edge_rf_fc']
    edge_out2_150 = net_150.layers['edge_rf_fc']
    edge_out2_175 = net_175.layers['edge_rf_fc']

    # combine resize
    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_050, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_075, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_100, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_125, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_150, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_175, tf.shape(image_batch)[1:3,])]), axis=0)

    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_050, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_075, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_100, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_125, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_150, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_175, tf.shape(image_batch)[1:3,])]), axis=0)

    edge_out2_100 = tf.image.resize_images(edge_out2_100, tf.shape(image_batch)[1:3,])
    edge_out2_125 = tf.image.resize_images(edge_out2_125, tf.shape(image_batch)[1:3,])
    edge_out2_150 = tf.image.resize_images(edge_out2_150, tf.shape(image_batch)[1:3,])
    edge_out2_175 = tf.image.resize_images(edge_out2_175, tf.shape(image_batch)[1:3,])
    edge_out2 = tf.reduce_mean(tf.stack([edge_out2_100, edge_out2_125, edge_out2_150, edge_out2_175]), axis=0)
                                           
    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))
    
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    pred_scores = tf.reduce_max(raw_output_all, axis=3)
    raw_output_all = tf.argmax(raw_output_all, axis=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.

    raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
    tail_output_rev = tf.reverse(tail_output, tf.stack([1]))
    raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_edge_all = tf.expand_dims(raw_edge_all, dim=0)
    pred_edge = tf.sigmoid(raw_edge_all)
    res_edge = tf.cast(tf.greater(pred_edge, 0.5), tf.int32)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.compat.v1.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    try:
        # Run inference
        parsing_, scores, edge_ = sess.run([pred_all, pred_scores, pred_edge])
        
        # Get filename without path and extension
        img_id = os.path.basename(args.input).split('.')[0]
        
        # Create visualization
        msk = decode_labels(parsing_, num_classes=N_CLASSES)
        parsing_im = Image.fromarray(msk[0])
        
        # Save outputs
        parsing_vis_path = os.path.join(parsing_dir, f'{img_id}_vis.png')
        parsing_path = os.path.join(parsing_dir, f'{img_id}.png')
        edge_path = os.path.join(edge_dir, f'{img_id}.png')
        
        parsing_im.save(parsing_vis_path)
        cv2.imwrite(parsing_path, parsing_[0,:,:,0])
        cv2.imwrite(edge_path, edge_[0,:,:,0] * 255)
        
        # Copy the main segmentation output to the specified output path
        cv2.imwrite(args.output, parsing_[0,:,:,0])
        
        print(f"Saved segmentation to {args.output}")
        print(f"Saved visualization to {parsing_vis_path}")
        print(f"Saved edge map to {edge_path}")
        
        return 0
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()
        print("Finished processing")

if __name__ == '__main__':
    sys.exit(main())
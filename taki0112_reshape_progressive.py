#
#   aydao ~ 2019
#
#   Use this if you started training a taki0112 model from scratch
#   First, run this script first (example usage below)
#   This will copy over the layers from your original model into another suited for the next script
#   The script will drop a taki0112 checkpoint with its counter set to "2" (chosen arbitrarily by me)
#   At that point, you can run other taki0112_to_nvlabs script to convert it to the nvlabs.pkl
#
#
from StyleGAN import StyleGAN
import argparse
from utils import *
import os
import pickle
import numpy as np
import PIL.Image
import copy
import dnnlib
import dnnlib.tflib as tflib
from dnnlib import EasyDict
import config
from metrics import metric_base
from training.training_loop import process_reals
from training import misc
import tensorflow as tf
import sys, time

"""parsing and configuration"""
def parse_args():
    desc = "Convert taki0112 StyleGAN checkpoint to NVlabs StyleGAN pkl (copies over the model weights)"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--cache_dir", type=str, default="./",
                        help="The cache directory to save the NVlabs pkl in")

    parser.add_argument("--dataset", type=str, default="FFHQ",
                        help="The dataset name what you want to generate")
                        
    parser.add_argument("--gpu_num", type=int, default=1, help="The number of gpu")

    parser.add_argument("--start_res", type=int, default=8, help="The number of starting resolution")
    
    parser.add_argument("--img_size", type=int, default=1024, help="The target size of image")
    parser.add_argument("--progressive", type=str2bool, default=True, help="use progressive training")

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint/",
                        help="Directory name to save the checkpoints")

    parser.add_argument("--result_dir", type=str, default="./results/",
                        help="Directory name to save the generated images")

    # Extra args for taki0112 code
    # Do not change these args, the code will automatically set them for you
    # parser.add_argument("--progressive", help=argparse.SUPPRESS)
    parser.add_argument("--phase", help=argparse.SUPPRESS)
    parser.add_argument("--sn", help=argparse.SUPPRESS)

    args = parser.parse_args()
    args = check_args(args)
    args = handle_extra_args(args)
    return args

"""checking arguments"""
def check_args(args):

    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    return args
    
"""handle the extra args based on required args"""
def handle_extra_args(args):
    args.phase = "train"
    # assuming you are not using spectral normalization since NVlabs does not use it
    # you probably *could* figure out how to convert with sn, but it'd take some more tinkering
    args.sn = False
    # preempt the taki0112 code from making needless directories...
    args.sample_dir = args.result_dir
    args.log_dir = args.result_dir
    # magic values not needed for this script
    args.iteration = 0
    args.max_iteration = 2500
    args.batch_size = 1
    args.test_num = 1
    args.seed = True
    return args

def make_temp_dataset_file(dataset_dir):
    filename = dataset_dir + "/nvlabs_to_taki0112_tempfile.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("this file is safe to delete")
    # give it a second
    time.sleep(1)
    return filename

def delete_temp_dataset_file(args, dataset_dir, filename):
    try:
        os.remove(filename)
        if len(os.listdir(dataset_dir)) == 0:
            os.rmdir(dataset_dir)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

"""main"""
def main():
    #
    #
    #   Example usage: python taki0112_reshape_progresive.py --dataset FFHQ --start_res 8 --img_size 512
    #
    #
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    checkpoint_dir = args.checkpoint_dir
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        # this is a hack since the taki0112 expects a dataset folder which may not exist
        dataset = args.dataset
        dataset_dir = "./dataset/" + dataset
        temp_dataset_file = make_temp_dataset_file(dataset_dir)

        # build the taki0112 StyleGAN architecture (vanilla Tensorflow)
        gan = StyleGAN(sess, args)
        
        # you have to go through this process to initialize everything needed to load the checkpoint...
        gan.build_model()
        
        # remove the temp file and the directory if it is empty
        delete_temp_dataset_file(args, dataset_dir, temp_dataset_file)
        
        # Initialize TensorFlow.
        tflib.init_tf()
        
        tf.global_variables_initializer().run()
        gan.saver = tf.train.Saver(max_to_keep=10)
        gan.load(checkpoint_dir)
        
        copy_layers = []
        vars = tf.trainable_variables("discriminator")
        vars_vals = sess.run(vars)
        for var, val in zip(vars, vars_vals):
            copy_layers.append((var.name,val))
        vars = tf.trainable_variables("generator")
        vars_vals = sess.run(vars)
        for var, val in zip(vars, vars_vals):
            copy_layers.append((var.name,val))
        
    return args, copy_layers
    
def copy_over(args, copy_layers):
    tf.reset_default_graph() 
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        checkpoint_dir = args.checkpoint_dir
        args.progressive = True
        dataset = args.dataset
        dataset_dir = "./dataset/" + dataset
        temp_dataset_file = make_temp_dataset_file(dataset_dir)
        gan2 = StyleGAN(sess, args)
        gan2.build_model()
        delete_temp_dataset_file(args, dataset_dir, temp_dataset_file)
        tflib.init_tf()
        tf.global_variables_initializer().run()
        gan2.saver = tf.train.Saver(max_to_keep=10)
        gan2.load(checkpoint_dir)
        
        update_layers = []
        variables = []
        variables += tf.trainable_variables("discriminator")
        variables += tf.trainable_variables("generator")
        
        variable_dict = {}
        for variable in variables:
            variable_name = variable.name
            variable_dict[variable_name] = variable
            
        for copy_layer in copy_layers:
            copy_name, copy_value = copy_layer
            variable = variable_dict[copy_name]
            update_layer = tf.assign(variable, copy_value)
            update_layers.append(update_layer)
            
        sess.run(update_layers)
        
        # just picking 2 as the counter for the taki model
        counter = 2
        gan2.save(checkpoint_dir, counter)
            
if __name__ == "__main__":
    args, copy_layers = main()
    copy_over(args, copy_layers)

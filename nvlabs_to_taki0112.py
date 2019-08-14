#
#   aydao ~ 2019
#
#   Convert network-snapshot-######.pkl StyleGAN models to more general tensorflow checkpoints
#
#   This script relies on both the NVlabs StyleGAN GitHub repository and the taki0112 StyleGAN repo
#   It assumes both are in the same directory as this script
#
from StyleGAN import StyleGAN
import argparse
from utils import *
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import tensorflow as tf
import sys

"""parsing and configuration"""
def parse_args():
    desc = "Convert NVlabs StyleGAN pkl to taki0112 StyleGAN checkpoint (copies over the model weights)"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--nvlabs", type=str, help="The source NVlabs StyleGAN, a network .pkl file")

    parser.add_argument("--dataset", type=str, default="FFHQ",
                        help="The dataset name what you want to generate")
                        
    parser.add_argument("--gpu_num", type=int, default=1, help="The number of gpu")

    parser.add_argument("--sn", type=str2bool, default=False, help="use spectral normalization")

    parser.add_argument("--img_size", type=int, default=1024, help="The target size of image")

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint/",
                        help="Directory name to save the checkpoints")

    parser.add_argument("--result_dir", type=str, default="./results/",
                        help="Directory name to save the generated images")

    # Extra args for taki0112 code
    # Do not change these args, the code will automatically set them for you
    parser.add_argument('--start_res', help=argparse.SUPPRESS)
    parser.add_argument('--progressive', help=argparse.SUPPRESS)
    parser.add_argument('--phase', help=argparse.SUPPRESS)

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
    # these are things you either do not need for this script, or can be set automatically
    args.start_res = args.img_size
    # assuming you are trasnferring from a progressive NVlabs StyleGAN model
    # though when you continue training you likely do not want it progressive any more
    args.progressive = True
    args.phase = "train"
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


"""main"""
def main():
    #
    #    Usage example:
    #    python [this_file].py --nvlabs ./cache/karras2019stylegan-ffhq-1024x1024.pkl
    #        --dataset FFHQ --img_size 1024 --gpu_num 2
    #
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    
    
    checkpoint_dir = args.checkpoint_dir
    nvlabs_stylegan_pkl_name = args.nvlabs
    # the taki0112 StyleGAN models expect the following naming prefix
    model_name = "StyleGAN.model"
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    
        gan = StyleGAN(sess, args)

        gan.build_model()
        # now that a progressive model is built, turn off progressive
        # the progressive structure is needed to copy over from NVlabs, but you won't need this in training later
        # basically, this just saves this out in a folder without the _progressive tag in the directory name
        # this shouldn't cause any weird side effects. Probably.
        args.progressive = False
        gan.progressive = False

        tflib.init_tf()

        gan.sess = sess
        tf.global_variables_initializer().run()
        gan.saver = tf.train.Saver(max_to_keep=10)
        gan.writer = tf.summary.FileWriter(gan.log_dir + "/" + gan.model_dir, gan.sess.graph)
        
        counter = 0
        gan.save(checkpoint_dir, counter)
        # Or, you can use this instead if you want to base the taki0112 network on an existing one
        # gan.load(checkpoint_dir)
        
        # Now moving to NVlabs code
        src_d = "D"
        dst_d = "discriminator"
        src_gs = "G_synthesis_1" # "G_synthesis"
        dst_gs = "generator/g_synthesis"
        src_gm = "G_mapping_1" # "G_mapping"
        dst_gm = "generator/g_mapping"
        
        # Load the existing NVlabs StyleGAN network
        G, D, Gs = pickle.load(open(nvlabs_stylegan_pkl_name, "rb"))
        
        vars = tf.trainable_variables(src_gm)
        vars_vals = sess.run(vars)
        
        # Copy over the discriminator weights
        for (new, old) in zip(tf.trainable_variables(dst_d), tf.trainable_variables(src_d)):
            update_weight = [tf.assign(new, old)]
            sess.run(update_weight)
            temp_vals = sess.run([new, old])
        
        # Copy over the Generator's mapping network weights
        for (new, old) in zip(tf.trainable_variables(dst_gm), tf.trainable_variables(src_gm)):
            update_weight = [tf.assign(new, old)]
            sess.run(update_weight)
            temp_vals = sess.run([new, old])
        
        # Because the two network architectures use slightly different columns on one variable,
        # you must set up code to handle the edge case transpose of the first case
        first = True
        for (new, old) in zip(tf.trainable_variables(dst_gs), tf.trainable_variables(src_gs)):
            temp_vals = sess.run([new, old])
            if new.shape != old.shape:
                # you need a transpose with perm # old = tf.reshape(old, tf.shape(new))
                # DO NOT USE RESHAPE (made this mistake here and the results work but are quite terrifying)
                if (first):
                    first = False
                    old = tf.transpose(old, perm=[0, 2, 3, 1])
                else:
                    old = tf.transpose(old, perm=[0, 1, 3, 2])
            update_weight = [tf.assign(new, old)]
            sess.run(update_weight)
            
        # Also, assign the NVlabs Gs dlatent_avg to the w_avg in the taki0112 network
        new = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator") if "avg" in str(x)][0]
        old = G.get_var("dlatent_avg")
        update_weight = [tf.assign(new, old)]
        sess.run(update_weight)
        vars = [new]
        vars_vals = gan.sess.run(vars)
        vars_vals = sess.run(vars)
        
        # Save the new taki0112 StyleGAN checkpoint
        # I elect to set the counter to 1 here to differentiate from the souce checkpoint (at 0)
        gan.saver = tf.train.Saver(max_to_keep=10)
        counter = 1
        gan.save(checkpoint_dir, counter)


if __name__ == "__main__":
    main()

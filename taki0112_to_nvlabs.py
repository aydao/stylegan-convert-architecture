#
#   aydao ~ 2019
#
#   Convert taki0112 StyleGAN checkpoints to network-snapshot-######.pkl StyleGAN models
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

    parser.add_argument("--kimg", type=str, help="kimg/iteration of the NVlabs pkl, use format ######")
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
    # these are things you either do not need for this script, or can be set automatically
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
    #    Usage example:
    #    python [this_file].py --kimg ###### --dataset [your data] --gpu_num 1 
    #       --start_res 8 --img_size 512 --progressive True
    #
    #
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    checkpoint_dir = args.checkpoint_dir
    nvlabs_stylegan_pkl_kimg = args.kimg
    nvlabs_stylegan_pkl_name = "network-snapshot-"+nvlabs_stylegan_pkl_kimg+".pkl"
    
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        # this is a hack since the taki0112 expects a dataset folder which may not exist
        dataset = args.dataset
        dataset_dir = "./dataset/" + dataset
        temp_dataset_file = make_temp_dataset_file(dataset_dir)
        
        
        # build the taki0112 StyleGAN architecture (vanilla Tensorflow)
        gan = StyleGAN(sess, args)
        
        
        # you have to go through this process to initialize everything needed to load the checkpoint...
        original_start_res = args.start_res
        args.start_res = args.img_size
        gan.start_res = args.img_size
        gan.build_model()
        args.start_res = original_start_res
        gan.start_res = original_start_res
        
        # remove the temp file and the directory if it is empty
        delete_temp_dataset_file(args, dataset_dir, temp_dataset_file)
        
        # Initialize TensorFlow.
        tflib.init_tf()
        
        tf.global_variables_initializer().run()
        
        
        vars = tf.trainable_variables("discriminator")
        vars_vals = sess.run(vars)
        for var, val in zip(vars, vars_vals):
            print(var.name)
        
        gan.saver = tf.train.Saver(max_to_keep=10)
        gan.load(checkpoint_dir)
        
        #
        #
        #   Make an NVlabs StyleGAN network (default initialization)
        #
        #
        
        # StyleGAN initialization parameters and options, if you care to change them, do so here
        desc          = "sgan"                                                                 
        train         = EasyDict(run_func_name="training.training_loop.training_loop")         
        G             = EasyDict(func_name="training.networks_stylegan.G_style")               
        D             = EasyDict(func_name="training.networks_stylegan.D_basic")               
        G_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          
        D_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          
        G_loss        = EasyDict(func_name="training.loss.G_logistic_nonsaturating")           
        D_loss        = EasyDict(func_name="training.loss.D_logistic_simplegp", r1_gamma=10.0) 
        dataset       = EasyDict()                                                             
        sched         = EasyDict()                                                             
        grid          = EasyDict(size="4k", layout="random")                                   
        metrics       = [metric_base.fid50k]                                                   
        submit_config = dnnlib.SubmitConfig()                                                  
        tf_config     = {"rnd.np_random_seed": 1000}                                           
        drange_net              = [-1,1]
        G_smoothing_kimg        = 10.0
        
        # Dataset.
        desc += "-"+args.dataset
        dataset = EasyDict(tfrecord_dir=args.dataset)
        train.mirror_augment = True
        
        # Number of GPUs.
        gpu_num = args.gpu_num
        if gpu_num == 1:
            desc += "-1gpu"; submit_config.num_gpus = 1
            sched.minibatch_base = 4
            sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
        elif gpu_num == 2:
            desc += "-2gpu"; submit_config.num_gpus = 2
            sched.minibatch_base = 8
            sched.minibatch_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
        elif gpu_num == 4:
            desc += "-4gpu"; submit_config.num_gpus = 4
            sched.minibatch_base = 16
            sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
        elif gpu_num == 8:
            desc += "-8gpu"; submit_config.num_gpus = 8
            sched.minibatch_base = 32
            sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}
        else:
            print("ERROR: invalid number of gpus:",gpu_num)
            sys.exit(-1)

        # Default options.
        train.total_kimg = 0
        sched.lod_initial_resolution = 8
        sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)

        # Initialize dnnlib and TensorFlow.
        # ctx = dnnlib.RunContext(submit_config, train)
        tflib.init_tf(tf_config)

        # Construct networks.
        with tf.device('/gpu:0'):
            print('Constructing networks...')
            dataset_resolution = args.img_size
            dataset_channels = 3 # fairly sure everyone is using 3 channels ... # training_set.shape[0],
            dataset_label_size = 0 # training_set.label_size,
            G = tflib.Network('G',
                num_channels=dataset_channels,
                resolution=dataset_resolution,
                label_size=dataset_label_size,
                **G)
            D = tflib.Network('D',
                num_channels=dataset_channels,
                resolution=dataset_resolution,
                label_size=dataset_label_size,
                **D)
            Gs = G.clone('Gs')
        G.print_layers(); D.print_layers()

        print('Building TensorFlow graph...')
        with tf.name_scope('Inputs'), tf.device('/cpu:0'):
            lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
            lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
            minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
            minibatch_split = minibatch_in // submit_config.num_gpus
            Gs_beta         = 0.5 ** tf.div(tf.cast(minibatch_in, tf.float32),
                                G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0

        
        src_d = "discriminator"
        dst_d = "D"
        src_gs = "generator/g_synthesis"
        dst_gs = "G_synthesis" # "G_synthesis_1" <<<< this is handled later
        src_gm = "generator/g_mapping"
        dst_gm = "G_mapping" # "G_mapping_1" <<<< this is handled later
        
        
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
                # DO NOT USE RESHAPE
                # (made this mistake here and the results work but are quite terrifying)
                if (first):
                    first = False
                    old = tf.transpose(old, perm=[0, 3, 1, 2])
                else:
                    old = tf.transpose(old, perm=[0, 1, 3, 2])
            update_weight = [tf.assign(new, old)]
            sess.run(update_weight)
            
        # also update the running average network (not 100% sure this is necessary)
        dst_gs = "G_synthesis_1"
        dst_gm = "G_mapping_1"
        for (new, old) in zip(tf.trainable_variables(dst_gm), tf.trainable_variables(src_gm)):
            update_weight = [tf.assign(new, old)]
            sess.run(update_weight)
            temp_vals = sess.run([new, old])
        first = True
        for (new, old) in zip(tf.trainable_variables(dst_gs), tf.trainable_variables(src_gs)):
            temp_vals = sess.run([new, old])
            if new.shape != old.shape:
                # you need a transpose with perm # old = tf.reshape(old, tf.shape(new))
                # DO NOT USE RESHAPE
                # (made this mistake here and the results work but are quite terrifying)
                if (first):
                    first = False
                    old = tf.transpose(old, perm=[0, 3, 1, 2])
                else:
                    old = tf.transpose(old, perm=[0, 1, 3, 2])
            update_weight = [tf.assign(new, old)]
            sess.run(update_weight)
            
        # Also, assign the w_avg in the taki0112 network to the NVlabs Gs dlatent_avg
        new = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="G")
                if "dlatent_avg" in str(x)][0] # G.get_var("dlatent_avg")
        old = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
                if "avg" in str(x)][0]
        update_weight = [tf.assign(new, old)]
        sess.run(update_weight)
        vars = [new]
        vars_vals = gan.sess.run(vars)
        vars_vals = sess.run(vars)
        
        misc.save_pkl((G, D, Gs), "./"+nvlabs_stylegan_pkl_name)
    
if __name__ == "__main__":
    main()

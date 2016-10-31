

from inputproducer import InputProducer
from vgg16 import Vgg16
from gnet import GNet
from locnet import LocNet

from numpy.random import randint
from scipy.misc import imresize

from utils import img_with_bbox

import matplotlib.pylab as plt
import numpy as np 
import tensorflow as tf

import sys
import os
import time


tf.app.flags.DEFINE_integer('epoch', 10,
                          """Number of epoches for trainning""")
tf.app.flags.DEFINE_integer('n_samples_per_batch', 200,
                          """Number of samples per batch for trainning""")
tf.app.flags.DEFINE_integer('iter_max', 1349,
							"""Max iter times through imgs""")
tf.app.flags.DEFINE_bool('train', True,
						"""true for train, false for eval""")

tf.app.flags.DEFINE_string('model_name', 'model',
						"""true for train, false for eval""")


FLAGS = tf.app.flags.FLAGS


## Define varies path
DATA_ROOT = 'data/Dog1'
PRE_ROOT = os.path.join(DATA_ROOT, 'img_loc')
IMG_PATH = os.path.join(DATA_ROOT, 'img')
GT_PATH = os.path.join(DATA_ROOT, 'groundtruth_rect.txt')
VGG_WEIGHTS_PATH = 'vgg16_weights.npz'

TB_SUMMARY = os.path.join('tb_summary', FLAGS.model_name)
if not os.path.isdir('tb_summary'):
    os.mkdir('tb_summary')
if not os.path.isdir(TB_SUMMARY):
    os.mkdir(TB_SUMMARY)

CKPT_PATH = 'checkpoint'
if not os.path.isdir(CKPT_PATH):
    os.mkdir(CKPT_PATH)

model_name = FLAGS.model_name+'.ckpt'
CKPT_MODEL = os.path.join(CKPT_PATH, model_name)



def init_vgg():
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	vgg = Vgg16(VGG_WEIGHTS_PATH, sess)
	return sess, vgg



print('Reading the first image...')
## Instantiate inputProducer and retrive the first img
# with associated ground truth. 
inputProducer = InputProducer(IMG_PATH, GT_PATH)
img, gt, _  = next(inputProducer.gen_img)

# Register ops and tensors
sess, vgg = init_vgg()
gnet = GNet('gnet', vgg.conv5_3_norm)
locnet = LocNet('locnet', gnet.out_layer)
saver = tf.train.Saver()

if FLAGS.train:
    # Train nets 
    input_batch, gt_M_batch, loc_batch = gen_batches(img, gt, num_samples=FLAGS.n_samples_per_batch, batch_sz=10)
    vars_to_train = gnet.variables + locnet.variables + vgg.variables
    locnet_loss, gnet_loss = locnet.loss(), gnet.loss()
    total_loss = locnet_loss + 0.05*gnet_loss

    # registor summay tensors
    tf.scalar_summary('LocNet_loss', locnet_loss)
    tf.scalar_summary('GNet_loss', gnet_loss)
    tf.scalar_summary('Total_loss', total_loss)
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(TB_SUMMARY, sess.graph)

    # Backprop using SGD and updates variables
    global_step = tf.Variable(0, trainable=False)
    lr_exp = tf.train.exponential_decay(
            1e-5, # Initial learning rate 
            global_step, 
            1e4, # Decay steps 
            0.9, # Decay rate 
            name='sg_lr')
    optimizer = tf.train.GradientDescentOptimizer(lr_exp)
    train_op = optimizer.minimize(total_loss, var_list= vars_to_train, global_step=global_step)
    sess.run(tf.initialize_variables(vars_to_train + [global_step]))

    num_epoch = 20
    print('Start training the SGNets........ for %s epochs'%num_epoch)
    gs = 1
    for ep in range(num_epoch):
        step = 0
        print('Total batches in each epoch: ', len(input_batch))
        for img_distorted, gt_M, loc in zip(input_batch, gt_M_batch, loc_batch):
            gs += 1
            t = time.time()
            fd = {vgg.imgs: img_distorted, gnet.gt_M: gt_M, locnet.gt_loc: loc}
            pre_M_g, pre_loc, locl, gnetl, l, _, lr = sess.run([gnet.pre_M, locnet.logit, locnet_loss, gnet_loss, total_loss, train_op, lr_exp], feed_dict=fd)
            losses += [(l, locl, gnetl)]
            locs += [(pre_loc, loc)]
            
            # Write summaries to tensorboard.
            if step % 20 == 0:
                summary_img = tf.image_summary('pre_M', pre_M_g)
                summary, img_summary = sess.run([merged, summary_img], feed_dict=fd)
                
                writer.add_summary(summary, global_step=gs)
                writer.add_summary(img_summary, global_step=gs)
            
            # validate
            if step % 200 == 0:
                inputProducer = InputProducer(IMG_PATH, GT_PATH)
                for i in range(randint(1, 1349)):
                    img, gt_cur, s  = next(inputProducer.gen_img)
                    
                convas = np.zeros((max(img.shape), max(img.shape), 3))
                convas[:img.shape[0], :img.shape[1]] = img
                convas = imresize(convas,(224,224))

                fd = {vgg.imgs : [convas]}
                pre_loc = sess.run(locnet.logit, feed_dict=fd)
                pre_loc = pre_loc[0]
                
                # accuracy = 0 gives perfect score
                # value in accuracy is in unit of deviated pixel.
                acc = np.sum(gt_cur) - np.sum(pre_loc)
                acc_summary = sess.run(tf.scalar_summary('accuracy', acc))
                writer.add_summary(acc_summary, global_step=gs)
                print('Epoch: ', ep+1, 'Step: ', (ep+1)*step, 'Loss : %.2f'%l, \
                    'Speed: %.2f second/batch'%(time.time()-t), 'Lr: ', lr)
                print('LocNet loss: ', locl)
                print('GNet   loss: ', gnetl)
                print('predicted location: ',[int(i) for i in pre_loc])
                print('True      location: ', loc[-1])
            step += 1
        saver.save(sess, CKPT_MODEL)
else:
    # restor from saved model and do evaluations. 
    saver.restore(sess, CKPT_MODEL)
    for i in range(FLAGS.iter_max):
        i += 1
        t_enter = time.time()
        # Gnerates next frame infos
        img, gt_cur, s  = next(inputProducer.gen_img)
        convas = np.zeros((max(img.shape), max(img.shape), 3))
        convas[:img.shape[0], :img.shape[1]] = img
        convas = imresize(convas,(224,224))
        
        
        ## Perform Target localiation predicted by GNet
        # Get heat map predicted by GNet
        fd = {vgg.imgs : [convas]}
        pre_loc = sess.run(locnet.logit, feed_dict=fd)
        print(time.time() - t_enter, 'test time!')
        pre_loc = pre_loc[0]
        for k,p in enumerate(pre_loc):
            p = int(p)
            if p > 224:
                print('outbound warning!', p)
                pre_loc[k] = 224
                
        print('pre: ', [int(i) for i in pre_loc], 'actual: ', gt_cur)
        # Draw bbox on image. And print associated IoU score.
        img_bbox = img_with_bbox(img, pre_loc,c=1)
        #img_bbox = img_with_bbox(img_bbox, gt_cur, c=0)
        file_name = inputProducer.imgs_path_list[i-1].split('/')[-1]
        file_name = os.path.join(PRE_ROOT, file_name)
        plt.imsave(file_name, img_bbox)
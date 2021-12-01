import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
from modelac import UnetModel
from datetime import datetime
import os
import logging
import aug
import random
#import string
import numpy as np
import scipy.io
import time
import SimpleITK as sitk

from metric import dice_score2 
from metric import sensitivity 
try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir
import scipy.io as sio   
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
FLAGS = tf.flags.FLAGS
class_weight = [0.5,1,1,1]
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('D', 48, 'image size, default: 240')
tf.flags.DEFINE_integer('H', 272, 'image patch_size, default: 120')
tf.flags.DEFINE_integer('W', 208, 'image patch_size, default: 120')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_float('learning_rate', 1e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 56,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_integer('num_class', 8,
                        'number of class, default: 9')
tf.flags.DEFINE_string('DataPath', '/home/libo/DATA_REG/neo2dh/',
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('valPath', '/home/libo/DATA_REG/n2dtest/',
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('GraphPath', './pretrained/reg.pb',
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('model_path', '/home/libo/segment/largefov.npy',
                       'vgg16 path, default:')
tf.flags.DEFINE_string('valPath2', './val/',
                       'validata path, default:')
#tf.flags.DEFINE_string('Y', 'facade/tfrecords/Y.tfrecords',
                       #'Y tfrecords file for training, default:')
tf.flags.DEFINE_integer('NUM_ID',25885,
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('load_model',None,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
                        
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
OutputPath = '/home/libo/regbrain/rlst/'
def data_reader(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)
    if img_file.name.endswith('.png') and img_file.is_file():
      file_paths.append(img_file.path)
    if img_file.name.endswith('.mat') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
#    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths
def train():
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
#    checkpoints_dir2 = "checkpoints2/" + FLAGS.load_model.lstrip("checkpoints2/")
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/{}".format(current_time)
#    checkpoints_dir2 = "checkpoints2/{}".format(current_time)
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass
  '''
  grp2 = tf.Graph()
  with grp2.as_default():
    reader = Reader2(FLAGS.DataPath, name='U',
        image_size=FLAGS.image_size, batch_size=FLAGS.batch_size)
    name,x,y= reader.feed()
  '''
  print('~~~~~~~~~~~~~~~~~~~~~!!!!')
  graph = tf.Graph()
  with graph.as_default():
    Unet_Model = UnetModel(
        DataPath=FLAGS.DataPath,
        GraphPath = FLAGS.GraphPath,
        model_path = FLAGS.model_path,
        batch_size=FLAGS.batch_size,
        D=FLAGS.D,
        H=FLAGS.H,
        W=FLAGS.W,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf,
        num_class = FLAGS.num_class
    )
    loss_cc,s_t,l_st,seg,loss_seg= Unet_Model.model()
#    loss_all = Unet_Model.model()
    optimizers = Unet_Model.optimize(loss_cc)
    optimizers2 = Unet_Model.optimize2(loss_seg)
#    optimizers1 = Unet_Model.optimize(loss1)
#    reader = Reader2(FLAGS.DataPath, name='U',
#        image_size=FLAGS.image_size, batch_size=FLAGS.batch_size)
#    name,x,y= reader.feed()
#    summary_op = tf.summary.merge_all()
#    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    
    src = tf.placeholder(tf.float32,
        shape=[1,48, 272, 208, 1])
    tgt = tf.placeholder(tf.float32,
        shape=[1,48, 272, 208, 1])
    label = tf.placeholder(tf.float32,
        shape=[1,48, 272, 208, 8])
    label_t = tf.placeholder(tf.float32,
        shape=[1,48, 272, 208, 8]) 
        
    with tf.gfile.FastGFile(FLAGS.GraphPath, 'rb') as f_file:
        graph_f1 = tf.GraphDef()
        graph_f1.ParseFromString(f_file.read())
#        tf.import_graph_def(graph_f1, name='')
        y,flow,l = tf.import_graph_def(graph_f1,
                    input_map={'src': src,'tgt': tgt,'label': label,'label_t': label_t},
                    return_elements=['y:0','flow:0','l:0'],
                    name="reg")
    
    saver = tf.train.Saver()
  tf_config = tf.ConfigProto() 
#  tf_config = tf.ConfigProto(allow_soft_placement=True) 
#  tf_config.gpu_options.allow_growth = True 
#  tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
  #with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=True)) as sess:
  with tf.Session(graph=graph,config = tf_config) as sess:
#  sess = tf.Session(graph=graph)
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      print(meta_graph_path)
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0
    
#    while True:
#    sess2 = tf.Session(graph=grp2)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#      coord.clear_stop()
    try:
  #      fake_Y_pool = ImagePool(FLAGS.pool_size)
  #      fake_X_pool = ImagePool(FLAGS.pool_size)
  #      fake_Z_pool = ImagePool(FLAGS.pool_size)
      loss_3000 = 0
      loss_1 = 0
      loss_2 = 0
      loss_echo = 0
      echo = 120
      cst = 40
      count = 0
      list = os.listdir(FLAGS.DataPath) 
      while not coord.should_stop():
  #    while True:
        
#        print('~!!!!!!!!!!!!!')
#        a = time.time()
#        x_val,y_val = sess.run([x,y])
        
        file_paths_X = data_reader(FLAGS.DataPath)
        val_files = data_reader(FLAGS.valPath)
        num_hard = len(file_paths_X)
        i_hard = 0
        while(1):
          if not coord.should_stop():
            file_path = file_paths_X[i_hard]
#            print(file_path)
            data=sio.loadmat(file_path) 
            d_t = np.float32(data['dt'])
            d_t = d_t.astype(np.float32)
            
            d_s = np.float32(data['ds'])
            d_s = np.squeeze(d_s)
            
            l_s = np.float32(data['ls'])
            l_s = np.squeeze(l_s)
            
            d_s,l_s = aug.zm(d_s,l_s)
            
            l_t = np.float32(data['lt'])
            l_t = np.squeeze(l_t)
            
            l_s_h = aug.L2HOT(l_s,8)
            l_t_h = aug.L2HOT(l_t,8)
            
            l_s = l_s[np.newaxis,:,:,:,np.newaxis]
            d_s = d_s[np.newaxis,:,:,:,np.newaxis]
            d_t = d_t[np.newaxis,:,:,:,np.newaxis]

            l_s_h = l_s_h[np.newaxis,:,:,:,:]
            l_t_h = l_t_h[np.newaxis,:,:,:,:]
            
            if step >= 3000:#train unet
              
              if step % 2 == 0:
                
                pred_val,l_st_val= (sess.run([y,l], feed_dict={src: d_s,tgt: d_t,label: l_s_h,label_t: l_t_h}))
                
                _, Dice_val = (
                    sess.run([optimizers2, loss_seg],feed_dict={Unet_Model.dt: pred_val,Unet_Model.lt: l_st_val}))
                loss_3000 = loss_3000 + Dice_val
                loss_1 = loss_1 + Dice_val
                loss_2 = loss_2 + Dice_val
                loss_echo = loss_echo + Dice_val
              else:
                pred_val,l_st_val= (sess.run([s_t,l_st], feed_dict={Unet_Model.ds: d_s,Unet_Model.dt: d_t,Unet_Model.ls: l_s_h,Unet_Model.lt: l_t_h}))
                d_s = np.squeeze(d_s)
                d_t = np.squeeze(d_t)
                l_s_h = np.squeeze(l_s_h)
                l_t_h = np.squeeze(l_t_h)
                _, Dice_val = (
                    sess.run([optimizers2, loss_seg],feed_dict={Unet_Model.dt: pred_val,Unet_Model.lt: l_st_val}))
                loss_3000 = loss_3000 + Dice_val
                loss_1 = loss_1 + Dice_val
                loss_2 = loss_2 + Dice_val
                loss_echo = loss_echo + Dice_val
              
              i_hard = i_hard + 1
              if i_hard == num_hard:
                i_hard = 0
                file_paths_X = data_reader(FLAGS.DataPath)
              step += 1
              
            elif step < 3000:
            
              _,loss_cc_val,pred_val,l_st_val= (sess.run([optimizers,loss_cc,s_t,l_st], feed_dict={Unet_Model.ds: d_s,Unet_Model.dt: d_t,Unet_Model.ls: l_s_h,Unet_Model.lt: l_t_h}))
            
              loss_3000 = loss_3000 + loss_cc_val
              loss_1 = loss_1 + loss_cc_val
              loss_2 = loss_2 + loss_cc_val
              loss_echo = loss_echo + loss_cc_val
              
              i_hard = i_hard + 1
              if i_hard == num_hard:
                i_hard = 0
                file_paths_X = data_reader(FLAGS.DataPath)
              step += 1

         
          else:
            break
          if count % cst == 0:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logger.info('-----------Step %d:-------------' % step)
            logger.info('  loss_1  : {}'.format(loss_1))
            logger.info('  loss_2  : {}'.format(loss_2))
            loss_1 = 0
            loss_2 = 0
          if count % echo == 0:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logger.info('-----------Step %d:-------------' % step)
            logger.info('!!!!!!!!!!!!loss_3000!!!!!!!!!!!!!!!!!   : {}'.format(loss_3000)) 
            logger.info('!!!!!!!!!!!!LOSS_ECHO!!!!!!!!!!!!!!!!!   : {}'.format(loss_echo)) 
            loss_3000 = 0
            loss_echo = 0

            
          count += 1            

           
    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      logging.info("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()

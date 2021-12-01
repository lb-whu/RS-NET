"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""
#coding:utf-8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import SimpleITK as sitk
import aug
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'pretrained/trans_3001.pb', 'model path (.pb)')
tf.flags.DEFINE_string('model1', 'pretrained/feat.pb', 'model path (.pb)')
tf.flags.DEFINE_string('model2', 'pretrained/new2.pb', 'model path (.pb)')

#tf.flags.DEFINE_string('modelcy', 'pretrained/cycada_gen.pb', 'model path (.pb)')

tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 400')
tf.flags.DEFINE_string('valPath', './val/',
                       'validata path, default:')
tf.flags.DEFINE_string('DataPath', '/home/libo/DATA_REG/neo2dh/',
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('MUPath', './Munit/',
                       'validata path, default:')
tf.flags.DEFINE_string('rsltPath', './val_result/',
                       'validata path, default:')
tf.flags.DEFINE_string('GraphPath', '/home/libo/regbrain/pretrained/trans_3001.pb',
                       'X tfrecords file for training, default:')
def inference():
  print('~~~!!!!!!!!!!')
  graph = tf.Graph()
  list = os.listdir(FLAGS.DataPath) 
#  list2 = os.listdir(FLAGS.MUPath) 
  with graph.as_default():
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
#    with tf.gfile.FastGFile(FLAGS.modelcy, 'rb') as model_filecy:
#        graph_defcy = tf.GraphDef()
#        graph_defcy.ParseFromString(model_filecy.read())
            
  sess = tf.Session(graph=graph)
  count = 0
  for i in range(len(list)):
#    file_names = list[i].split("_")
#    if file_names[0] == '3' and file_names[2] == 'i1' and file_names[4] == '1.mat':
#    if list[i] == '10_19.mat':
      path = os.path.join(FLAGS.DataPath,list[i])
      data = scipy.io.loadmat(path)
      d_t = np.float32(data['dt'])
      d_t = d_t.astype(np.float32)
      
      d_s = np.float32(data['ds'])
      d_s = np.squeeze(d_s)
      
      l_s = np.float32(data['ls'])
      l_s = np.squeeze(l_s)
      
      d_s,l_s = aug.zm(d_s,l_s)
      
      l_t = np.float32(data['lt'])
      l_t = np.squeeze(l_t)
      
      m_s = np.float32(data['ms'])
      m_s = np.squeeze(m_s)
      
      var_s = np.float32(data['vars'])
      var_s = np.squeeze(var_s)
      
      l_s_h = aug.L2HOT(l_s,8)
      l_t_h = aug.L2HOT(l_t,8)
      
      l_s = l_s[np.newaxis,:,:,:,np.newaxis]
      d_s = d_s[np.newaxis,:,:,:,np.newaxis]
      d_t = d_t[np.newaxis,:,:,:,np.newaxis]
      m_s = m_s[np.newaxis,:,:,:,np.newaxis]
      var_s = var_s[np.newaxis,:,:,:,np.newaxis]
      l_s_h = l_s_h[np.newaxis,:,:,:,:]
      l_t_h = l_t_h[np.newaxis,:,:,:,:]
            
            
      pred_val,l_st_val= (sess.run([y,l], feed_dict={src: d_s,tgt: d_t,label: l_s_h,label_t: l_t_h}))
                
      pred_val = np.squeeze(pred_val)
      l_st_val = np.squeeze(l_st_val)
      
      indst = np.argmax(l_st_val,axis = 3)
      indst = indst.astype(np.int32)
      
      d_t = np.squeeze(d_t)
      l_t = np.squeeze(l_t)
      
      scipy.io.savemat(os.path.join(FLAGS.valPath,path.split('/')[-1]),{'ds':pred_val,'ls':indst,'dt':d_t,'lt':l_t},do_compression = True) 
      
      resultImage = sitk.GetImageFromArray(pred_val)
      sitk.WriteImage(resultImage, FLAGS.valPath +'pred_val_'+path.split('/')[-1].split('.')[0]+ '.nii')
      resultImage = sitk.GetImageFromArray(indst)
      sitk.WriteImage(resultImage, FLAGS.valPath+'l_st_'+path.split('/')[-1].split('.')[0]+ '.nii')
      
      count = count + 1
  count = 0



def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()

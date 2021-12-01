"""
Networks for voxelwarp model
"""

# third party
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.models import Model
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras
import numpy as np

# local
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
import losses


class Unet2:
  def __init__(self, name, is_training, ngf=64, norm='instance', num_class = 9,model_path='largefov.npy'):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.num_class = num_class
    self.model_path = model_path
  def __call__(self, tgt, enc_nf_seg=[32,64,128,256],  seg_dec_nf=[128,64,32,8]):
  
  #    src = Input(shape=vol_size + (1,))
  #    tgt = Input(shape=vol_size + (1,))
    with tf.variable_scope(self.name):
#      with tf.device("/device:GPU:0"):
#        x_in = concatenate([src, tgt])
        
        x0 = self.myConv(tgt, enc_nf_seg[0], 1)  # 24x136x104
        x0_ = self.myConv(x0, enc_nf_seg[0], 2)  # 24x136x104
        x1 = self.myConv(x0_, enc_nf_seg[1], 1)  # 12x68x52
        x1_ = self.myConv(x1, enc_nf_seg[1], 2)  # 12x68x52
        x2 = self.myConv(x1_, enc_nf_seg[2], 1)  # 6x34x26
        x2_ = self.myConv(x2, enc_nf_seg[2], 2)  # 6x34x26
        x3 = self.myConv(x2_, enc_nf_seg[3], 1)  # 3x17x13
        x3_ = self.myConv(x3, enc_nf_seg[3], 1)  # 3x17x13

        
        sx = self.myConv(x3_, seg_dec_nf[0])
        sx = self.myConv(sx, seg_dec_nf[0])
        sx = UpSampling3D()(sx)
        sx = concatenate([sx, x2])
        
        sx = self.myConv(sx, seg_dec_nf[1])
        sx = self.myConv(sx, seg_dec_nf[1])
        sx = UpSampling3D()(sx)
        sx = concatenate([sx, x1])
        
        sx = self.myConv(sx, seg_dec_nf[2])
        sx = self.myConv(sx, seg_dec_nf[2])
        sx = UpSampling3D()(sx)
        sx = concatenate([sx, x0])
        
        sx = self.myConv(sx, seg_dec_nf[3])
        seg = Activation('sigmoid')(sx)
      
      
      
  #    model = Model(inputs=[src, tgt], outputs=[y, flow])
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    return seg


  def myConv(self,x_in, nf, strides=1):
    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out
  def sample(self, src,tgt):
    y, flow,l = self.__call__(src,tgt,label, enc_nf=[16,32,32], dec_nf=[32,32,32,8,8,3])
#    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return y, flow,l
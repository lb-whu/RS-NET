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


class Unet1:
  def __init__(self, name, is_training, ngf=64, norm='instance', num_class = 9,model_path='largefov.npy'):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.num_class = num_class
    self.model_path = model_path
  def __call__(self, src,tgt,label,label_t, enc_nf_reg=[16,32,32], dec_nf=[32,32,32,8,8,3]):
  
  #    src = Input(shape=vol_size + (1,))
  #    tgt = Input(shape=vol_size + (1,))
    with tf.variable_scope(self.name):
#      with tf.device("/device:GPU:0"):
        x_in = concatenate([src, tgt])
        x0_ = self.myConv(x_in, enc_nf_reg[0], 2)  # 24x136x104
        x1_ = self.myConv(x0_, enc_nf_reg[1], 2)  # 12x68x52
        x2_ = self.myConv(x1_, enc_nf_reg[2], 2)  # 6x34x26

    
        x = self.myConv(x2_, dec_nf[0])
        x = UpSampling3D()(x)
        x = concatenate([x, x1_])
        
        x = self.myConv(x, dec_nf[1])
        x = UpSampling3D()(x)
        x = concatenate([x, x0_])

        x = self.myConv(x, dec_nf[2])
        x = self.myConv(x, dec_nf[3])
    
        x = UpSampling3D()(x)
        x = concatenate([x, x_in])
        x = self.myConv(x, dec_nf[4])
        x = self.myConv(x, dec_nf[5])
    
        flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                      kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)
#      with tf.device("/device:GPU:1"):
        y = Dense3DSpatialTransformer()([src, flow])
        llist = []
    
        for k in range(0,8):
          temp = tf.expand_dims(label[:,:,:,:,k],4)
          llist.append(Dense3DSpatialTransformer()([temp, flow]))
        l = tf.stack(llist,axis = 4)
#        m_y = Dense3DSpatialTransformer()([m_src, flow])
#        var_y = Dense3DSpatialTransformer()([var_src, flow])
        ll = tf.argmax(l,axis = 4)
        ll = tf.squeeze(ll,[4])
        oh_l = tf.one_hot(ll,8,name = 'l')
        yt = self.intraTrans(y,oh_l,tgt,label_t,8)
  #    model = Model(inputs=[src, tgt], outputs=[y, flow])
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    return yt, flow ,oh_l



 
  def intraTrans(self,ds,ls,dt,lt,l_num):
#    print('ds',ds.shape)
#    print('ls',ls.shape)
#    print('dt',dt.shape)
#    print('lt',lt.shape)
#    ds = tf.squeeze(ds,[4])
#    dt = tf.squeeze(dt,[4])
    snum_img = tf.reduce_sum(ls,axis=(0,1,2,3))
    aver_cls = tf.reduce_sum(ds *ls,axis = (0,1,2,3))/snum_img
    m_img= tf.reduce_sum(aver_cls*ls,axis = (4))
    m_img = tf.expand_dims(m_img,4)
    var = (ds-m_img)**2
#    var = tf.expand_dims(var,4) 
    var_img = tf.sqrt(tf.reduce_sum(tf.reduce_sum(var*ls,axis = (0,1,2,3))*ls/snum_img,axis = (4)))
    var_img = tf.expand_dims(var_img,4) 
    
    tnum_img = tf.reduce_sum(lt,axis=(0,1,2,3))
    tm_img= (tf.reduce_sum(dt *lt,axis = (0,1,2,3)))/tnum_img
    ttm_img = tf.reduce_sum(tm_img*lt,axis = (4))
    tsm_img = tf.reduce_sum(tm_img*ls,axis = (4))
    ttm_img = tf.expand_dims(ttm_img,4)
    tsm_img = tf.expand_dims(tsm_img,4)
    tvar = (dt-ttm_img)**2
#    tvar = tf.expand_dims(tvar,4) 
    tvar_img = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tvar*lt,axis = (0,1,2,3))*ls/tnum_img,axis = (4)))
    tvar_img = tf.expand_dims(tvar_img,4)

    t_img = tf.add(tvar_img*(ds-m_img)/(var_img+1e-8), tsm_img,name='y')

    return t_img

  def myConv(self,x_in, nf, strides=1):
    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out
  def sample(self, src,tgt,label,label_t):
    y, flow,l = self.__call__(src,tgt,label,label_t, enc_nf_reg=[16,32,32], dec_nf=[32,32,32,8,8,3])
#    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return y, flow,l
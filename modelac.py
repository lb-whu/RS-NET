import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ops
import random
import string 
import numpy as np
from net1 import Unet1
#from net2 import Unet2
from ACnet import Unet2
REAL_LABEL = 0.9
class UnetModel:
  def __init__(self,
               DataPath='',
               GraphPath='',
               model_path='',
               batch_size=1,
               D=112,
               H=112,
               W=112,
               use_lsgan=True,
               norm='instance',
               learning_rate=0.2e-3,
               beta1=0.5,
               ngf=64,
               num_class = 4,
              ):
    """
    Args:
      X_train_file: string, X tfrecords file for training
      Y_train_file: string Y tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      lambda1: integer, weight for forward cycle loss (X->Y->X)
      lambda2: integer, weight for backward cycle loss (Y->X->Y)
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
    """
    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.D = D
    self.W = W
    self.H = H
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.DataPath = DataPath
    self.GraphPath = GraphPath
    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.Unet1 = Unet1('Unet1', self.is_training, ngf=ngf, norm=norm, num_class=num_class,model_path=model_path)
    self.Unet2 = Unet2('Unet2', self.is_training, ngf=40, norm=norm, num_class=num_class,model_path=model_path)
    
#    self.feature = tf.placeholder(tf.float32,
#        shape=[batch_size,6, int(patch_size/8), int(patch_size/8), 4*ngf])
    
#    self.c1 = tf.placeholder(tf.float32,
#        shape=[batch_size,int(patch_size/1), int(patch_size/1), ngf])
#    self.filename = tf.placeholder(tf.string)
#    self.c2 = tf.placeholder(tf.float32,
#        shape=[batch_size,int(patch_size/2), int(patch_size/2), 2*ngf])
#    self.c3 = tf.placeholder(tf.float32,
#        shape=[batch_size,int(patch_size/4), int(patch_size/4), 4*ngf])
    
    
    self.ds = tf.placeholder(tf.float32,
        shape=[batch_size,D, H, W, 1])
    self.ls = tf.placeholder(tf.float32,
        shape=[batch_size,D, H, W, 8])
    self.lt = tf.placeholder(tf.float32,
        shape=[batch_size,D, H, W, 8])
    self.dt = tf.placeholder(tf.float32,
        shape=[batch_size,D, H, W, 1])
    self.m_s = tf.placeholder(tf.float32,
        shape=[batch_size,D, H, W, 1])
    self.var_s = tf.placeholder(tf.float32,
        shape=[batch_size,D, H, W, 1])
#    self.f = tf.placeholder(tf.float32,
#        shape=[batch_size,patch_size, patch_size, patch_size,2])
  def model(self):  
    '''
    reader = Reader2(self.DataPath, name='U',
        image_size=self.image_size, batch_size=self.batch_size)
    name,x,y= reader.feed()
    '''
 #   Y_reader = Reader(self.Y_train_file, name='Y',
     #   image_size=self.image_size, batch_size=self.batch_size)
#    Z_reader = Reader(self.Z_train_file, name='Z',
#        image_size=self.image_size, batch_size=self.batch_size)    
#    name,x,y= reader.feed()
    
#    y = np.load(self.DataPath_img+self.filename)
#    x = np.load(self.DataPath_gt+self.filename)
    print('UNET-------------------')
#    print(y2)
#    _,_,_,feature = self.Unet.downsample(self.y,None)
    s_t,flow,l_t = self.Unet1(src=self.ds,tgt=self.dt,label=self.ls,label_t=self.lt)
    seg = self.Unet2(input=self.dt)
    loss_cc = self.loss_MSE(I=s_t, J=self.dt)
#    loss_g = self.gradientLoss(y_pred=s_t, y_true=self.dt)
    
    loss_seg = self.loss_SEG(self.lt,seg)
#    print('losslosslossloss~~~',loss.shape)
#    loss_all = loss1*0.5 + loss2*0.5
#    tf.add_to_collection(name='loss', value=loss)
#    loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

        
    return loss_cc,s_t,l_t,seg,loss_seg#,x,y#H_loss,D_Z_loss,fake_z

  def optimize(self, loss_all):#H_loss,D_Z_loss
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = 1e-5
      end_learning_rate = 1e-6
      start_decay_step = 5000
      decay_steps = 40000
      beta1 = self.beta1
      learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                            decay_steps, end_learning_rate,
                                            power=1.0),
                  starter_learning_rate
          )

      )

      optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
      gradients = optimizer.compute_gradients(loss,var_list=variables,colocate_gradients_with_ops=True)

      learning_step = (
              optimizer.apply_gradients(gradients)
              )

      #learning_step = (
      #    tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
      #            .minimize(loss, global_step=global_step, var_list=variables)
      #)
      return learning_step

    Unet_optimizer = make_optimizer(loss_all, self.Unet1.variables, name='Adam_Unet')

    with tf.control_dependencies([Unet_optimizer]):#H_optimizer,D_Z_optimizer
      return tf.no_op(name='optimizers')

  def optimize2(self, loss_seg):#H_loss,D_Z_loss
    def make_optimizer2(loss, variables, name='Adam2'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = 1e-3
      end_learning_rate = 1e-4
      start_decay_step = 3000
      decay_steps = 40000
      beta1 = self.beta1
      learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                            decay_steps, end_learning_rate,
                                            power=1.0),
                  starter_learning_rate
          )

      )

      optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
      gradients = optimizer.compute_gradients(loss,var_list=variables,colocate_gradients_with_ops=True)

      learning_step = (
              optimizer.apply_gradients(gradients)
              )

      #learning_step = (
      #    tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
      #            .minimize(loss, global_step=global_step, var_list=variables)
      #)
      return learning_step

    Unet_optimizer2 = make_optimizer2(loss_seg, self.Unet2.variables, name='Adam_seg')

    with tf.control_dependencies([Unet_optimizer2]):#H_optimizer,D_Z_optimizer
      return tf.no_op(name='optimizer2')
      
  def gradientLoss(self,y_true, y_pred, penalty='l2'):
    dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
    dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    
#    if (penalty == 'l2'):
#        dy = dy * dy
#        dx = dx * dx
#        dz = dz * dz
    d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
    return d/2.0    
    
  def loss(self,I, J,win=[9, 9, 9]):
    I2 = I*I
    J2 = J*J
    IJ = I*J

    filt = tf.ones([win[0], win[1], win[2], 1, 1])

    I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
    J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
    I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
    J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
    IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

    win_size = win[0]*win[1]*win[2]
    u_I = I_sum/win_size
    u_J = J_sum/win_size

    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

    cc = cross*cross / (I_var*J_var+1e-5)

    # if(voxel_weights is not None):
    #	cc = cc * voxel_weights

    return -1.0*tf.reduce_mean(cc)
  def loss_MSE(self,I, J):
    loss = tf.reduce_mean(tf.square(I - J))
    return loss
    
  def loss_SEG(self, x,layer_seg1):
    e=1e-8
    loss_type = 'jaccard'
    inse1 = tf.reduce_sum(layer_seg1 * x, axis=(1,2,3))
    if loss_type == 'jaccard':
        l1 = tf.reduce_sum(layer_seg1 * layer_seg1, axis=(1,2,3))
        r1 = tf.reduce_sum(x * x, axis=(1,2,3))
    else:
        l1 = tf.reduce_sum(layer_seg1, axis=(1,2,3))
        r1 = tf.reduce_sum(x, axis=(1,2,3))

    dice1 = (2. * inse1 + e) / (l1 + r1 + e)
    dice1 = tf.reduce_mean(dice1)
    
    return 1 -dice1 
    
  def unetLoss2(self,Unet, x,y1,y2):
    """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    layer_seg1,layer_seg2 = Unet(y1,y2)
    print('xxxxxxx:::',x.shape)
    print('layer_seg:::',layer_seg2.shape)
    x = tf.cast(x, dtype=tf.float32)
#    x = tf.squeeze(x,[4])
#    print('x::::::',x.shape)
#    label = tf.cast(tf.one_hot(tf.cast(x,tf.int32),4,1,0),tf.float32)
#    weight = tf.constant([0.5,1,1,1])
#    print('weight::::::',weight.dtype)
#    loss = tf.nn.weighted_cross_entropy_with_logits(logits=pred,targets=label, pos_weight = weight)
            
            
    loss_seg = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x, logits=layer_seg2, name='loss'))
#    loss_gan = tf.reduce_mean(tf.pow(layer_gan-y,2))
    return loss_seg
    

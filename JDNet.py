#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras as ks
import tensorflow as tf
from keras import Model
from keras.layers import Input, add, Concatenate, Dense, Dropout, Flatten, Conv2D, AveragePooling2D,Conv2DTranspose
from keras.layers import GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ReLU, ZeroPadding2D
from keras.layers import Multiply, Reshape, TimeDistributed


# In[25]:



#激發
def _after_conv(in_tensor):
    norm = BatchNormalization()(in_tensor)
    return ReLU()(norm)

def conv3(in_tensor, filters, s=1):
    conv = Conv2D(filters, kernel_size=3, strides=s, padding='same')(in_tensor)
    return _after_conv(conv)

def resnet_block_wo_bottleneck(in_tensor, filters, downsample=False):
    conv = in_tensor
    conv = conv3(conv,filters)
    conv = Conv2D(filters, kernel_size=3, strides=1, padding='same')(conv)
    result = add([conv, in_tensor])
    return result

def convx_wo_bottleneck(in_tensor, filters, n_times, downsample_1=False):
    res = in_tensor
    res = resnet_block_wo_bottleneck(res, filters, downsample_1)
    return res

def transpose(in_tensor, filters):
    conv_u = Conv2DTranspose(filters, kernel_size=(3,3), strides=2, padding='same')(in_tensor)
    return conv_u

def upsize(in_tensor, filters, times):
    conv_u = in_tensor
    for i in range(times):
        conv_u = transpose(conv_u, filters)
        
    return conv_u
    
def _pre_res_blocks(in_tensor):
    conv = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(in_tensor)
    conv = _after_conv(conv)
    return conv
   
def _post_res_blocks(in_tensor):
    conv = Conv2D(3, kernel_size=1, strides=1, padding='same')(in_tensor)
    # pool = layers.MaxPool2D(3, 2, padding='same')(conv)
    #conv_transpose = Conv2DTranspose(3, (1,1), strides=(2,2))(pool)#將(28,28,3)轉成(56,56,3)
    return conv


def _first_net(input_shape=(224,224,3),
               loss='MSE',
               opt='sgd',
               filters=[64, 128, 256, 512],
               times=[2, 2, 2, 2],#n_convx=times
               convx_fn=convx_wo_bottleneck):
    in_layer = Input(input_shape)
    pre_pocess = _pre_res_blocks(in_layer)#channel transformation
    conv1x = conv3(pre_pocess, filters[0], 2)#第一次將長寬/2,112
    convr1x = convx_fn(conv1x, filters[0], 0, False)#第一次結果 要給下一層及還原大小(112 to 224)
    up_conv1x = upsize(convr1x, filters[0],1)
    
    conv2x = conv3(convr1x, filters[1], 2)#第二次將長寬/2,64
    convr2x = convx_fn(conv2x, filters[1], 0, False)#第二次結果 要給下一層及還原大小(64 to 224)
    up_conv2x = upsize(convr2x, filters[0],2)
    
    conv3x = conv3(convr2x, filters[2], 2)#第三次將長寬/2,32
    convr3x = convx_fn(conv3x, filters[2], 0, False)#第三次結果 要給下一層及還原大小(32 to 224)
    up_conv3x = upsize(convr3x, filters[0],3)            
    
    conv4x = conv3(convr3x, filters[3], 2)#第四次將長寬/2,16
    convr4x = convx_fn(conv4x, filters[3], 0, False)#第四次結果 要給下一層及還原大小(16 to 224)
    up_conv4x = upsize(convr4x, filters[0],4) 

    total_conv = Concatenate()([pre_pocess, up_conv1x, up_conv2x, up_conv3x, up_conv4x])
    post_conv = _post_res_blocks(total_conv)
    '''
    
    conv3x = convx_fn(conv2x, convx[1], n_convx[1], False)
    conv4x = convx_fn(conv3x, convx[2], n_convx[2], False)
    conv5x = convx_fn(conv4x, convx[3], n_convx[3], False)
    conv6x = convx_fn(conv5x, 16, 2, True)
    
    conv7u = upsize(conv6x,downsampled,convx[0])
    conv8u = upsize(conv7u,conv1x,convx[1])
    conv9u = upsize(conv8u,in_layer,convx[2])
    '''
    '''upsize = Conv2DTranspose(56, kernel_size=(3,3), strides=2, padding='same')(conv6x)
    upsize = Concatenate()([downsampled, upsize])
    upsize = Conv2D(56, kernel_size=(3,3), strides=1, padding='same')(upsize)
    three_dim = tf.keras.layers.Reshape((56, 56, 56, 1))(upsize)
    time_1 = tf.keras.layers.TimeDistributed(Conv2D(1, kernel_size=(3,3), strides=(2,2), padding='same'))(three_dim)
    #raining = _post_res_blocks(conv5x)
    clean = tf.keras.layers.Reshape((28,28, 56))(time_1)'''
    
    
    '''upsize_2 = Conv2DTranspose(128, 3, 2, padding='same')(conv7u)
    upsize_2 = Concatenate()([downsampled, upsize_2])
    upsize_2 = tf.keras.layers.Reshape((upsize_2.shape[3], 56, 56, 1))(upsize_2)
    time_2 = tf.keras.layers.TimeDistributed(Conv2D(1,kernel_size=(7,7), strides=1, padding='same'))(upsize_2)
    clean_2 = tf.keras.layers.Reshape((56, 56, upsize_2.shape[1]))(time_2)'''
    preds = _post_res_blocks(post_conv)
    model = Model(in_layer, preds)
    
    model.compile(loss=loss, optimizer=opt)
    return model


def first_net(input_shape=(224,224,3), loss='MSE', opt='adam'):
    return _first_net(input_shape, loss, opt)

'''
if __name__ == '__main__':
    model = first_net()
    print(model.summary())
'''


# In[ ]:





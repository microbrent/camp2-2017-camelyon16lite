import math
import numpy as np
import os
import random
import tensorflow as tf
from PIL import Image 

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

sess = tf.InteractiveSession()

# x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# stride of 1 and zero-padded, so the output size is the same as the input size
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Layer 1
#   convolution: 32 features for each 5x5 patch
#   max pooling:

W_conv1 = weight_variable([5, 5, 3, 32]) # patch size, num input channels, num output channels
b_conv1 = bias_variable([32]) # one component for each output channel

#x_image = tf.reshape(x, [-1,28,28,1]) # ?, image width, image height, num color channels

# convolve x_image with weight tensor, add bias, apply ReLU, and finally max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 28x28x32
h_pool1 = max_pool_2x2(h_conv1) #14x14x32

# Layer 2
#   convolution: 64 features for each 5x5 patch
#   max pooling:

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 14x14x64
h_pool2 = max_pool_2x2(h_conv2) # 7x7x64


# Densely connected layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Dropout (reduce overfitting -> dropout before readout)
keep_prob = tf.placeholder(tf.float32) # create placeholder so we can turn off during testing
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Readout Layer
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

##
## Train and evaluate the model
##

cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv) )
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run( tf.global_variables_initializer() )

def tile_image_as_ndarray(filename):
  tile = Image.open(filename)
  cropped_tile = tile.crop([0, 0, 28,28]) #HACK: f'ed up when creating tiles; some are 29, 30, 31 px high
  return np.array(cropped_tile)

def load_tiles(dir):
  return list(map(lambda basename: tile_image_as_ndarray(dir + basename), os.listdir(dir)))

def create_target(bla, num):
  y_ = np.array(bla * num)
  y_ = np.reshape(y_, [-1, len(bla)])
  return y_

def generate_training_test_data( res ):

  normal_tiles_dir = "D:\\camelyon16\\" + str(res)+ "\\tiles_28x28_normal\\"
  normal_tiles = load_tiles(normal_tiles_dir)
  y_normal = create_target([0, 1], len(normal_tiles))

  tumor_neg_tiles_dir = "D:\\camelyon16\\" + str(res)+ "\\tiles_28x28_tumor_neg\\"
  tumor_neg_tiles = load_tiles(tumor_neg_tiles_dir)
  y_tumor_neg = create_target([0, 1], len(tumor_neg_tiles))

  tumor_pos_tiles_dir = "D:\\camelyon16\\" + str(res)+ "\\tiles_28x28_tumor_pos\\"
  tumor_pos_tiles = load_tiles(tumor_pos_tiles_dir)
  y_tumor_pos = create_target([1, 0], len(tumor_pos_tiles))

  num_tumor_pos_tiles = len(tumor_pos_tiles)
  random.shuffle(tumor_neg_tiles)
  random.shuffle(normal_tiles)

  all_images = np.concatenate((normal_tiles[0:num_tumor_pos_tiles], tumor_neg_tiles[0:num_tumor_pos_tiles], tumor_pos_tiles))
  all_targets = np.concatenate((y_normal[0:num_tumor_pos_tiles], y_tumor_neg[0:num_tumor_pos_tiles], y_tumor_pos))
  all_data = list(zip(all_images, all_targets))
  random.shuffle(all_data)

  N = len(all_data)
  n = math.floor(0.5*len(all_data))
  training_data = all_data[0:n]
  testing_data = all_data[n+1:N]
  return training_data, testing_data

training_data, testing_data = generate_training_test_data(512)

# train
batch_size = 50
for i in range(0, len(training_data), batch_size):
   batch = training_data[i:i+batch_size]
   batch0, batch1 = zip(*batch)
   images = np.array(batch0)
   targets = np.reshape(np.array(batch1), [-1,2])
   if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x_image:images, y_:targets, keep_prob:1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
   train_step.run(feed_dict={x_image:images, y_:targets, keep_prob:0.5})

# test 
batch0, batch1 = zip(*testing_data)
test_images = np.array(batch0)
test_targets = np.reshape(np.array(batch1), [-1,2])
print("test accuracy %g"%accuracy.eval(feed_dict={x_image: test_images, y_:test_targets, keep_prob:1.0}))

# make sure to save results, especially if it takes 40 min to generat them

import tensorflow as tf
import time
from datetime import timedelta
import math
import numpy as np
import logging
import matplotlib.pyplot as plt

def create_weights(shape):
	return tf.Variable(tf.random_uniform(shape, -0.05,0.05))

def create_biases(size):
	return tf.Variable(tf.zeros(shape=[size]))

def create_convolutional_layer(input,num_input_channels,conv_filter_size,num_filters):

	#Defining the weights that will be trained using create_weights function.
	weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
	#Defining biases using the create_biases function. These are also trained.
	biases = create_biases(num_filters)

	## Creating the convolutional layer
	layer = tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='VALID') #SAME
	layer += biases
	layer = tf.nn.relu(layer)
	layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding = 'SAME')

	return layer


def create_flatten_layer(layer):
	#The shape of the layer will be [batch_size img_size img_size num_channels]
	layer_shape = layer.get_shape()

	## Number of features will be img_height * img_width * num_channels.
	num_features = layer_shape[1:4].num_elements()
	layer = tf.reshape(layer, [-1, num_features])

	return layer

def create_fc_layer(input,num_inputs,num_outputs,use_relu=True):
	#Defining trainable weights and biases.
	weight = create_weights(shape=[num_inputs, num_outputs])
	biases = create_biases(num_outputs)

	#Fully connected layer takes input x and produces wx+b. Since, these are matrices, we use matmul function in tensorflow
	layer = tf.matmul(input, weight) + biases
	if use_relu:
		layer = tf.nn.relu(layer)
	return layer

beginTime = time.time()
image_size = 101
num_channels = 3
num_classes = 3
classes = ['center','left','right']
batch_size = 256
total_iterations = 0
ts_results_0 = []
ts_results_1 = []
ts_results_2 = []
ts_results_3 = []
ts_results_4 = []
ts_results_5 = []
ts_results_6 = []
ts_results_7 = []
ts_results_8 = []
ts_results_9 = []

def train():
	global total_iterations

	for i in range(0,1):

		with tf.Session() as session:
			session.run(tf.global_variables_initializer())
			session.run(tf.local_variables_initializer())

			x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, num_channels], name='x')
			## labels
			y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
			y_true_cls = tf.argmax(y_true, dimension=1)


			#Network graph params
			filter_size_conv1 = 4
			num_filters_conv1 = 32

			filter_size_conv2 = 4
			num_filters_conv2 = 32

			filter_size_conv3 = 4
			num_filters_conv3 = 32 #64

			filter_size_conv4 = 3
			num_filters_conv4 = 32

			fc_layer_size = 200 #

			layer_conv1 = create_convolutional_layer(input=x,num_input_channels=num_channels,conv_filter_size=filter_size_conv1,num_filters=num_filters_conv1)

			layer_conv2 = create_convolutional_layer(input=layer_conv1,num_input_channels=num_filters_conv1,conv_filter_size=filter_size_conv2,num_filters=num_filters_conv2)

			layer_conv3 = create_convolutional_layer(input=layer_conv2,num_input_channels=num_filters_conv2,conv_filter_size=filter_size_conv3,num_filters=num_filters_conv3)

			layer_conv4 = create_convolutional_layer(input=layer_conv3,num_input_channels=num_filters_conv3,conv_filter_size=filter_size_conv4,num_filters=num_filters_conv4)

			layer_flat = create_flatten_layer(layer_conv4)

			layer_fc1 = create_fc_layer(input=layer_flat,num_inputs=layer_flat.get_shape()[1:4].num_elements(),num_outputs=fc_layer_size,use_relu=True)

			layer_fc2 = create_fc_layer(input=layer_fc1,num_inputs=fc_layer_size,num_outputs=num_classes,use_relu=False)#outputs n classes

			print("out: ",layer_fc2.get_shape())


			batch = tf.Variable(0)

			y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

			y_pred_cls = tf.argmax(y_pred, dimension=1)

			saver = tf.train.Saver(tf.global_variables())

			#read the previously saved network
			saver.restore(session, './models/model_0'+'/tree-classes-model')
			#save the version of the network ready that can be compiled for NCS
			saver.save(session,'./models/model_0'+'/tree-classes-model_inference')


## Training
train()

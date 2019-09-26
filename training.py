import kfolds
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
	#Defining biases using the create_biases function.
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
	#Dropout Rate
	dropout_prob = 0.5
	if use_relu:
		layer = tf.nn.relu(layer)
		layer = tf.nn.dropout(layer,dropout_prob)
	return layer

beginTime = time.time()
dataset_path = './trailDetection/dataset/'
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

images,labels, path, cls, train_index, test_index, num_examples_train = kfolds.split_train_sets(dataset_path, classes, image_size)

def train():
	global total_iterations

	for i in range(0,2):
		print('kfold: ', i)

		#read the training and testing data
		data = kfolds.read_train_sets(images, train_index[i], test_index[i], image_size, labels, path, cls)

		num_iterations=int((len(data.train.labels)/batch_size)*5)


		tf.reset_default_graph()
		with tf.Graph().as_default(),tf.Session() as session:

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

			learningRate = tf.train.exponential_decay(learning_rate=0.05,global_step=batch*batch_size,decay_steps=len(data.train.labels),decay_rate=0.95,staircase=True)

			y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

			y_pred_cls = tf.argmax(y_pred, dimension=1)

			session.run(tf.global_variables_initializer())


			saver = tf.train.Saver()

			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)

			cost = tf.reduce_mean(cross_entropy)

			optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost,global_step=batch)

			correct_prediction = tf.equal(y_pred_cls, y_true_cls)

			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			for j in range(total_iterations,num_iterations):

				x_batch, y_true_batch,_,cls_batch = data.train.next_batch(batch_size)
				x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

				feed_dict_tr = {x: x_batch, y_true: y_true_batch}
				feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

				session.run(optimizer, feed_dict=feed_dict_tr)
				if j % int(len(data.train.labels)/batch_size) == 0:
					train_acc,train_loss, lr = session.run([accuracy,cost, learningRate], feed_dict=feed_dict_tr)
					val_acc,val_loss = session.run([accuracy,cost], feed_dict=feed_dict_val)
					epoch = int(j/int(len(data.train.labels)/batch_size))
					print("Epoch: {0}, Train Acc: {1:.2f}, Train Cost {2:.2f}, Test Acc: {3:.2f}, Test Cost: {4:.2f}, Learning Rate: {5:.5f}".format(epoch,train_acc, train_loss, val_acc, val_loss, lr))

					saver.save(session, './models/model_'+str(i)+'/tree-classes-model_'+str(i))

## Training
train()

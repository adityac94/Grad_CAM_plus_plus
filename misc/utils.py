import numpy as np
import tensorflow as tf
import GuideReLU as GReLU
import models.vgg16 as vgg16
import models.vgg_utils as vgg_utils
from skimage.transform import resize
import matplotlib.pyplot as plt
import os,cv2
from scipy.misc import imread, imresize
import tensorflow as tf
from tensorflow.python.framework import graph_util

def guided_BP(image, label_id = -1):	
	g = tf.get_default_graph()
	with g.gradient_override_map({'Relu': 'GuidedRelu'}):
		label_vector = tf.placeholder("float", [None, 1000])
		input_image = tf.placeholder("float", [None, 224, 224, 3])

		vgg = vgg16.Vgg16()
		with tf.name_scope("content_vgg"):
		    vgg.build(input_image)

		cost = vgg.fc8*label_vector
	
		# Guided backpropagtion back to input layer
		gb_grad = tf.gradients(cost, input_image)[0]

		init = tf.global_variables_initializer()
	
	# Run tensorflow 
	with tf.Session(graph=g) as sess:    
		sess.run(init)
		output = [0.0]*vgg.prob.get_shape().as_list()[1] #one-hot embedding for desired class activations
		if label_id == -1:
			prob = sess.run(vgg.prob, feed_dict={input_image:image})
		
			vgg_utils.print_prob(prob[0], './synset.txt')

			#creating the output vector for the respective class
			index = np.argmax(prob)
			print "Predicted_class: ", index
			output[index] = 1.0

		else:
			output[label_id] = 1.0
		output = np.array(output)
		gb_grad_value = sess.run(gb_grad, feed_dict={input_image:image, label_vector: output.reshape((1,-1))})

	return gb_grad_value[0] 

def grad_CAM_plus(filename, label_id = -1):
	g = tf.get_default_graph()
	init = tf.global_variables_initializer()
	
	# Run tensorflow 
	sess = tf.Session()

	#define your tensor placeholders for, labels and images
	label_vector = tf.placeholder("float", [None, 1000])
	input_image = tf.placeholder("float", [1, 224, 224, 3])
	label_index = tf.placeholder("int64", ())

	#load vgg16 model
	vgg = vgg16.Vgg16()
	with tf.name_scope("content_vgg"):
	    vgg.build(input_image)
	#prob = tf.placeholder("float", [None, 1000])

	#get the output neuron corresponding to the class of interest (label_id)
	cost = vgg.fc8*label_vector

	# Get last convolutional layer gradients for generating gradCAM++ visualization
	target_conv_layer = vgg.conv5_3
	target_conv_layer_grad = tf.gradients(cost, target_conv_layer)[0]
	
	#first_derivative
	first_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad 	
	
	#second_derivative
	second_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad*target_conv_layer_grad 

	#triple_derivative
	triple_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad*target_conv_layer_grad*target_conv_layer_grad  

	sess.run(init)

	img1 = vgg_utils.load_image("./images/" + filename)
		
	output = [0.0]*vgg.prob.get_shape().as_list()[1] #one-hot embedding for desired class activations
		#creating the output vector for the respective class
	
	if label_id == -1:
		prob_val = sess.run(vgg.prob, feed_dict={input_image:[img1]})
		vgg_utils.print_prob(prob_val[0], './synset.txt')
		#creating the output vector for the respective class
		index = np.argmax(prob_val)
		orig_score = prob_val[0][index]
		print "Predicted_class: ", index
		output[index] = 1.0
		label_id = index
	else:
		output[label_id] = 1.0	
	output = np.array(output)

	conv_output, conv_first_grad, conv_second_grad, conv_third_grad = sess.run([target_conv_layer, first_derivative, second_derivative, triple_derivative], feed_dict={input_image:[img1], label_index:label_id, label_vector: output.reshape((1,-1))})
	
	global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

	alpha_num = conv_second_grad[0]
	alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
	alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
	alphas = alpha_num/alpha_denom

	weights = np.maximum(conv_first_grad[0], 0.0)
	#normalizing the alphas
	
	alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)
	
	alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))
	
	deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
	#print deep_linearization_weights
	grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

	# Passing through ReLU
	cam = np.maximum(grad_CAM_map, 0)
	cam = cam / np.max(cam) # scale 0 to 1.0   

	cam = resize(cam, (224,224))
	# Passing through ReLU
	cam = np.maximum(grad_CAM_map, 0)
	cam = cam / np.max(cam) # scale 0 to 1.0    
	cam = resize(cam, (224,224))


	gb = guided_BP([img1], label_id)
	visualize(img1, cam, filename, gb) 
	return cam

def visualize(img, cam, filename,gb_viz):
    gb_viz = np.dstack((
            gb_viz[:, :, 2],
            gb_viz[:, :, 1],
            gb_viz[:, :, 0],
        ))

    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()
  
    fig, ax = plt.subplots(nrows=1,ncols=3)

    plt.subplot(131)
    plt.axis("off")
    imgplot = plt.imshow(img)

    plt.subplot(132)
    gd_img = gb_viz*np.minimum(0.25,cam).reshape(224,224,1)
    x = gd_img
    x = np.squeeze(x)
    
    #normalize tensor
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
   
    x = np.clip(x, 0, 255).astype('uint8')

    plt.axis("off")
    imgplot = plt.imshow(x, vmin = 0, vmax = 20)

    cam = cam*-1.0 + 1.0
    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    plt.subplot(133)
    plt.axis("off")
    imgplot = plt.imshow(cam_heatmap)

    plt.savefig("output/" + filename, dpi=300)
    plt.close(fig)


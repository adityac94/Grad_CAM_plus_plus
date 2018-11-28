#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
#import setGPU
from keras.models import model_from_json
from keras.layers.core import Lambda
import tensorflow as tf
import os
import cv2
import numpy as np
from skimage.transform import resize
import scipy.ndimage
import matplotlib.pyplot as plt
import c3d_model
import sys
import keras.backend as K
# K.set_image_dim_ordering('th')
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
dim_ordering = K.image_dim_ordering
print "[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
        dim_ordering)
backend = dim_ordering

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def diagnose(data, verbose=True, label='input', plots=False, backend='tf'):
    # Convolution3D?
    if data.ndim > 2:
        if backend == 'th':
            data = np.transpose(data, (1, 2, 3, 0))
        #else:
        #    data = np.transpose(data, (0, 2, 1, 3))
        min_num_spatial_axes = 10
        max_outputs_to_show = 3
        ndim = data.ndim
        print "[Info] {}.ndim={}".format(label, ndim)
        print "[Info] {}.shape={}".format(label, data.shape)
        for d in range(ndim):
            num_this_dim = data.shape[d]
            if num_this_dim >= min_num_spatial_axes: # check for spatial axes
                # just first, center, last indices
                range_this_dim = [0, num_this_dim/2, num_this_dim - 1]
            else:
                # sweep all indices for non-spatial axes
                range_this_dim = range(num_this_dim)
            for i in range_this_dim:
                new_dim = tuple([d] + range(d) + range(d + 1, ndim))
                sliced = np.transpose(data, new_dim)[i, ...]
                print("[Info] {}, dim:{} {}-th slice: "
                      "(min, max, mean, std)=({}, {}, {}, {})".format(
                              label,
                              d, i,
                              np.min(sliced),
                              np.max(sliced),
                              np.mean(sliced),
                              np.std(sliced)))
        if plots:
            # assume (l, h, w, c)-shaped input
            if data.ndim != 4:
                print("[Error] data (shape={}) is not 4-dim. Check data".format(
                        data.shape))
                return
            l, h, w, c = data.shape
            if l >= min_num_spatial_axes or \
                h < min_num_spatial_axes or \
                w < min_num_spatial_axes:
                print("[Error] data (shape={}) does not look like in (l,h,w,c) "
                      "format. Do reshape/transpose.".format(data.shape))
                return
            nrows = int(np.ceil(np.sqrt(data.shape[0])))
            # BGR
            if c == 3:
                for i in range(l):
                    mng = plt.get_current_fig_manager()
                    mng.resize(*mng.window.maxsize())
                    plt.subplot(nrows, nrows, i + 1) # doh, one-based!
                    im = np.squeeze(data[i, ...]).astype(np.float32)
                    im = im[:, :, ::-1] # BGR to RGB
                    # force it to range [0,1]
                    im_min, im_max = im.min(), im.max()
                    if im_max > im_min:
                        im_std = (im - im_min) / (im_max - im_min)
                    else:
                        print "[Warning] image is constant!"
                        im_std = np.zeros_like(im)
                    plt.imshow(im_std)
                    plt.axis('off')
                    plt.title("{}: t={}".format(label, i))
                plt.show()
                #plt.waitforbuttonpress()
            else:
                for j in range(min(c, max_outputs_to_show)):
                    for i in range(l):
                        mng = plt.get_current_fig_manager()
                        mng.resize(*mng.window.maxsize())
                        plt.subplot(nrows, nrows, i + 1) # doh, one-based!
                        im = np.squeeze(data[i, ...]).astype(np.float32)
                        im = im[:, :, j]
                        # force it to range [0,1]
                        im_min, im_max = im.min(), im.max()
                        if im_max > im_min:
                            im_std = (im - im_min) / (im_max - im_min)
                        else:
                            print "[Warning] image is constant!"
                            im_std = np.zeros_like(im)
                        plt.imshow(im_std)
                        plt.axis('off')
                        plt.title("{}: o={}, t={}".format(label, j, i))
                    plt.show()
                    #plt.waitforbuttonpress()
    elif data.ndim == 1:
        print("[Info] {} (min, max, mean, std)=({}, {}, {}, {})".format(
                      label,
                      np.min(data),
                      np.max(data),
                      np.mean(data),
                      np.std(data)))
        print("[Info] data[:10]={}".format(data[:10]))

    return

def main():
    show_images = False
    diagnose_plots = False
    model_dir = './models'
    global backend

    # override backend if provided as an input arg
    if len(sys.argv) > 1:
        if 'tf' in sys.argv[1].lower():
            backend = 'tf'
        else:
            backend = 'th'
    print "[Info] Using backend={}".format(backend)

    if backend == 'th':
        print "hi"
        model_weight_filename = os.path.join(model_dir, 'sports1M_weights_th.h5')
        model_json_filename = os.path.join(model_dir, 'sports1M_weights_th.json')
    else:
        print "hello"
        model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
        model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

    print("[Info] Reading model architecture...")
    model = model_from_json(open(model_json_filename, 'r').read())
    # model = c3d_model.get_model(backend=backend)

    # visualize model
    model_img_filename = os.path.join(model_dir, 'c3d_model.png')
    if not os.path.exists(model_img_filename):
        from keras.utils.visualize_util import plot
        plot(model, to_file=model_img_filename)

    print("[Info] Loading model weights...")
    model.load_weights(model_weight_filename)
    print("[Info] Loading model weights -- DONE!")
    model.compile(loss='mean_squared_error', optimizer='sgd')

    print("[Info] Loading labels...")
    with open('sports1m/labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))

    print("[Info] Loading a sample video...")

    f = open("scores.txt","w")

    for filename in sorted(os.listdir("videos/")):
        try:
            cap = cv2.VideoCapture("videos/" + filename)
            print filename
            vid = []
            while True:
                ret, img = cap.read()
                if not ret:
                    break
                vid.append(cv2.resize(img, (171, 128)))
            vid = np.array(vid, dtype=np.float32)

            start_frame = 1000

            X = vid[start_frame:(start_frame + 16), :, :, :]

            # subtract mean
            mean_cube = np.load('models/train01_16_128_171_mean.npy')
            mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))

            # center crop
            X = X[:, 8:120, 30:142, :] # (l, h, w, c)

            if backend == 'th':
                X = np.transpose(X, (3, 0, 1, 2)) # input_shape = (3,16,112,112)
            else:
                pass                              # input_shape = (16,112,112,3)

            if 'lambda' in model.layers[-1].name:
                model.layers.pop()
                model.outputs = [model.layers[-1].output]
                model.output_layers = [model.layers[-1]]
                model.layers[-1].outbound_nodes = []


            # inference
            output = model.predict_on_batch(np.array([X]))

            #################################################
            print X.shape
            predicted_class = np.argmax(output)
            print predicted_class
            print output[0][predicted_class], labels[predicted_class]
            

            nb_classes = len(labels)#487
            target_layer = lambda x: target_category_loss(x, predicted_class, nb_classes)
            model.add(Lambda(target_layer, output_shape = target_category_loss_output_shape))
            temp_label = np.zeros(output.shape)
            temp_label[0][int(np.argmax(output))] = 1.0
            loss = K.sum(model.layers[-1].output*(temp_label))

            for i in range(14):
                ###########Choose a conv layer to generate saliency maps##########
                if model.layers[i].name == "conv3a":
                    conv_output = model.layers[i].output


            grads = normalize(K.gradients(loss, conv_output)[0])

            first_derivative = tf.exp(loss)*grads 
            print first_derivative[0]
            print tf.exp(loss)
                    
            #second_derivative
            second_derivative = tf.exp(loss)*grads*grads 
            print second_derivative[0]

            #triple_derivative
            triple_derivative = tf.exp(loss)*grads*grads*grads
            print triple_derivative[0] 


            gradient_function = K.function([model.layers[0].input, K.learning_phase()], [conv_output, grads, first_derivative, second_derivative, triple_derivative])
            grads_output, grads_val, conv_first_grad, conv_second_grad, conv_third_grad = gradient_function([np.array([X]), 0])
            grads_output, grads_val, conv_first_grad, conv_second_grad, conv_third_grad = grads_output[0, :, :], grads_val[0, :, :, :], conv_first_grad[0, :, :, :], conv_second_grad[0, :, :, :], conv_third_grad[0, :, :, :]
            print grads_output.shape, np.max(grads_output), np.min(grads_output)
            print grads_val.shape, np.max(grads_val), np.min(grads_val)
            print conv_first_grad.shape,np.max(conv_first_grad), np.min(conv_first_grad)
            print conv_second_grad.shape,np.max(conv_second_grad), np.min(conv_second_grad)
            print conv_third_grad.shape,np.max(conv_third_grad), np.min(conv_third_grad)


            ############## FOR GRAD-CAM #########################################

            weights = np.mean(grads_val, axis = (0, 1, 2))
            print weights.shape
            cam = np.zeros(grads_output.shape[0 : 3], dtype = np.float32)
            print cam.shape

            cam = np.sum(weights*grads_output, axis=3)
            print np.max(cam),np.min(cam)

            cam = np.maximum(cam, 0)
            cam = scipy.ndimage.zoom(cam, (2, 4, 4))
            heatmap = cam / np.max(cam)
            print np.max(heatmap),np.min(heatmap)
            print heatmap.shape

            vid_mod = X*heatmap.reshape((16,112,112,1))
            print vid_mod.shape
            output_mod = model.predict_on_batch(np.array([vid_mod]))

            predicted_class_mod = output_mod[0].argsort()[::-1][0]
            print output_mod[0][predicted_class_mod], labels[predicted_class_mod]
            print output_mod[0][predicted_class], labels[predicted_class]

            ################SAVE THE VIDEO AS FRAMES###############

            for i in range(heatmap.shape[0]):
                cam_mod = heatmap[i].reshape((112,112,1))

                gd_img_mod = X[i]*cam_mod

                gd_img_mod = cv2.resize(gd_img_mod, (640,480))
                cv2.imwrite("image-%05d.jpg" %i, gd_img_mod)

            ####################### GRAD-CAM UPTO THIS ############################

            ############## FOR GradCAM++ #################################
            global_sum = np.sum(conv_third_grad.reshape((-1,256)), axis=0)
            #print global_sum
                    
            alpha_num = conv_second_grad
            alpha_denom = conv_second_grad*2.0 + conv_third_grad*global_sum.reshape((-1,))
            alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
            alphas = alpha_num/alpha_denom

            weights = np.maximum(conv_first_grad, 0.0)
            #normalizing the alphas
            alphas_thresholding = np.where(weights, alphas, 0.0)

            alpha_normalization_constant = np.sum(np.sum(np.sum(alphas_thresholding, axis=0),axis=0),axis=0)
            alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))


            alphas /= alpha_normalization_constant_processed.reshape((1,1,1,256))
            #print alphas



            deep_linearization_weights = np.sum((weights*alphas).reshape((-1,256)),axis=0)
            #print deep_linearization_weights
            grad_CAM_map = np.sum(deep_linearization_weights*grads_output, axis=3)
            print np.max(grad_CAM_map),np.min(grad_CAM_map)

            grad_CAM_map = scipy.ndimage.zoom(grad_CAM_map, (2, 4, 4))
            print np.max(grad_CAM_map),np.min(grad_CAM_map)
            # Passing through ReLU
            vid_cam = np.maximum(grad_CAM_map, 0)
            vid_heatmap = vid_cam / np.max(vid_cam) # scale 0 to 1.0  
            print vid_heatmap.shape


            vid_mod_plus = X*vid_heatmap.reshape((16,112,112,1))
            print vid_mod_plus.shape
            output_mod_plus = model.predict_on_batch(np.array([vid_mod_plus]))
            predicted_class_mod_plus = output_mod_plus[0].argsort()[::-1][0]
            print output_mod_plus[0][predicted_class_mod_plus], labels[predicted_class_mod_plus]
            print output_mod_plus[0][predicted_class], labels[predicted_class]


            ################SAVE THE VIDEO AS FRAMES###############


            for i in range(vid_heatmap.shape[0]):
                vid_cam_mod = vid_heatmap[i].reshape((112,112,1))
            

                vid_gd_img_mod = X[i]*vid_cam_mod 
                vid_gd_img_mod = cv2.resize(vid_gd_img_mod, (640,480))
                cv2.imwrite(os.path.join("./output", "image-%05d.jpg" %i), vid_gd_img_mod)

                
                X_mod = cv2.resize(X[i], (640,480))
                cv2.imwrite("original-image-%05d.jpg" %i, X_mod)

            #############GRAD-CAM++ UPTO THIS ####################################

            #############Write the scores into a file#############################
            f.write(str(output[0][predicted_class]) + " " + str(output_mod[0][predicted_class]) + " " + str(output_mod_plus[0][predicted_class]) + "\n")

        except:
            print filename
            continue



    # sort top five predictions from softmax output
    top_inds = output[0].argsort()[::-1][:5]  # reverse sort and take five largest items
    print('\nTop 5 probabilities and labels:')
    for i in top_inds:
        print('{1}: {0:.5f}'.format(output[0][i], labels[i]))

if __name__ == '__main__':
    main()

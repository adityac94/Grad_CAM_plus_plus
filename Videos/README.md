C3D Model for Grad-CAM and Grad-CAM++ with Keras + TensorFlow
=============================================================

The scripts here are inspired by [`C3D Model for Keras`](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2) gist, but specifically for Keras + TensorFlow (not Theano-backend).

To reproduce results:

- Run each of these steps:

1. Download the pretrained weights of the C3D model (sports1M_weights_tf.h5 and sports1M_weights_tf.json) from the following link, and copy the file to the models/ subdirectory.: `https://drive.google.com/drive/folders/11fi8rtYnPmiyMngUVeOF4vBzUjnISYiH?usp=sharing`
2. Download sport1mil labels: `bash sports1m/get_labels.sh`
3. Make sure the default keras config (in `~/.keras/keras.json`) has: `tf` image_dim_ordering, and `tensorflow` backend.
4. Download some test video and copy to the videos/ subdirectory.
5. Run test: `python test_model.py`

Prerequisites
=============
Known to work with the following python packages:
- Keras==2.0.6
- tensorflow==1.4.0
- h5py==2.8.0
- numpy==1.14.2

Results
=======
For every video in the videos/ subdirectory, this code will generate saliency maps for Grad-CAM and Grad-CAM++ with respect to the convolution layer selected in the code.

The top 5 labels will also be reported, and should look something like:

```
Top 5 probabilities and labels:
basketball: 0.71422
streetball: 0.10293
volleyball: 0.04900
greco-roman wrestling: 0.02638
freestyle wrestling: 0.02408
```

References
==========

1. [C3D Model for Keras](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2)
2. [Original C3D implementation in Caffe](https://github.com/facebook/C3D)
3. [C3D paper](https://arxiv.org/abs/1412.0767)

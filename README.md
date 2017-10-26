# Grad_CAM++
A generalized gradient-based CNN visualization technique

Kindly download the pretrained weights of the vgg16 network from the following link, and copy the file to the models/ subdirectory.
https://drive.google.com/drive/folders/0BzS5KZjihEdyUjBHcGFNRnk4bFU?usp=sharing

USAGE:
python classify.py -f water-bird.JPEG  

Arguments:
-f: path to input image
-gpu: the gpu id to use, 0-indexed
-l: class label, default is -1 (chooses the class predicted by the model)
-o: Specify output file name for Grad-CAM++ visualization, default is 'output.jpeg'. All results would be saved in the output/ subdirectory.


FOR HELP:
python classify.py -h


The above code is for the vgg16 network, pretrained on imagenet.  
We tested our code on tensorflow 1.3, compatibility with other versions is not guaranteed.

Acknowledgements
Parts of the code have been borrowed and modified from: 
https://github.com/Ankush96/grad-cam.tensorflow
https://github.com/insikk/Grad-CAM-tensorflow

If using this code, please cite our work:



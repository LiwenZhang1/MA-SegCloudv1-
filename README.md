# MA-SegCloudv1- A novel ground-based cloud image segmentation method based on a multibranch asymmetric convolution module and attention mechanism

   With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

![image](https://github.com/LiwenZhang1/MA-SegCloudv1-/blob/master/Figure1.png)

Please cite the above paper if you intend to use whole/part of the code. This code is only for academic and research purposes.
# Usage:

1. Dataset：The SWINySeg data set is available for download at http://vintage.winklerbros.net/swinyseg.html. All images are normalized to binary images, the size is changed to 320×320, and the training set and test set are divided by voc2pspnet.py.

2. Training: Set the path for train.py to read images and labels, and load training and validation sets.

   (a).Read image from file: img = Image.open(r"filepath" + '/' + name + ".jpg")

   (b).Read label image from file: label = Image.open(r"filepath" + '/' + name + ".png")

   (c).Set hyperparameters such as learning rate, optimizer, loss function, etc.

3. prediction：Load the trained weight file, set the file path to save the prediction results, and run the predict.py.

4. test: Load predictions and labels, run test.py.

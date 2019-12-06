# Image Segmentation using FastAI - Eye Segmentation
Eye segmentation using FastAI

Training data: 500 images/masks over 20 epochs.

Accuracy: Achieved 95.7% accuracy.

Accuracy calculation was done by comparing the number of pixels between the ground truth and the predicted mask (ignoring the background).

Image example:

<img src="images/eye.png" height="200" width="300">
![Image](images/eye.png "Training Image"){:height="200px" width="200px"}

Mask example:

![Mask](images/mask.png "Mask - Ground Truth"){:height="200px" width="200px"}

Output:

![Output](images/output.PNG "Output")

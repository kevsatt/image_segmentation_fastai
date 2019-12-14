# Image Segmentation using FastAI - Eye Segmentation
Eye segmentation using FastAI

Training data: 500 images/masks over 100 epochs.

Accuracy calculation was done by comparing the number of pixels between the ground truth and the predicted mask (ignoring the background).

def acc_image_seg(input, target):
    """ accuracy measure disregarding the background pixels """
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()
    
Accuracy: Achieved 96.7% accuracy.

<p align="center">
Image example:
<br><br>
<img src="images/eye.png" height="300" width="300">
  </p>
  
<p align="center">
Mask example:
<br><br>
<img src="images/mask.png" height="300" width="300">
 </p>
 
<p align="center">
Output:
<br><br>
<img src="images/output.PNG" height="450" width="450">
</p>

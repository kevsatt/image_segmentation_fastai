# Image Segmentation using FastAI - Eye Segmentation
Eye segmentation using FastAI

Training data: 500 images/masks over 100 epochs.

Accuracy calculation was done by comparing the number of pixels between the ground truth and the predicted mask (ignoring the background).
    
Accuracy: Achieved 96.7% accuracy.

Experimented using Dice loss function compared with the default categorical cross-entropy loss function to see if accuracy or F1-score would increase with unsuccessful results.

I used F1-score (the balance between precision and recall) as a heuristic measurement to compare models.

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

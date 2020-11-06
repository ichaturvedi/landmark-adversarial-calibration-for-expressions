Landmark Calibration for Facial Expressions
---
This code implements the model discussed in Landmark Calibration for Facial Expression. Accurately predictiong landmarks is critical for detecting subtle emotions such as anger. Here we use prinipal component analysis to calibrate landmarks. Next, we train a translation model to generate face expressions from landmarks. We show that calibration can increase the resolution of the generated image significantly. 

Requirements
---
This code is based on the Deepwalk code found at:
https://github.com/MayankSingal/PyTorch-Pixel-Level-Domain-Transfer

Facial Landmarks
---
Extract the landmarks<br>
*python facial_landmarks.py -p shape_predictor_68_face_landmarks.dat -i emotion1.jpg*
- p is pretrained detector (available at https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat)
- i is input face image

Calibration
---
Calibrate an emotion using SVD<br>
*[par_x, par_y] = calibrate(goldface, goldland, targetface, targetland)*
- goldface is a high intensity emotional face
- goldland contains landmarks for goldface
- targetface is a low intensity emotional face
- targetland contains landmarks for targetface

Example calibration for Happy emotion. The first face is the Gold standard. The second face is the target ( without(red) and with(green) calibration)<br><br>
![gold_original](https://user-images.githubusercontent.com/65399216/98350932-bbd5bf80-2067-11eb-93f6-27eba6a3ab60.jpg)
![sample_original](https://user-images.githubusercontent.com/65399216/98350943-bed0b000-2067-11eb-9ba4-b993e6f61b99.jpg)
![sample_calibrated](https://user-images.githubusercontent.com/65399216/98350955-c1cba080-2067-11eb-84f1-16dc357b8a3a.jpg)




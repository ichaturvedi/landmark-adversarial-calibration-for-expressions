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

![gold_original](https://user-images.githubusercontent.com/65399216/98348335-53391380-2064-11eb-835e-e2567e53705f.jpg = 100x100)
![sample_original](https://user-images.githubusercontent.com/65399216/98348352-58965e00-2064-11eb-94fa-571821ccf839.jpg = 100x100)
![sample_calibrated](https://user-images.githubusercontent.com/65399216/98348360-5c29e500-2064-11eb-8793-6f84d6e41240.jpg = 100x100)


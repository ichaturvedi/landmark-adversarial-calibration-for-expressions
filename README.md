Landmark Calibration for Facial Expressions
---
This code implements the model discussed in Landmark Calibration for Facial Expression. Accurately predictiong landmarks is critical for detecting subtle emotions such as anger. Here we use prinipal component analysis to calibrate landmarks. Next, we train a translation model to generate face expressions from landmarks. We show that calibration can increase the resolution of the generated image significantly. 

Requirements
---
This code is based on the Deepwalk code found at:
https://github.com/MayankSingal/PyTorch-Pixel-Level-Domain-Transfer

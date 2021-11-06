# Facial-Recognition-for-Retails--Enhance-In-store-Customer-experience-EAGLE
#Description: 
The system is able to recognize the face of customers visiting to the store and send them welcome text messages and the offers/coupons for them on their personal mobile number once they enter in the Store.
#Tools Used: YOLO V3, Sklearn, Google colab, Labellmg, Open CV, IOT. 
#Programming Language: Python




The idea behind YOLO is this: There are no classification/detection modules that need to sync with each other and no recurring region proposal loops as in previous 2-stage detectors (see my post on early object detectors like RCNN). It’s basically convolutions all the way down (with the occasional maxpool layer). Instead of cropping out areas with high probability for an object and feeding them to a network that finds boxes, a single monolithic network needs to take care of feature extraction, box regression and classification. While previous models had two output layers — one for the class probability distribution and one for box predictions, here a single output layer contains everything in different features.



YOLO-V3 Architecture
Inspired by ResNet and FPN (Feature-Pyramid Network) architectures, YOLO-V3 feature extractor, called Darknet-53 (it has 52 convolutions) contains skip connections (like ResNet) and 3 prediction heads (like FPN) — each processing the image at a different spatial compression.


Feature Pyramid Network (FPN): Dancing At Two Weddings
A Feature-Pyramid is a topology developed in 2017 by FAIR (Facebook A.I. Research) in which the feature map gradually decreases in spatial dimension (as is the usual case), but later the feature map expands again and is concatenated with previous feature maps with corresponding sizes. This procedure is repeated, and each concatenated feature map is fed to a separate detection head.

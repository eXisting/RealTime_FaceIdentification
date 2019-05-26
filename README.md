# Facenet: Real-time face recognition using deep learning Tensorflow 

Here is the implementation of face recognition in real time. Solution can fit various types of video sources thanks for OpenCV. For example, in this case, video feed from camera that connect over Wi-Fi local network is been used.

# Usage

- First of all you must prepare your recogniation classifier. To do that you must train it on your aligned data set using Transfer learning and existing model as starting point. Model can be found udner model folder. It is Inception-V1, so you could connect any of your models if you want.

For prapre an classifier:
1. Get dataset of human photos
2. Create folder training_data
3. Put all of your photos under training_data folder splited by their own folder
4. Run TRAIN command

- Use your classifier to classify image. You can feed a photo file or feed video stream. To do that look into the identify_face_image.py and identify_face_video.py

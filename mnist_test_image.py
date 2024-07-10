# MNIST CNN test
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

MODEL_SAVE_FOLDER_PATH = './model/'
model=tf.keras.models.load_model(MODEL_SAVE_FOLDER_PATH+"mnist_model.hdf5")
model.summary()

file="num8.jpg"		# 흰바탕-검은글씨
image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#image = cv2.resize(image, (28,28))
image = image.astype('float32')
image = image.reshape(1, 28, 28, 1)
image = 255-image          #inverts image. 검은바탕에 흰글시
image /= 255 	               #0..1사이의 값

plt.imshow(image.reshape(28, 28),cmap='Greys')
plt.show()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print('학번 : 20180757')
print('이름 : 강민준')
print('학과 : 소프트웨어융합학과')
pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)
for i in range(10):
    print(class_names[i], pred[0][i])
print("max arg=",np.argmax(pred),"[",np.max(pred),"]")

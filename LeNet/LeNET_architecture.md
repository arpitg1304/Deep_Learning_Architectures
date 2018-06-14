

```python
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Conv2D, MaxPooling2D
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
```


```python
# Creating a LeNel model class that can be imported to use anywher in the script

class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath= None):
        model = Sequential()
        # First set of conv, relu and maxpooling
        model.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf",  strides=(2, 2)))
        
        # Second set
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf",  strides=(2, 2)))
        
        # Flatten layer and dense 
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        
        # Softmax in the end
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        if weightsPath is not None:
            model.load_weights(weightsPath)
        
        return model
```


```python
# Writing driver script to use the LeNet model class created above
TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import cv2
```


```python
print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
data = data[:, :, :, np.newaxis]
(trainData, testData, trainLabels, testLabels) = train_test_split(data / 255.0, dataset.target.astype("int"), test_size=0.33)
```

    [INFO] downloading MNIST...



```python
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)
data.shape
```




    (70000, 28, 28, 1)




```python
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
# model.fit(trainData, trainLabels, batch_size=128, nb_epoch=20,verbose=1)
model.load_weights("weights")
```

    /home/arpit/.local/lib/python3.5/site-packages/ipykernel_launcher.py:10: UserWarning: Update your `MaxPooling2D` call to the Keras 2 API: `MaxPooling2D(strides=(2, 2), data_format="channels_last", pool_size=(2, 2))`
      # Remove the CWD from sys.path while we load stuff.
    /home/arpit/.local/lib/python3.5/site-packages/ipykernel_launcher.py:15: UserWarning: Update your `MaxPooling2D` call to the Keras 2 API: `MaxPooling2D(strides=(2, 2), data_format="channels_last", pool_size=(2, 2))`
      from ipykernel import kernelapp as app



```python
model.save_weights("weights", overwrite=True)
(loss, accuracy) = model.evaluate(testData, testLabels,batch_size=128, verbose=1)
```

    23100/23100 [==============================] - 7s 284us/step



```python
print(accuracy *100)
```

    98.27272727685573



```python
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)
 

    image = (testData[i][0] * 255).astype("uint8")
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(prediction[0]), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
 
    # show the image and prediction
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], np.argmax(testLabels[i])))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)
```

    [INFO] Predicted: 5, Actual: 5
    [INFO] Predicted: 9, Actual: 9
    [INFO] Predicted: 9, Actual: 9
    [INFO] Predicted: 2, Actual: 2
    [INFO] Predicted: 2, Actual: 2
    [INFO] Predicted: 0, Actual: 0
    [INFO] Predicted: 9, Actual: 9
    [INFO] Predicted: 1, Actual: 1
    [INFO] Predicted: 5, Actual: 5
    [INFO] Predicted: 8, Actual: 8


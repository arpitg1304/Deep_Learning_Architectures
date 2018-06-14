
# coding: utf-8

# In[1]:


from sklearn.cross_validation import train_test_split
import sys
import os
import tensorflow as tf
tf.set_random_seed(1000)
import numpy as np
np.random.seed(1000)
import sys
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


# In[2]:


# Managing the dataset
import tflearn.datasets.oxflower17 as oxflower17
x, y = oxflower17.load_data(one_hot=True)
x_path='/tmp/oxford_flower_17_x.npy'
y_path='/tmp/oxford_flower_17_y.npy'
import numpy as np
np.save(x_path, x)
np.save(y_path, y)
x = np.load(x_path)
y = np.load(y_path)


# In[3]:


x_train, x_test_pre, y_train, y_test_pre = train_test_split(x, y, test_size=0.20, random_state=42)
x_test, x_validation, y_test, y_validation = train_test_split(x_test_pre, y_test_pre, test_size=0.1)

# x = tf.placeholder(tf.float32, [None, 224, 224, 3])
# y = tf.placeholder(tf.float32, [None, 17]) # no of flower speces in the dataset


# In[4]:


# Defining functions to generate layers
def dense(W, x, b):
    z = tf.add((tf.matmul(x, W)),b)
    a = tf.nn.relu(z)
    return a

def maxPool2d(x, kernel_size, stride_size):
    return(tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1],padding="SAME"))

def conv2D(x, W, b, stride_size):
    xW = tf.nn.conv2d(x, W, strides=[1, stride_size, stride_size, 1],padding="SAME")
    z = tf.nn.bias_add(xW, b)
    a = tf.nn.relu(z)
    return a


# In[5]:


# Building the AlexNet Model
tf.reset_default_graph()
x_init = tf.contrib.layers.xavier_initializer()
n_classes = 17

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, 17]) # no of flower speces in the dataset

def AlexNet(img_input):
    
    
    # 1st conv layer: conv, pool, batch_norm
    w_c1 = tf.get_variable("w_c1", [11,11,3,96], initializer=x_init)
    b_c1 = tf.Variable(tf.zeros([96]))
    c1 = conv2D(img_input, w_c1, b_c1, stride_size=4)
    p1 = maxPool2d(c1, kernel_size=2, stride_size=2)
    bn1 = tf.contrib.layers.batch_norm(p1)
    
    # 2nd conv layer: conv,pool, batch_norn
    w_c2 = tf.get_variable("w_c2", [5,5,96,256], initializer=x_init)
    b_c2 = tf.Variable(tf.zeros([256]))
    c2 = conv2D(bn1, w_c2, b_c2, stride_size=1)
    p2 = maxPool2d(c2, kernel_size=2, stride_size=1)
    bn2 = tf.contrib.layers.batch_norm(p2)
    
    # 3rd conv layer: conv, norm, (no pooling)
    w_c3 = tf.get_variable("w_c3", [3,3,256,384], initializer=x_init)
    b_c3 = tf.Variable(tf.zeros([384]))
    c3 = conv2D(bn2, w_c3, b_c3, stride_size=1)
    bn3 = tf.contrib.layers.batch_norm(c3)
    
    # 4th conv layer: conv, norm, (no pooling)
    w_c4 = tf.get_variable("w_c4", [3,3,384,384], initializer=x_init)
    b_c4 = tf.Variable(tf.zeros([384]))
    c4 = conv2D(bn3, w_c4, b_c4, stride_size=1)
    bn4 = tf.contrib.layers.batch_norm(c4)
    
    # 5th conv layer: conv, pool, norm
    w_c5 = tf.get_variable("w_c5", [3,3,384,256], initializer=x_init)
    b_c5 = tf.Variable(tf.zeros([256]))
    c5 = conv2D(bn4, w_c5, b_c5, stride_size=1)
    p3 = maxPool2d(c5, kernel_size=2, stride_size=2)
    bn5 = tf.contrib.layers.batch_norm(p3)
    
    # 1st dense layer: flatten the conv layer
    
    flattened = tf.reshape(bn5, [-1, 28*28*256])
    
    w_d1 = tf.get_variable("w_d1", [28*28*256,4096], initializer=x_init)
    b_d1 = tf.Variable(tf.zeros([4096]))
    d1 = dense(w_d1, flattened, b_d1)
    drop_d1 = tf.nn.dropout(d1, 0.5)
    
    # 2nd dense layer
    w_d2 = tf.get_variable("w_d2", [4096,4096], initializer=x_init)
    b_d2 = tf.Variable(tf.zeros([4096]))
    d2 = dense(w_d2, drop_d1, b_d2)
    drop_d2 = tf.nn.dropout(d2, 0.5)
    
    # 3rd dense layer
    w_d3 = tf.get_variable("w_d3", [4096,1000], initializer=x_init)
    b_d3 = tf.Variable(tf.zeros([1000]))
    d3 = dense(w_d3, drop_d2, b_d3)
    drop_d3 = tf.nn.dropout(d3, 0.5)
    
    # Output layer
    w_out = tf.get_variable("w_out", [1000, n_classes], initializer=x_init)
    b_out = tf.Variable(tf.zeros([n_classes]))
    out = tf.add((tf.matmul(drop_d3, w_out)), b_out)
    return out


# In[6]:


prediction = AlexNet(x)


# In[7]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
optimizer = tf.train.AdamOptimizer().minimize(cost)


# defining model evaluation metrics

correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy_pct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100


# In[ ]:


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config_proto = tf.ConfigProto(device_count = {'GPU': 1})
sess = tf.Session(config=config_proto)
initializer_op = tf.global_variables_initializer()
sess.run(initializer_op)
    


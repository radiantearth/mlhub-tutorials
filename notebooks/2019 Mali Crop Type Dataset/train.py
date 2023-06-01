import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import tensorflow as tf; print("tensorflow", tf.__version__)
#required libraries
import os
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
print(tf.version.VERSION)
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam 



#source and label file paths
data = Path("./data").resolve()
source = Path("./data/umd_mali_crop_type_source").resolve()
label = Path("./data/umd_mali_crop_type_labels").resolve()

#locate source and label items
src_files=next(os.walk(source))[1]
label_loc=next(os.walk(label))[1]
#exclude _common folder, as it's not a label folder
label_loc.remove('_common')
#find the label ID e.g num is 31 in umd_mali_crop_type_source_31
X= np.zeros((len(src_files), 256, 256,3), dtype=np.uint8)
Y= np.zeros((len(src_files), 256, 256), dtype=np.uint8)
j=0
for item in label_loc:
    num=item.split('_')[5]
    field = np.asarray(Image.open(f"{label}/{item}/labels.tif")) #load label image
    for src in src_files:
        if src[:-11]==f"umd_mali_crop_type_source_{num}": #find all source items belonging to that label ID
            source_img=np.asarray(Image.open(f"{data}/rgb_source/{src}.tif"))
            X[j]=source_img
            Y[j]=field
            j=j+1
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

def conv_block(inputs=None, n_filters=4, dropout_prob=0, max_pooling=True):
    #code reference: https://www.kaggle.com/yesa911/aerial-semantic-segmentation-96-acc
    conv = Conv2D(n_filters, #filter number to determine number of abstractions the network is able to extract from image data
                  kernel_size = 3,   #3x3  
                  activation='relu',
                  padding='same',
                  kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    conv = Conv2D(n_filters, 
                  kernel_size = 3, 
                  activation='relu',
                  padding='same',
                  kernel_initializer=tf.keras.initializers.HeNormal())(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2,2))(conv)
        
        
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    
    up = Conv2DTranspose(
                 n_filters,  
                 kernel_size = 3,
                 strides=(2,2),
                 padding='same')(expansive_input)
    
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,  
                 kernel_size = 3,   
                 activation='relu',
                 padding='same',
                 kernel_initializer=tf.keras.initializers.HeNormal())(merge)
    
    conv = Conv2D(n_filters,  
                 kernel_size = 3,  
                 activation='relu',
                 padding='same',
                 kernel_initializer=tf.keras.initializers.HeNormal())(conv)
    
    
    return conv


def unet_model(input_size=(256, 256, 3), n_filters=4, n_classes=5): #number of classes
    #code reference: https://www.kaggle.com/yesa911/aerial-semantic-segmentation-96-acc
    inputs = Input(input_size)
    
    #contracting path
    cblock1 = conv_block(inputs, n_filters)
    
    cblock2 = conv_block(cblock1[0], 2*n_filters)
    
    cblock3 = conv_block(cblock2[0], 4*n_filters)
    
    cblock4 = conv_block(cblock3[0], 8*n_filters, dropout_prob=0.3) 
    
    cblock5 = conv_block(cblock4[0],16*n_filters, dropout_prob=0.3, max_pooling=None)  
    
    #expanding path
    ublock6 = upsampling_block(cblock5[0], cblock4[1],  8 * n_filters)
    ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters*4)
    ublock8 = upsampling_block(ublock7,cblock2[1] , n_filters*2)
    ublock9 = upsampling_block(ublock8,cblock1[1],  n_filters)

    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)
    #conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(n_classes, kernel_size=1, padding='same')(conv9)  
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


img_height = 256 #our image size
img_width = 256
num_channels = 3 #modify since we have 3 channels (rgb)

unet = unet_model((img_height, img_width, num_channels))#save the model based on best validation loss
model_checkpoint = ModelCheckpoint(str(data)+'/unet_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
#stop training if no improvements after 50 epochs
model_earlyStopping = EarlyStopping(min_delta= 0.001, patience=50) #stop after 50 epochs if no improvements

#setting up and compiling the model
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.003, name="Adam"
)
loss= loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
unet.compile(optimizer=optimizer, loss= loss,\
                  metrics = ['accuracy'])

batch_size=8
 
history = unet.fit(x=x_train, y=y_train,
              validation_data=(x_val, y_val),
              steps_per_epoch = len(x_train)//batch_size,
              validation_steps = len(x_val)//batch_size,
              batch_size=batch_size, epochs=1, callbacks=[model_checkpoint, model_earlyStopping])


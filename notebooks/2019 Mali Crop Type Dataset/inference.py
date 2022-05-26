import os
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

loaded_model = tf.keras.models.load_model(str(data)+'/unet_model.hdf5') #loading the model
losses=0
accuracies=0
for i in range(len(x_test)):
    losses+= loaded_model.evaluate(np.expand_dims(x_test[i], 0),np.expand_dims(y_test[i], 0), verbose=0)[0]
    accuracies+=loaded_model.evaluate(np.expand_dims(x_test[i], 0),np.expand_dims(y_test[i], 0), verbose=0)[1]
avg_loss=losses/len(x_test)
avg_accuracy=accuracies/len(x_test)
print("The average loss on the test set is: ", avg_loss)
print("The average accuracy on the test set is: ", avg_accuracy*100, "%")
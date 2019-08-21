test_dir = './test_data'


#class dirs
import os

files = os.listdir(os.path.join(test_dir, 'images'))
#preprocessing
batch_size = 32   
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
    rescale=1./255)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical',
    color_mode='rgb') # set as training data


from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from keras import applications
from keras.optimizers import SGD

# build the VGG16 network#layers + optimizer
batch_size = 32
import keras_metrics
metrics= ['categorical_accuracy', keras_metrics.precision(), keras_metrics.recall()]

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
x = base_model.output
x = Flatten(input_shape=base_model.output_shape[1:])(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(8, activation = 'softmax') (x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:19]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=1e-4, momentum=0.9),metrics=metrics)
model.summary()


model.load_weights('xray_horse_studied_weights', by_name=True)

predict = model.predict_generator(test_generator, steps = 1)
print(os.listdir(os.path.join(test_dir, 'images')))
print(predict)
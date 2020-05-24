import os, sys, glob, math, json, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

###### preinitialize globals, change according to command-line args
INPUT_DIR = './'
OUTPUT_DIR = './'
MODEL_NAME = None # default is InceptionV3
FULL_NAME = None
MODEL_TYPE = None # initialize to be None
# Image Size
IMAGE_SIZE = (300,300)
# Batch Size
BATCH_SIZE = 150
# Learning Rate
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100
SUBSET     = 1000000
# # threads
# os.environ['OMP_NUM_THREADS'] = '6'
N_THREADS = 1
# data size
TRAINING_SIZE = 0
VALIDATION_SIZE = 0
# add fully connected layer
ADD_FC = False


def make_model():
    base_model = MODEL_TYPE(weights='imagenet', 
                           include_top=False, 
                           input_shape= IMAGE_SIZE + (3,),                       
                           backend = keras.backend, 
                           layers = keras.layers, 
                           models = keras.models, 
                           utils = keras.utils
                           )

    # add the fully-connected layers
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    if ADD_FC:
        x = Dense(512, activation='relu')(x)
    predictions = Dense(9, activation='softmax', name='predictions')(x) # 9-class prediction

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Train all layers
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(train_generator, val_generator, class_weights):
    model = make_model()
    checkpoint = ModelCheckpoint(
        OUTPUT_DIR+'models/checkpoint-%s.h5' % FULL_NAME,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='auto'
    )
    
    csv_logger = CSVLogger(OUTPUT_DIR+'training_logs/%s.csv' % FULL_NAME, separator=',', append = True)


    if os.path.exists(OUTPUT_DIR+'models/checkpoint-%s.h5' % FULL_NAME):
        print('Restoring checkpoint')
        model.load_weights(OUTPUT_DIR+'models/checkpoint-%s.h5' % FULL_NAME)


    history_train = model.fit_generator(train_generator,
                                        epochs=NUM_EPOCHS,
                                        steps_per_epoch=TRAINING_SIZE // BATCH_SIZE, # mini-batch
                                        validation_data=val_generator,
                                        validation_steps=VALIDATION_SIZE // BATCH_SIZE, # mini-batch
                                        class_weight=class_weights,
                                        callbacks=[checkpoint, csv_logger],
                                        verbose=1)

    # Save model training history
    with open(OUTPUT_DIR+'training_logs/%s.json' % FULL_NAME, 'w') as f:
        json.dump(str(history_train.history), f)
    print("Saved model training history to disk")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type = str, help = "Input directory")
    parser.add_argument("--output_dir", type = str, help = "Output directory")
    parser.add_argument("--model_type", type = str, help = "Type of model used")
    parser.add_argument("--batch_size", type = int, default = 150, help = "Batch size used")
    parser.add_argument("--learning_rate", type = float, default = 1e-5, help = "Learning rate")
    parser.add_argument("--image_size", type = int, default = 300, help = 'Image size of model (an int)?')
    parser.add_argument("--num_epochs", type = int, default = 100, help = 'Number of epochs to train')
    parser.add_argument("--subset", type = int, default = 1000000, help = 'Length of generator')
    # parser.add_argument("--add_FC", type = bool, default = False, help = 'Add fully connected layer before softmax for obtaining features?')
    parser.add_argument("--add_FC", action='store_true', help = 'Add fully connected layer before softmax for obtaining features?')
    parser.add_argument("--n_threads", type = int, default = 1, help = 'Number of threads')
    args = parser.parse_args()
    return args

def main():
    global INPUT_DIR, OUTPUT_DIR, MODEL_NAME, FULL_NAME, IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
    global SUBSET, ADD_FC, MODEL_TYPE, N_THREADS, TRAINING_SIZE, VALIDATION_SIZE
    args = get_args()
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    MODEL_NAME = args.model_type
    # Image Size
    IMAGE_SIZE = (args.image_size,args.image_size)
    # Batch Size
    BATCH_SIZE = args.batch_size
    # Learning Rate
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    SUBSET     = args.subset
    # threads
    N_THREADS = args.n_threads
    os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
    # add FC
    ADD_FC = args.add_FC
    
    if not ADD_FC:
        FULL_NAME = '%s-%s-%s-%s' % (MODEL_NAME, IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE)
    else:
        FULL_NAME = '%s-%s-%s-%s-FC' % (MODEL_NAME, IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE)
    
    print("***************************** ::: Art Connoisseur ::: *****************************")
    print("* INPUT_DIR = ", INPUT_DIR)
    print("* OUTPUT_DIR = ", OUTPUT_DIR)
    print("* MODEL_NAME = ", MODEL_NAME)
    print("* IMAGE_SIZE = ", IMAGE_SIZE)
    print("* BATCH_SIZE = ", BATCH_SIZE)
    print("* LEARNING_RATE = ", LEARNING_RATE)
    print("* NUM_EPOCHS = ", NUM_EPOCHS)
    print("* SUBSET = ",  SUBSET)
    print("* ADD_FC = ", ADD_FC)
    print("* N_THREADS = ", N_THREADS)
    print("***********************************************************************************")
    print("\n\n\n")

    # create output directories
    if not os.path.exists(OUTPUT_DIR+"training_logs/"):
        os.makedirs(OUTPUT_DIR+"training_logs/")
    if not os.path.exists(OUTPUT_DIR+"models/"):
        os.makedirs(OUTPUT_DIR+"models/")

    # determine model type
    if 'VGG' in MODEL_NAME.upper():
        ## VGG-16
        from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
        MODEL_TYPE = VGG16
    elif ("RESNET" in MODEL_NAME.upper())&("V2" not in MODEL_NAME.upper()):
        ## ResNet152
        from tensorflow.keras.applications.resnet import preprocess_input, ResNet152
        MODEL_TYPE = ResNet152
    elif ("RESNET" in MODEL_NAME.upper())&("V2" in MODEL_NAME.upper()):
        ## ResNet152V2
        from tensorflow.keras.applications.resnet_v2 import preprocess_input, ResNet152V2
        MODEL_TYPE = ResNet152V2
    elif 'RESNEXT' in MODEL_NAME.upper():
        ## ResNeXt101
        from keras_applications.resnext import ResNeXt101
        from tensorflow.keras.applications.resnet import preprocess_input
        MODEL_TYPE = ResNeXt101
    elif ("RESNET" not in MODEL_NAME.upper())&("INCEPTION" in MODEL_NAME.upper()):
        ## Inception-V3
        from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3
        MODEL_TYPE = InceptionV3
    elif ("RESNET" in MODEL_NAME.upper())&("INCEPTION" in MODEL_NAME.upper()):
        ## Inception-Resnet V2
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        MODEL_TYPE = InceptionResNetV2
    elif "EFFICIENTNET" in MODEL_NAME.upper():
        ## Efficient Net
        from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
        MODEL_TYPE = EfficientNetB7
    elif "NASNET" in MODEL_NAME.upper():
        ## NASNetLarge
        from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
        MODEL_TYPE = NASNetLarge

    # import data
    print("* loading data ...")
    train_df = pd.read_csv(INPUT_DIR+'train.csv')
    test_df = pd.read_csv(INPUT_DIR+'test.csv')
    train_labels = train_df[['filename','Style']]
    test_labels = test_df[['filename','Style']]
    train_image_names = pd.Series(os.listdir(INPUT_DIR+'data/ALL')).to_frame()
    train_image_names.columns = ['filename']
    train_generator_df = train_labels.merge(train_image_names,on='filename')
    train_generator_df['Style'] = train_generator_df['Style'].astype(str)
    test_generator_df = test_labels.merge(train_image_names,on='filename')
    test_generator_df['Style'] = test_generator_df['Style'].astype(str)

    datagen = ImageDataGenerator(
        fill_mode='nearest',
        horizontal_flip=True,  # randomly flip images
        rescale=1/255.,
        preprocessing_function=preprocess_input,
        data_format=None,
    )

    def create_generator(generator_df):
        global INPUT_DIR
        return datagen.flow_from_dataframe(
            generator_df,
            INPUT_DIR+'data/ALL',
            x_col='filename',
            y_col='Style',
            has_ext=True,  # If image extension is given in x_col
            target_size=IMAGE_SIZE,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=BATCH_SIZE,
            shuffle=True
        )


    train_generator_df = train_generator_df[:SUBSET]
    train_generator = create_generator(train_generator_df)
    test_generator_df  = test_generator_df[:SUBSET]
    val_generator = create_generator(test_generator_df)

    # # balance class weights
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_generator_df['Style']), train_generator_df['Style'])

    # def lr_schedule(epoch):
#     lr = 1e-3
#     if epoch > 120:   lr *= 0.5e-3
#     elif epoch > 100: lr *= 1e-3
#     elif epoch > 50:  lr *= 1e-2
#     elif epoch > 20:  lr *= 1e-1
#     elif epoch > 15:  lr *= 0.2
#     elif epoch > 10:  lr *= 0.5
#     print('Learning rate: ', lr)
#     return lr
    TRAINING_SIZE = train_generator_df.shape[0]
    VALIDATION_SIZE = test_generator_df.shape[0]

    print("* training model ...")
    train_model(train_generator, val_generator, class_weights)


if __name__ == '__main__':
    main()


import datetime
import numpy as np
import img_to_array
import constants as CONST

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K

SPLIT = .7

def save_ela_diff():
    arr = img_to_array.img_to_array_dir("ela/CASIA_V2/diff/", 100, 100)
    arr = np.array(arr)
    np.savetxt('data/ela_diff3.txt', arr, fmt='%d')

def load_ela_diff():
    return np.loadtxt('data/ela_diff3.txt', dtype=int)

def get_timestamp():
    return "[" + str(datetime.datetime.now().time()) + "]"

def perform_ela():
    #
    # get raw data from ELA
    #
    # save_ela_diff()
    print(get_timestamp() + " loading raw data")
    elaDiffArr = load_ela_diff()
    print (get_timestamp() + " done loading raw data")

    #
    # split raw data into Au and Tp
    #
    elaDiffAu = elaDiffArr[:CONST.CASIA2_ELA_DIFF_AU_LAST+1]
    elaDiffTp = elaDiffArr[CONST.CASIA2_ELA_DIFF_TP_FIRST:]

    # shuffle the Au and Tp
    np.random.shuffle(elaDiffAu)
    np.random.shuffle(elaDiffTp)

    # split raw data into training and test sets based on SPLIT
    # iAuSplit = int(elaDiffAu.shape[0]*SPLIT)
    # iTpSplit = int(elaDiffTp.shape[0]*SPLIT)

    # override the split constant to balance CASIA V2 dataset
    # because the Tp set is significantly smaller than the Au set.
    iAuSplit = 4000
    print("iAuSplit: ", iAuSplit)
    trainSet = np.concatenate((elaDiffAu[:iAuSplit], elaDiffTp[:iAuSplit]))
    testSet = np.concatenate((elaDiffAu[iAuSplit:], elaDiffTp[iAuSplit:]))

    print(elaDiffAu.shape, elaDiffTp.shape)
    print(trainSet.shape, testSet.shape)

    # set training labels
    trainLabels = np.concatenate((np.full((iAuSplit, 1), 0), np.full((iAuSplit, 1), 1)))
    testLabels = np.concatenate((np.full((elaDiffAu.shape[0]-iAuSplit, 1), 0), np.full((elaDiffTp.shape[0]-iAuSplit,1),1)))
    np.savetxt('data/ela_diff2_train_labels.txt', trainLabels, fmt='%d')
    np.savetxt('data/ela_diff2_test_labels.txt', testLabels, fmt='%d')
    print(get_timestamp(), "data shape: ", trainSet.shape, testSet.shape)
    print(get_timestamp(), "label shape: ", trainLabels.shape, testLabels.shape)
    if ((trainSet.shape[0] != trainLabels.shape[0]) or (testSet.shape[0] != testLabels.shape[0])):
        print(get_timestamp(), "data and label set mismatch")
        quit()



    # train MLP for ELA data
    model = Sequential()
    model.add(Dense(1024, input_dim=trainSet.shape[1], activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    opt = SGD(lr=0.2, momentum=0.7)
    # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # opt = 'rmsprop'

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(trainSet, trainLabels,
              epochs=100,
              batch_size=128,
              verbose=1,
              validation_data=(testSet, testLabels))

    # model = load_model("data/ela_mlp_2_7_200.h5")

    score = model.evaluate(testSet, testLabels, batch_size=128)
    print(get_timestamp(), "score: ", score)

    model.save("data/ela_mlp_2_7_200_adam.h5")

def perform_cnn():

    # convert the images to 128x128 patches
    arr = img_to_array.keras_img_to_array_dir("casia2/Au/", 128, 128)
    auArr = np.array(arr)
    # np.savetxt('data/cnn_au.txt', arr, fmt='%d')
    arr = img_to_array.keras_img_to_array_dir("casia2/Tp/", 128, 128)
    tpArr = np.array(arr)
    # np.savetxt('data/cnn_tp.txt', arr, fmt='%d')

    # auArr = np.loadtxt('data/cnn_au.txt', dtype=int)
    # tpArr = np.loadtxt('data/cnn_tp.txt', dtype=int)

    # shuffle the Au and Tp
    # np.random.shuffle(auArr)
    # np.random.shuffle(tpArr)

    iAuSplit = 4000
    print("iAuSplit: ", iAuSplit)
    trainSet = np.concatenate((auArr[:iAuSplit], tpArr[:iAuSplit]))
    testSet = np.concatenate((auArr[iAuSplit:], tpArr[iAuSplit:]))

    print(auArr.shape, tpArr.shape)
    print(trainSet.shape, testSet.shape)

    # set training labels
    trainLabels = np.concatenate((np.full((iAuSplit, 1), 0), np.full((iAuSplit, 1), 1)))
    testLabels = np.concatenate((np.full((auArr.shape[0]-iAuSplit, 1), 0), np.full((tpArr.shape[0]-iAuSplit,1),1)))
    np.savetxt('data/cnn_train_labels.txt', trainLabels, fmt='%d')
    np.savetxt('data/cnn_test_labels.txt', testLabels, fmt='%d')
    print(get_timestamp(), "data shape: ", trainSet.shape, testSet.shape)
    print(get_timestamp(), "label shape: ", trainLabels.shape, testLabels.shape)
    if ((trainSet.shape[0] != trainLabels.shape[0]) or (testSet.shape[0] != testLabels.shape[0])):
        print(get_timestamp(), "data and label set mismatch")
        quit()

    # train CNN
    # followed tut at: http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
    # and: https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
    model = Sequential()
    inputShape = (128, 128, 3)

    model.add(Conv2D(30, kernel_size=(5, 5), strides=(2, 2),
                     activation='relu',
                     input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

# perform_ela()
perform_cnn()

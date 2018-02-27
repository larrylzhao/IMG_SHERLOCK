import datetime
import numpy as np
import img_to_array
import constants as CONST

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

SPLIT = .7

def save_ela_diff():
    arr = img_to_array.img_to_array_dir("ela/CASIA_V2/diff/", 100, 100)
    arr = np.array(arr)
    np.savetxt('data/ela_diff2.txt', arr, fmt='%d')

def load_ela_diff():
    return np.loadtxt('data/ela_diff2.txt', dtype=int)

def get_timestamp():
    return "[" + str(datetime.datetime.now().time()) + "]"

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
# np.random.shuffle(elaDiffAu)
# np.random.shuffle(elaDiffTp)

# split raw data into training and test sets based on SPLIT
iAuSplit = int(elaDiffAu.shape[0]*SPLIT)
iTpSplit = int(elaDiffTp.shape[0]*SPLIT)

# override the split constant to balance CASIA V2 dataset
# because the Tp set is significantly smaller than the Au set.
iAuSplit = 4000
print("iAuSplit: ", iAuSplit)
trainSet = np.concatenate((elaDiffAu[:iAuSplit], elaDiffTp[:iAuSplit]))
testSet = np.concatenate((elaDiffAu[iAuSplit:], elaDiffTp[iAuSplit:]))

print(trainSet.shape, testSet.shape)
print(elaDiffAu.shape, elaDiffTp.shape)

# set training labels
trainLabels = np.concatenate((np.full((iAuSplit,1),0), np.full((iAuSplit,1),1)))
testLabels = np.concatenate((np.full((elaDiffAu.shape[0]-iAuSplit,1),0), np.full((elaDiffTp.shape[0]-iAuSplit,1),1)))
# np.savetxt('data/ela_diff2_test_labels.txt', testLabels, fmt='%d')
print(get_timestamp(), "data shape: ", trainSet.shape, testSet.shape)
print(get_timestamp(), "label shape: ", trainLabels.shape, testLabels.shape)
if ((trainSet.shape[0] != trainLabels.shape[0]) or (testSet.shape[0] != testLabels.shape[0])):
    print(get_timestamp(), "data and label set mismatch")
    quit()



# train MLP for ELA data
model = Sequential()
model.add(Dense(5000, input_dim=trainSet.shape[1], activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.2, momentum=0.7)
opt = sgd
# opt = 'rmsprop'

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(trainSet, trainLabels,
          epochs=10,
          batch_size=128)

# model = load_model("data/ela_mlp_2_7_10.h5")

score = model.evaluate(testSet, testLabels, batch_size=128)
print(get_timestamp(), "score: ", score)

# model.save("data/ela_mlp_2_7_10.h5")

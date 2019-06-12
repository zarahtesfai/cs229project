import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm

data = np.load('./data/data.npz')['data']

numTrain = int(data.shape[0] * 0.8)

train = data[: numTrain]
test = data[numTrain :]

trainX = train[:, 0 : -2]
trainY = train[:, -1 :]

testX = test[:, 0 : -2]
testY = test[:, -1 :]


scaler = StandardScaler()
scaler.fit(trainX)

trainXScaled = scaler.tranform(trainX)
testXScaled = scaler.tranform(testX)

pca = PCA(n_components=16)
pca.fit(trainXScaled)

trainXPca = pca.transform(trainXScaled)
testXPca = pca.transform(testXScaled)

svmClassifier = svm.SVC(gamma='scale', verbose=True)
svmClassifier.fit(trainXPca, trainY)

trainPredictions = svmClassifier.predict(trainXPca)
trainAccuracy = np.sum(trainPredictions == trainY) / len(trainY)


testPredictions = svmClassifier.predict(testXPca)
testAccuracy = np.sum(testPredictions == testY) / len(testY)

print('Train accuracy: ', trainAccuracy)
print('Test accuracy: ', testAccuracy)

import pandas as pd
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential
from sklearn import datasets
from sklearn.model_selection import train_test_split

# MNIST 데이터를 내려받는다.
data_dict = datasets.fetch_mldata('MNIST Original')
X = data_dict['data']
Y = data_dict['target']
# 데이터를 학습셋과 시험셋으로 나눈다.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# 데이터를 4차원로 변환한다.
X_train = X_train.reshape((63000, 28, 28, 1))
X_test = X_test.reshape((7000, 28, 28, 1))
Y_train = pd.get_dummies(Y_train).as_matrix()

nb_samples = X_train.shape[0]
nb_classes = Y_train.shape[1]

# 학습 변수
BATCH_SIZE = 16

# 모델 하이퍼 파라미터.
KERNEL_WIDTH = 5
KERNEL_HEIGHT = 5
STRIDE = 1
N_FILTERS = 10

# 모델을 만든다.
model = Sequential()
model.add(Convolution2D(
    nb_filter=N_FILTERS,
    input_shape=(28, 28, 1),
    nb_row=KERNEL_HEIGHT,
    nb_col=KERNEL_WIDTH,
    subsample=(STRIDE, STRIDE))
)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Flatten())
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 학습
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
print('학습을 시작합니다.')
model.fit(X_train, Y_train, nb_epoch=10)

# 학습된 모델을 평가합니다.
probs = model.predict_proba(X_test)
preds = model.predict(X_test)
pred_classes = model.predict_classes(X_test)
true_classes = Y_test
(pred_classes == true_classes).sum()

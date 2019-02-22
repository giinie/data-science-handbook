import pandas as pd
import statsmodels.datasets as sm
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# 데이터셋을 불러온다.
df = sm.elnino.load_pandas().data
X = df.as_matrix()[:, 1:-1]
X = (X - X.min()) / (X.max() - X.min())
Y = df.as_matrix()[:, -1].reshape(61)
# 간단한 전처리
Y = (Y - Y.min()) / (Y.max() - Y.min())
# 학습셋과 시험셋으로 나눈다.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# 모델을 만든다.
model = Sequential()
# LSTM의 메모리는 20차원 벡터
model.add(LSTM(20, input_shape=(11, 1)))
# 예측값은 1차원 (스칼라) 값이다.
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adadelta')
# 학습 수행
model.fit(X_train.reshape((54, 11, 1)), Y_train, nb_epoch=5)
# 학습한 모델을 평가한다.
proba = model.predict_proba(X_test.reshape((7, 11, 1)), batch_size=32)
pred = pd.Series(proba.flatten())
true = pd.Series(Y_test.flatten())
print('예측값과 실젯값의 상관계수 :', pred.corr(true))

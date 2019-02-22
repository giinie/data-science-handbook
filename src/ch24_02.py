import pandas as pd
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 아이리스 데이터셋 준비
ds = datasets.load_iris()
X = ds['data']
Y = pd.get_dummies(ds['target']).as_matrix()

# 학습 및 시험셋으로 나눔
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# 신경망 구현 : 4차원 입력을 받고 은닉 계층은 노드 50개로 이루어짐
# 최종 출력은 3차원이며, 소프트맥스를 활성함수로 이용함
model = Sequential([
    Dense(50, input_dim=4, activation='sigmoid'),
    Dense(3, activation='softmax')
])

# 문제와 활성함수에 맞는 목적함수 설정
model.compile(
    loss='categorical_crossentropy',
    optimizer='adadelta'
)
# 학습 수행
model.fit(X_train, Y_train, nb_epoch=5)
# 학습 결과 평가
proba = model.predict_proba(X_test, batch_size=32)
pred = pd.Series(proba.flatten())
true = pd.Series(Y_test.flatten())
print('상관계수 :', pred.corr(true))

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as datasets
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

matplotlib.rc('font', family="D2Coding")

# 데이터 불러오기
diabetes = datasets.load_diabetes()
X, Y = normalize(diabetes['data']), diabetes['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)
linear = LinearRegression()
linear.fit(X_train, Y_train)
preds_linear = linear.predict(X_test)
corr_linear = round(pd.Series(preds_linear).corr(pd.Series(Y_test)), 3)
rsquared_linear = r2_score(Y_test, preds_linear)

print('선형 계수 :', linear.coef_)
plt.scatter(preds_linear, Y_test)
plt.title('선형 회귀 결과. 상관 관계 = %f R^2 점수 = %f' % (corr_linear, rsquared_linear))
plt.xlabel('예측값')
plt.ylabel('실젯값')
# 비교를 위한 x=y 라인 추가
plt.plot(Y_test, Y_test, 'k--')
plt.show()

lasso = Lasso()
lasso.fit(X_train, Y_train)
preds_lasso = lasso.predict(X_test)
corr_lasso = round(pd.Series(preds_lasso).corr(pd.Series(Y_test)), 3)
rsquared_lasso = round(r2_score(Y_test, preds_lasso), 3)

print('라소 계수 :', lasso.coef_)
plt.scatter(preds_lasso, Y_test)
plt.title('라소 회귀 결과. 상관 관계 = %f R^2 점수 = %f' % (corr_lasso, rsquared_lasso))
plt.xlabel('예측값')
plt.ylabel('실젯값')
# 비교를 위한 x=y 라인 추가
plt.plot(Y_test, Y_test, 'k--')
plt.show()

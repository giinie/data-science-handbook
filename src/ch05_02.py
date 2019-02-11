import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import sklearn.datasets as ds

matplotlib.rc('font', family="D2Coding")

# 팬더스 데이터프레임 생성
bs = ds.load_boston()
df = pd.DataFrame(bs.data, columns=bs.feature_names)
df['MEDV'] = bs.target

# 일반적인 산포도
# df.plot(x='CRIM', y='MEDV', kind='scatter')
# plt.title('일반축에 나타낸 범죄 발생률')
# plt.show()

# 로그를 적용
df.plot(x='CRIM', y='MEDV', kind='scatter', logx=True)
plt.title('Crime rate on logarithmic axis')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family="D2Coding")

z = np.zeros(1000)
x = np.random.exponential(size=1000)
data = np.concatenate([z, x])
pd.Series(data).hist(bins=100)
plt.title('0에 대부분의 값이 몰려 있음')
D = pd.Series(data)
X = pd.Series(x)
D.hist(bins=100)
plt.show()

(D > 0).value_counts().rename({True: '> 0', False: '= 0'}).plot(kind='pie')
plt.title('0인 값의 비율')
plt.show()
X.hist(bins=100)

plt.title('0보다 큰 값의 히스토그램')
plt.show()

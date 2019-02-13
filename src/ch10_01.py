import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from numpy import random
from scipy.spatial import distance

matplotlib.rc('font', family="D2Coding")

d = 2
data = random.uniform(size=d * 1000).reshape((1000, d))
distance = distance.cdist(data, data)
pd.Series(distance.reshape(1000000)).hist(bins=50)
plt.title('%i차원 공간에서 점 간 거리' % d)
plt.show()

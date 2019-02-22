import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family="D2Coding")

np.random.seed(10)
means = []
sum = 0.0

for i in range(1, 1000):
    # 파레토 분포에서 표본을 추출한다.
    sum += np.random.pareto(1)
    means.append(sum / i)

plt.plot(means)
plt.title('N개 샘플의 평균')
plt.xlabel('N')
plt.show()

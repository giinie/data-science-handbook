import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', family="D2Coding")

times, d = [], {}
for i in range(10000000):
    start = time.time()
    d[i] = i
    stop = time.time()
    times.append(stop - start)

plt.plot(times)
plt.xlabel('딕셔너리 길이')
plt.ylabel('성분을 추가하는데 걸리는 시간(초)')
plt.show()

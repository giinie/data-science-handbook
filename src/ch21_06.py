import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', family="D2Coding")


def time_numpy(n):
    a = np.arange(n)
    start = time.time()
    bigger = a + 1
    stop = time.time()
    return stop - start


def time_python(n):
    l = range(n)
    start = time.time()
    bigger = [x + 1 for x in l]
    stop = time.time()
    return stop - start


cn_trials = 10
ns = range(20, 500)
ratios = []
for n in ns:
    python_total = sum([time_python(n)
                        for _ in range(n_trials)])
    numpy_total = sum([time_numpy(n)
                       for _ in range(n_trials)])
    ratios.append(python_total / numpy_total)

plt.plot(ns, ratios)
plt.xlabel('배열의 길이')
plt.ylabel('파이썬/넘파이 비율')
plt.title('소요 시간 비교')
plt.show()

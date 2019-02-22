import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', family="D2Coding")


def duplicates_On2(lst):
    ct = 0
    for x in lst:  # 모든 값의 갯수를 한 번씩 센다.
        if lst.count(x) > 1:
            ct += 1
    return ct


def duplicates_On(lst):
    cts = {}
    for x in lst:  # 딕셔너리를 이용한다.
        if x in cts:
            cts[x] += 1
        else:
            cts[x] = 1

    counts_above_1 = [ct for x, ct in cts.items()
                      if ct > 1]
    return sum(counts_above_1)


def timeit(func, arg):  # 수행 시간을 재는 함수
    start = time.time()
    func(arg)
    stop = time.time()
    return stop - start


times_On, times_On2 = [], []
ns = range(50)
for n in ns:
    lst = list(np.random.uniform(size=n))
    times_On2.append(timeit(duplicates_On2, lst))
    times_On.append(timeit(duplicates_On, lst))

plt.plot(times_On2, '--', label='O(n^2)')
plt.plot(times_On, label='O(n)')
plt.xlabel('배열의 길이')
plt.ylabel('시간(초)')
plt.title('중복 값을 세는 데 걸리는 시간')
plt.legend(loc='upper left')
plt.show()

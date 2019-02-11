import matplotlib
from matplotlib import pyplot as plt
from statsmodels.datasets import co2

matplotlib.rc('font', family="D2Coding")


dta = co2.load_pandas().data
dta.plot()
plt.title('이산화탄소 농도')
plt.ylabel('PPM')
plt.show()

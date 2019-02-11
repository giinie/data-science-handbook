import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import sklearn.datasets

matplotlib.rc('font', family="D2Coding")


def get_iris_df():
    ds = sklearn.datasets.load_iris()
    df = pd.DataFrame(ds['data'], columns=ds['feature_names'])
    code_species_map = dict(zip(range(3), ds['target_names']))
    df['species'] = [code_species_map[c] for c in ds['target']]
    return df


df = get_iris_df()

# Pie Chart
# sums_by_species = df.groupby('species').sum()
# var = 'sepal width (cm)'
# sums_by_species[var].plot(kind='pie', fontsize=20)
# plt.ylabel(var, horizontalalignment='left')
# plt.title('꽃받침 너비로 분류한 붓꽃', fontsize=25)
# plt.savefig('iris_pie_for_one_variable.png')
# plt.close()

# sums_by_species = df.groupby('species').sum()
# sums_by_species.plot(kind='pie', subplots=True, layout=(2, 2), legend=False)
# plt.title('종에 따른 전체 측정값')
# plt.savefig('iris_pie_for_each_variable.png')
# plt.close()

# Bar Chart
# sums_by_species = df.groupby('species').sum()
# var = 'sepal width (cm)'
# sums_by_species[var].plot(kind='bar', fontsize=15, rot=30)
#
# plt.title('꽃받침 너비로 분류한 붓꽃', fontsize=20)
# plt.savefig('iris_bar_for_one_variable.png')
# plt.close()
#
# sums_by_species = df.groupby('species').sum()
# sums_by_species.plot(kind='bar', subplots=True, fontsize=12)
# plt.suptitle('종에 따른 전체 측정값')
# plt.savefig('iris_bar_for_each_variable.png')
# plt.close()

# Histogram
# df.plot(kind='hist', subplots=True, layout=(2, 2))
# plt.suptitle('붓꽃 히스토그램', fontsize=20)
# plt.show()

# for spec in df['species'].unique():
#     forspec = df[df['species'] == spec]
#     forspec['petal length (cm)'].plot(kind='hist', alpha=0.4, label=spec)
#
# plt.legend(loc='upper right')
# plt.suptitle('종에 따른 꽃잎 길이')
# plt.show()

# 대푯값
# col = df['petal length (cm)']
# average = col.mean()
# std = col.std()
# median = col.quantile(0.5)
# percentile25 = col.quantile(0.25)
# percentile75 = col.quantile(0.75)
# clean_avg = col[(col > percentile25) & (col < percentile75)].mean()
# clean_std = col[(col > percentile25) & (col < percentile75)].std()
#
# print('average :', average)
# print('std :', std)
# print('clean_avg :', clean_avg)
# print('clean_std :', clean_std)
# print('median :', median)
# print('per25 :', percentile25)
# print('per75 :', percentile75)

# Box Chart
# col = 'sepal length (cm)'
# df['ind'] = pd.Series(df.index).apply(lambda i: i % 50)
# df.pivot('ind', 'species')[col].plot(kind='box')
# plt.show()

# 산포도
# df.plot(kind='scatter', x='sepal length (cm)', y='sepal width (cm)')
# plt.title('길이 대 너비')
# plt.show()
# color = ['r', 'g', 'b']
# markers = ['.', '*', '^']
# fig, ax = plt.subplots(1, 1)
# for i, spec in enumerate(df['species'].unique()):
#     ddf = df[df['species'] == spec]
#     ddf.plot(kind='scatter',
#              x='sepal width (cm)', y='sepal length (cm)',
#              alpha=0.5, s=10*(i+1), ax=ax,
#              color=color[i], marker=markers[i], label=spec)
#
# plt.legend()
# plt.show()

# 산포 행렬
# from pandas.plotting import scatter_matrix
#
# scatter_matrix(df)
# plt.show()

# 히트맵(heatmap)
# df.plot(kind='hexbin', x='sepal width (cm)', y='sepal length (cm)')
# plt.show()

# 상관관계(Correlation)
# 케디의 법칙 :
# A와 B의 상관관계가 높으면 대체로 둘 중 하나가 다른 하나의 원인이 아니라
# 어떤 공통의 외부 요인 C가 존재한다.
print('corr', df['sepal width (cm)'].corr(df['sepal length (cm)']))
# 피어슨 상관관계
print('pearson :', df['sepal width (cm)'].corr(df['sepal length (cm)'], method='pearson'))
# 순서형 상관관계
print('spearman :', df['sepal width (cm)'].corr(df['sepal length (cm)'], method='spearman'))
print('kendall :', df['sepal width (cm)'].corr(df['sepal length (cm)'], method='kendall'))

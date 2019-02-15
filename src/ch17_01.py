import json
import urllib.request as rq

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import statsmodels.api as sm

matplotlib.rc('font', family="D2Coding")

START_DATE = '20161010'
END_DATE = '20181012'
WINDOW_SIZE = 7
TOPIC = 'Cat'
URL_TEMPLATE = ("https://wikimedia.org/api/rest_v1"
                "/metrics/pageviews/per-article/en.wikipedia/all-access"
                "/all-agents/%s/daily/%s/%s")

print(URL_TEMPLATE % (TOPIC, START_DATE, END_DATE))


# 조회수를 불러오는 함수
def get_time_series(topic, start, end):
    url = URL_TEMPLATE % (topic, start, end)
    json_data = rq.urlopen(url).read().decode('utf-8')
    data = json.loads(json_data)
    times = [rec['timestamp'] for rec in data['items']]
    values = [rec['views'] for rec in data['items']]
    times_formatted = pd.Series(times).map(
        lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:8]
    )
    time_index = times_formatted.astype('datetime64')
    return pd.DataFrame({'views': values}, index=time_index)


# 선형 회귀 모델을 학습하는 함수
def line_slope(ss):
    X = np.arange(len(ss)).reshape((len(ss), 1))
    linear.fit(X, ss)
    return linear.coef_


# 선형 회귀 모델을 하나 만든다.
# 이제 모델에 다양한 데이터를 계속 적용한다.
linear = sklearn.linear_model.LinearRegression()

df = get_time_series(TOPIC, START_DATE, END_DATE)

# 시계열 데이터 시각화
df['views'].plot()
plt.title('날짜별 조회수')
plt.show()

# 백분위를 기준으로 이상치를 제거한다.
max_views = df['views'].quantile(0.95)
df.views[df.views > max_views] = max_views

# 7일을 주기로 데이터 분석
decomp = sm.tsa.seasonal_decompose(df['views'].values, freq=7)
decomp.plot()
plt.suptitle('조회수 분석 결과')
plt.show()

# 날짜별로 과거 일주일의 평균, 최댓값, 최솟값 등 다양한 특징을 추출, 저장
df['mean_1week'] = df['views'].rolling(WINDOW_SIZE).mean()
df['max_1week'] = df['views'].rolling(WINDOW_SIZE).max()
df['min_1week'] = df['views'].rolling(WINDOW_SIZE).min()
df['slope'] = df['views'].rolling(WINDOW_SIZE).apply(line_slope)
df['total_views_week'] = df['views'].rolling(WINDOW_SIZE).sum()
df['day_of_week'] = df.index.astype(int) % 7
day_of_week_cols = pd.get_dummies(df['day_of_week'])
df = pd.concat([df, day_of_week_cols], axis=1)

# 예측값 준비
df['total_views_next_week'] = list(df['total_views_week'][WINDOW_SIZE:]) + [np.nan for _ in range(WINDOW_SIZE)]
INDEP_VARS = ['mean_1week', 'max_1week', 'min_1week', 'slope'] + list(range(6))
DEP_VAR = 'total_views_next_week'

n_records = df.dropna().shape[0]
test_data = df.dropna()[:n_records // 2]
train_data = df.dropna()[n_records // 2:]

linear.fit(train_data[INDEP_VARS], train_data[DEP_VAR])
test_preds_array = linear.predict(test_data[INDEP_VARS])
test_preds = pd.Series(test_preds_array, index=test_data.index)

print('예측값과 정답의 상관계수 :', test_data[DEP_VAR].corr(test_preds))

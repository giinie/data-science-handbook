import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

matplotlib.rc('font', family="D2Coding")

# 데이터 불러오기
faces_data = datasets.fetch_olivetti_faces()
person_ids, image_array = faces_data['target'], faces_data.images
# (64, 64) 크기의 데이터를 (4096, )으로 변환 (64 * 64 = 4096)
X = image_array.reshape((len(person_ids), 64 * 64))

# 군집화 알고리즘 실행
print('** 원본 데이터 군집화 결과')
model = KMeans(n_clusters=40)
model.fit(X)

print('군집화 성능 :', silhouette_score(X, model.labels_))
print('얼굴 일치율 :', adjusted_rand_score(model.labels_, person_ids))

# 주성분 분석 수행 (PCA, principal component analysis)
print('** 주성분 분석 후 군집화 결과')
pca = PCA(15)
pca.fit(X)
X_reduced = pca.transform(X)
model_reduced = KMeans(n_clusters=40)
model_reduced.fit(X_reduced)
labels_reduced = model_reduced.labels_
print('군집화 성능 :', silhouette_score(X_reduced, model_reduced.labels_))
print('얼굴 일치율 :', adjusted_rand_score(model_reduced.labels_, person_ids))

# 이미지를 직접 보며 결과를 확인
sample_face = image_array[0, :, :]
plt.imshow(sample_face)
plt.title('얼굴 예시')
plt.show()

# eigenface 0 (첫 번째 주성분)
eigenface0 = pca.components_[0, :].reshape((64, 64))
plt.imshow(eigenface0)
plt.title('아이겐페이스 0')
plt.show()
eigenface1 = pca.components_[1, :].reshape((64, 64))
plt.imshow(eigenface1)
plt.title('아이겐페이스 1')
plt.show()
# 스크리 도표 (skree plot)
pd.Series(pca.explained_variance_ratio_).plot()
plt.title('아이겐페이스 스크리 도표')
plt.show()

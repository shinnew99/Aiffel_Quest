# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 신유진
- 리뷰어 : 이효겸


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네 이하 주석 참조
- [O] 코드가 에러를 유발할 가능성이 없나요?
  > 네 이하 주석 참조
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네 이해하고 작성한것으로 판단됨 근거는 이하 주석 참조
- [O] 코드가 간결한가요?
  > 단락별 코드가 간결하게 분리되어있음

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import warnings
warnings.filterwarnings("ignore")

import os
from os.path import join

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb
import missingno as msno
import sklearn

print("xgboost version:", xgb.__version__)
print("lightgbm version:", lgb.__version__)
print("missingno version:", msno.__version__)
print("sklearn version:", sklearn.__version__)
xgboost version: 1.4.2
lightgbm version: 3.3.0
missingno version: 0.5.0
sklearn version: 1.0

# 데이터 로드
data_dir = os.getenv('HOME')+'/aiffel/kaggle_kakr_housing/data'

train_data_path = join(data_dir, 'train.csv')
test_data_path = join(data_dir, 'test.csv')


train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# 데이트 타입을 앞에 년월만 가져와서 인트로 변경
train_df['date2'] = train_df['date'].apply(lambda i: str(i[:6])).astype(int)
test_df['date2'] = test_df['date'].apply(lambda i: str(i[:6])).astype(int)
# test_df = test_df.drop('id')
train_df.head()

# 필요 없는 값인 아이디와 날짜 삭제
test_id = test_df['id']
test_df = test_df.drop(columns=['id', 'date'])
test_df.head()

# 데이터 셋 확인
print(train_df.shape, test_df.shape)
(15035, 22) (6468, 19)

# 결측치 확인
train_df.isnull().sum()
test_df.isnull().sum()

# 상관관계 확인
train_corr = train_df.corr()
train_corr

plt.figure(figsize=(10, 7))
sns.heatmap(train_corr, annot=True, fmt=".2f", cmap="Blues")

# 학습데이터에서 타겟값인 price와 필요 없는 값 삭제
train = train_df.drop(columns=['date', 'price', 'id'])
y = train_df['price']
train.head()

# rmse 오류 함수
def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
#         mean_suqred_error(np.expm1(y_test). np.expm1(y_pred))

# 랜덤 스테이트 값 지정
random_state = 2023

# 3개의 모델을 준비
gboost = GradientBoostingRegressor(random_state=random_state)
xgboost = xgb.XGBRegressor(n_estimators=500, learning_rate=0.2, max_depth=4, random_state=random_state)
lightgbm = lgb.LGBMRegressor(random_state=random_state)
models = [{'model':gboost, 'name':'GradientBoosting'}, {'model':xgboost, 'name':'XGBoost'},
          {'model':lightgbm, 'name':'LightGBM'}]

# 스코어 함수 
def get_cv_score(models, x, y):
    kfold = KFold(n_splits=5).get_n_splits(x.values)
    for m in models:
        CV_score = np.mean(cross_val_score(m['model'], X=x.values, y=y, cv=kfold))
        print(f"Model: {m['name']}, CV score:{CV_score:.4f}")

# 모델 스코어 확인
get_cv_score(models, train, y)
Model: GradientBoosting, CV score:0.8608
Model: XGBoost, CV score:0.8961
Model: LightGBM, CV score:0.8818

# 모델 학습과 결과 값의 평균 리턴 함수
def AveragingBlending(models, x, y, sub_x):
    for m in models : 
        m['model'].fit(x.values, y)
    predictions = np.column_stack([
        m['model'].predict(sub_x.values) for m in models
    ])
    return np.mean(predictions, axis=1)

# 모델로 부터 학습 및 결과값 추출
y_pred = AveragingBlending(models, train, y, test_df)
sub = pd.DataFrame(data={'id':test_id,'price':y_pred})
sub.to_csv('submission.csv', index=False)

# 모델의 하이퍼 파라미터 값을 찾기 위한 함수
def my_GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5):
    # 1. GridSearchCV 모델로 `model`을 초기화합니다.
    # model = LGBMRegressor(random_state=random_state)
    grid_model = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_squared_error', \
                            cv=5, verbose=1, n_jobs=5)
    # 2. 모델을 fitting 합니다.
    grid_model.fit(train, y)

    # 3. params, score에 각 조합에 대한 결과를 저장합니다.
    params = grid_model.cv_results_['params']
    score = grid_model.cv_results_['mean_test_score']

    # 데이터 프레임 생성
    results = pd.DataFrame(params)
    results['score'] = score

    # RMSLE 값 계산 후 정렬
    # 4. 데이터 프레임을 생성하고, RMSLE 값을 추가한 후 점수가 높은 순서로 정렬한 `results`를 반환합니다.
    results['RMSLE'] = np.sqrt(-1*results['score'])
    results = results.sort_values('RMSLE')

    return results

# 파라미터 그리드
param_grid = {
    'n_estimators': [500], # n_estimator를 크게 잡는 경우가 많았다
    'max_depth': [10],  # depth를 굳이 깊이 잡는 것 같지 않았다
    'learning_rate': [0.1] #learning rate은 0.1로 선택
}

# 모델 준비
model = GradientBoostingRegressor(random_state = random_state)

# 그리드로 파라미터값 찾기 
my_GridSearch(model, train, y, param_grid, verbose=1, n_jobs=1)
Fitting 5 folds for each of 1 candidates, totalling 5 fits


```

# 참고 링크 및 코드 개선
```python
수정할 코드는 없는 것으로 보임
```
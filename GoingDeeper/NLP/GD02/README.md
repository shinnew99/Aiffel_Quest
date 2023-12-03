# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 신유진
- 리뷰어 : 홍서이


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [o] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > n_words를 여러개로 설정하고, 로이터 뉴스 데이터를 tfidf vector로 만든ㄷ 뒤 이를 머신러닝 모델과 딥러닝 모델에 훈련시켜서 문제를 해결하였다.
- [o] 주석을 보고 작성자의 코드가 이해되었나요?
  > 작동 시간과 코드에 대한 설명이 주석으로 달려 있어서 이해하기 쉬웠다.

```Python
# 12-13분

# 각 분류기 정의
logreg = LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs')
cnb = ComplementNB()
gboost = GradientBoostingClassifier()

# VotingClassifier 생성 (soft voting)
voting_classifier = VotingClassifier(estimators=[
    ('logreg', logreg),
    ('cnb', cnb),
    ('gboost', gboost)
], voting='soft')

# 모델 훈련
voting_classifier.fit(tfidfv, y_train)
```

- [o] 코드가 에러를 유발할 가능성이 없나요?
  > 모델별, 데이터별로 변수를 다르게 설정해서 잘못된 모델, 데이터에 대해 처리되는 것을 방지하였다.
- [o] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네
- [-] 코드가 간결한가요?
  > 텍스트 길이 분포 및 클래스 분포의 경우 n_words의 영향을 받지 않으므로 여러번 그릴 필요가 없어 보입니다. 그리고 노트북 하단에 적어주신대로 함수화를 진행하면 코드가 더 간결해질 것 같습니다.


# 참고 링크 및 코드 개선
- 저희 팀원 이태훈 님 프로젝트 파일 링크입니다.
- [링크](https://github.com/git-ThLee/AIFFEL_Online_5th_thlee/blob/main/quest_12_Going%20Deeper(NLP)_2/Quest.ipynb)
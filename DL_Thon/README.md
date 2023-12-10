0은 혼자 해보다가 실패한것
<br>1은 어떻게든 모델 학습시킨것, 정확도는 매우 낮음
<br>1base_model은 exploration노드 8을 참고해서 만든것
<br>1LSTM은 서이님의 코드 참고한것

<br>
<br>2는 갓한준님의 코드를 가지고 와서 optimizer만 조금씩 손봤다. 
<br>RectifiedAdam_BERT_classification은 말그대로 optimizer를 RectifiedAdam 함수로 바꾸고 그에 해당하는 패키지도 호출, 그리고 그 뒤에 따라오는 각종 하이퍼파라미터도 튜닝했다(코드참고). 또, epoch을 각 5번(patience=10), 20번(patience=10), 10번(patience=2)으로 학습시켰을때 test accuracy값이 각각 0.9125, 0.875, 0.91이 나왔다.

<br>
<br>Adagrad_BERT_Classification으로는 Adagrad함수로 바꿨고, epoch 5, patience=10 로 나왔다. 

# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 신유진
- 리뷰어 : 이효겸


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 데이터셋 생성 과정이 잘 되어 있음
    - 하단 부분에 모델 결과에 대한 시각화가 되어 있어 해당 부분에 대한 모델 학습 진행이 정상적으로 되어있음을 알 수 있음
    

- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
  주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 전체적인 흐름이 잘 구성되어 있으며 markdown문법으로 해당 코드들이 무슨 작동을 위한 코드인지 설명 되어 있음
    - 주석도 잘 쓰여져 있음
  
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
  ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 디버깅 문제는 없었으나 메모리 경량 시도 코드가 기제되어 있음
   
    enc_tokens = np.memmap(filename='enc_tokens.memmap', mode='w+', dtype=np.int32, shape=(total, n_seq))
    segments = np.memmap(filename='segments.memmap', mode='w+', dtype=np.int32, shape=(total, n_seq))
    labels_nsp = np.memmap(filename='labels_nsp.memmap', mode='w+', dtype=np.int32, shape=(total,))
    labels_mlm = np.memmap(filename='labels_mlm.memmap', mode='w+', dtype=np.int32, shape=(total, n_seq))

  
- [X]  **4. 회고를 잘 작성했나요?**
    - 결과물의 대한 분석과 배운점과 느낀점이 제대로 적혀 있음
    
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 모델과 굳이 함수화를 하지 않아도 되는 것들에 대한 것들을 제외하고 제대로 함수화하고 클래스로 작성 되어 있음


# 참고 링크 및 코드 개선
```
해결 사항에 대한 내용(데이터셋 생성, 모델 등)을 포함한 전반적인 코드 자체에 대한 질문들이라 
일부분의 코드를 굳이 붙일 필요성을 느끼지 못하여 코드는 첨부하지 않음 
```
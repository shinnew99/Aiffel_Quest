# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 신유진
- 리뷰어 : 박혜원


# PRT(Peer Review Template)
- [ㅇ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 챗봇 훈련데이터 전처리 과정이 체계적으로 진행되었는가?
      " Step 2. 데이터 정제 , Step 3. 데이터 토큰화 , Step4. Augmentation, Step5. 데이터 벡터화 " 해당하는 코드 부분을 확인 했을 때, 훈련 데이터를 위한 전처리가 잘 진행 된 것 확인 하였습니다. 
    - transformer 모델을 활용한 챗봇 모델이 과적합을 피해 안정적으로 훈련되었는가?
      과적합을 확인할 수 있는 부분은 포함되어있지 않았지만, 두 가지 Parameter 조합에 대하여 학습을 진행하여 비교하였습니다.  
    - 챗봇이 사용자의 질문에 그럴듯한 형태로 답하는 사례가 있는가? 결과가 연계성은 있으나, 챗봇의 성능은 다소 떨어지는 것을 확인 하였습니다. 
      Translations
      > 1
    Q: 지루하다, 놀러가고 싶어.
    A: 지루 지루 낮잠 지루

      > 2
    Q: 오늘 일찍 일어났더니 피곤하다.
    A: 오늘 피곤
    
- [ㅇ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 주석이 아주 상세하게 작성되어 있던 것은 아니지만, 기능 및 Step 단위로 셀이 잘 구분되어있으며 마크다운으로 구분을 잘 해뒀다고 생각합니다. 
  
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” ”새로운 시도 또는 추가 실험을 수행”해봤나요?**

- [X]  **4. 회고를 잘 작성했나요?**

- [ㅇ]  **5. 코드가 간결하고 효율적인가요?**
  - 네. 
    아래와 거의 모든 과정을 함수화 하여 작성해주셨습니다. 

    ```
    def build_corpus(src_data, tgt_data):
    
    mecab = Mecab()
    
    def get_morphs(s):
        return mecab.morphs(s)
    
    mecab_src_corpus = list(map(get_morphs, src_data))
    mecab_tgt_corpus = list(map(get_morphs, src_data))
    
    mecab_num_tokens = [len(s) for s in mecab_src_corpus] + [len(s) for s in mecab_tgt_corpus]
    
    # 최대 길이를 (평균 + 2*표준편차)로 계산
    max_len = round(np.mean(mecab_num_tokens) + 2 * np.std(mecab_num_tokens))
    print(f'max_len : {max_len}')
    
    src_corpus, tgt_corpus = [], []
    for q, a in zip(mecab_src_corpus, mecab_tgt_corpus):
        if len(q) <= max_len and len(a) <= max_len:
            if q not in src_corpus and a not in tgt_corpus:
                src_corpus.append(q)
                tgt_corpus.append(a)
    
    return src_corpus, tgt_corpus

    ```
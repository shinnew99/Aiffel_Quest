# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 신유진
- 리뷰어 : 조대호


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [Δ] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 코드가 정상적으로 작동했다. 마지막 문제인 " 한국어 입력문장에 대해 한국어로 답변하는 함수를 구현"에서 문장이 아닌 단어 하나만 출력 되었다.

입력 : 뭐하고 있어?
출력 : 궁금하
궁금하
입력 : 이름이 뭐야?
출력 : 자

- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네 주석을 보고 코드를 이해하는데 도움이 되었습니다.
- [O] 코드가 에러를 유발할 가능성이 없나요?
  > 네 코드를 본 결과 에러가 발생하지 않았습니다.
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네 주석을 달아주어서 코드를 제대로 이해하고 작성한것 같습니다.
- [O] 코드가 간결한가요?
  > 네. 모델 부분을 함수로 처리하여 간결하게 작성했습니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

#마지막 결과값을 출력할 때 한 단어로만 출력되는 문제에 대한 해결
#Step5 모델 평가하기 부분에서 decoder_inference 함수 부분의 output_sequence 부분을 for문안으로 집어 넣어야 합니다.


def decoder_inference(sentence):
    sentence = preprocess_sentence(sentence)

  # 입력된 문장을 정수 인코딩 후, 시작 토큰과 종료 토큰을 앞뒤로 추가.
  # ex) Where have you been? → [[8331   86   30    5 1059    7 8332]]
    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  # 디코더의 현재까지의 예측한 출력 시퀀스가 지속적으로 저장되는 변수.
  # 처음에는 예측한 내용이 없음으로 시작 토큰만 별도 저장. ex) 8331
    output_sequence = tf.expand_dims(START_TOKEN, 0)

  # 디코더의 인퍼런스 단계
    for i in range(MAX_LENGTH):
    # 디코더는 최대 MAX_LENGTH의 길이만큼 다음 단어 예측을 반복합니다.
        predictions = model(inputs=[sentence, output_sequence], training=False)
        predictions = predictions[:, -1:, :]

    # 현재 예측한 단어의 정수
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 만약 현재 예측한 단어가 종료 토큰이라면 for문을 종료
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

    # 예측한 단어들은 지속적으로 output_sequence에 추가됩니다.
    # 이 output_sequence는 다시 디코더의 입력이 됩니다.
        output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)

    return tf.squeeze(output_sequence, axis=0)


#문장부호, 영단어, 숫자등을 제외하면 달라졌을까? 에 대해서
#저희가 (한글), (한글,영어),(한글,숫자)이렇게 토크나이징을 하여 모델돌려보고 내린 결론은
#영단어, 숫자에 대한 데이터가 부족하여 큰 차이는 발생하지 않았습니다.
```

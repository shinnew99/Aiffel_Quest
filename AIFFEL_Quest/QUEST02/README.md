# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 신유진
- 리뷰어 : 조준규


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 코드블록 별로 어떤 역할을 하는지 잘 알 수 있었습니다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > Import가 잘 되어있고 함수도 알맞게 작성해서 에러가 생길 부분은 없었습니다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 각 코드블록 별로 동작을 확인하기 위한 디버깅 코드들로 보아 제대로 이해했다고 생각합니다.
- [X] 코드가 간결한가요?
  > 구현하고자 하는 바에 알맞게 간결하게 작성되었습니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 사칙 연산 계산기
class calculator:
    # 예) init의 역할과 각 매서드의 의미를 서술
    def __init__(self, first, second):
        self.first = first
        self.second = second
    
    # 예) 덧셈과 연산 작동 방식에 대한 서술
    def add(self):
        result = self.first + self.second
        return result

a = float(input('첫번째 값을 입력하세요.')) 
b = float(input('두번째 값을 입력하세요.')) 
c = calculator(a, b)
print('덧셈', c.add()) 
```

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```

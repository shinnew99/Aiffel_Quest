# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 신유진
- 리뷰어 : 이효겸


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 모델이 정상적으로 작동됨
    - 파인튜닝을 시도하였고 90% 정확도를 달성한 로그가 남아 있음
    - 버켓팅 적용 
```
model2 = Classifier(
    model_name,
    dataset,
    TrainingArguments(
        output_dir, 
        evaluation_strategy="epoch",
        learning_rate=2e-5,   
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=2,
        num_train_epochs=1,
        warmup_steps=1000, 
        weight_decay=0.01,                 
        fp16=True,
    ))
```


```
model3 = Classifier(
    model_name,
    dataset,
    TrainingArguments(
        output_dir, 
        evaluation_strategy="epoch",
        learning_rate=2e-5,   
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
#         gradient_accumulation_steps=2,
#         eval_accumulation_steps=2,
        num_train_epochs=1,
        warmup_steps=1000, 
        weight_decay=0.01,                 
        fp16=True,
#         group_by_length=True
    ))
```
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
  주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드들 전에 어떤 내용을 진행하는 지에 대한 설명이 잘 되어 있음
  
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
  ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 튜닝 및 버켓팅을 적용하여 기록함
    
  
- [X]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제에 따른 회고록이 하단에 잘 작성 되어 있음
    
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 한번에 수행 할 수 있도록 클래스를 잘 활용하였음
```
class DataSet():
class Classifier():
```
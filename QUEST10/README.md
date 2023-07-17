# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 신유진
- 리뷰어 : 김성진


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 네, 정상 동작합니다.
  > 블러처리뿐만 아니라 다른 배경과의 합성도 진행하였습니다.
  > 찾은 문제점을 해결하는 방안이 작성되었습니다.
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네, 필요한 곳에 주석이 달려있고, 이해가 잘 됩니다.
  > 각 파트별로 제목이 작성되어 있습니다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 네, 없습니다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네, 코드 흐름상 코드를 잘 이해하고 작성되었습니다.
- [X] 코드가 간결한가요?
  > 네, 간결합니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.

- 파트별 제목

```python
#### 배경바꾸기
```

- 주석처리, 배경과의 합성

```python
# 배경과 im_orig(회색고양이) 합성
img_concat = np.where(img_mask_color == 255, resized_img_bg, img_orig)

plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()
```

- 사진에서 문제점 찾기

```
1. 휴대폰을 배경으로 인식 <br>
2. 머리카락, 옷 테두리의 불분명한 경계 <br> 
3. 옷을 배경으로 인식해서 blur 처리함 <br>
```

- 해결방법을 제안해 보기

```
1. 핸드폰을 사람으로 인식:
    - label_names에는 휴대폰이라는 라벨이 없는 것을 확인했다. 그렇다고 모든 물체들을 다 라벨링 할 수 없고, 특히 업로드 한 사진에서 사람이 핸드폰을 들고 있는 것을 확인 할 수 있듯이, 손끝에 닿은 물건마저 사람이라고 인식시키면 될 거 같다. 이때 오른쪽 사람 손끝에 식물마저도 사람으로 인식할 수 있으니, 반드시 손가락 까지 사람으로 인식하게 해야한다. segementation하는 과정은 대상의 중심을 핸드폰으로 두고, 핸드폰 자체를 사람이라고 인식하면 해결 할 수 있지 않을까?  <br> <br>

2. 불분명한 경계:
    - 사진에서 보면 경계부분이 사진포토샵에서 누끼를 따듯이 자연스럽지는 않다. 배경부분이 일부 덜 blur처리 되었다. 생각해본 방법으로는 segmentation으로 검출된 대상에서 전처리 또는 후처리를 할 때 이미지의 윤곽만 먼저 가져오고, 가장자리는 차라리 블러처리를 해버리면 포토샵에서 누끼를 따오듯이 검출하려는 대상만 정확히 blur 처리가 되지 않고 뚜렷해지지 않을까? <br> <br>

3. 옷을 배경으로 인식:
    - 왜 옷이 blur처리 됐는지 도저히 모르겠다..... 이 부분은 2번과 반대로, segmentation을 해올 때, 가장자리 이외의 모든 object, 검출대상들은 blur 처리를 하지 않도록 대상으로 정확히 인식하게 한다.<br> <br>
```


# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```

- 이 프로젝트와 직접적으로 관련성있는 글들은 찾기 어렵네요. semantic segmentation 살펴본 글들을 첨부합니다.
- [Three Ways to Improve Semantic Segmentation with Self-Supervised Depth Estimation](https://openaccess.thecvf.com/content/CVPR2021/papers/Hoyer_Three_Ways_To_Improve_Semantic_Segmentation_With_Self-Supervised_Depth_Estimation_CVPR_2021_paper.pdf)
- [The Beginner’s Guide to Semantic Segmentation](https://www.v7labs.com/blog/semantic-segmentation-guide)
- [Push-the-Boundary: Boundary-aware Feature Propagation for Semantic Segmentation of 3D Point Clouds](https://arxiv.org/abs/2212.12402v1)
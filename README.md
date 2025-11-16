# 고객 활동 예측 프로젝트 (Customer-activity-prediction)

본 프로젝트는 고객이 남긴 한국어 신청 문장을 기반으로  
**“홍보 활동 안함 / 보통 / 적극적”**의 세 단계로 고객 활동 수준을 자동 분류하는 모델을 구현한 것입니다.  
KoNLPy(Okt), FastText Korean Embedding, PyTorch RNN 구조를 이용하여  
한국어 환경에 최적화된 텍스트 분류 시스템을 구축하였습니다.

---

## 프로젝트 개요

고객의 자유 입력 문장(`Apply`)만으로  
서비스 홍보 활동 가능성을 예측하는 것이 본 프로젝트의 핵심 목표입니다.

원본 `Score` 데이터를 그대로 사용하는 것이 아니라,  
이를 **비즈니스 활동 관점에서 3단계 레벨로 재해석**하여 문제를 재정의하였습니다.

---

## 데이터 구성 및 라벨링 전략

사용 데이터: `customer_trial_requests.csv`

라벨 변환 기준은 아래와 같습니다:

| Score | Label | 설명 |
|-------|--------|--------|
| 1 ~ 2 | 0 | 홍보 활동 저조 |
| 3 | 1 | 보통 수준 |
| 4 ~ 5 | 2 | 적극적 활동 |

라벨링을 통해 고객의 행동 성향을 직관적으로 분류할 수 있도록 설계하였습니다.

---

## NLP 전처리 과정

한국어 특성을 고려하여 다음과 같은 전처리 파이프라인을 구성하였습니다:

1. 결측치 및 중복 문장 제거  
2. KoNLPy **Okt 형태소 분석기** 적용  
3. 명사·동사·형용사·부사 중심으로 토큰 필터링  
4. **FastText Korean(300d)** 임베딩 기반 벡터화  
5. 모든 문장을 `MAX_SEQ_LEN` 기준으로 패딩 및 정규화  
6. RNN 입력 처리를 위해 시퀀스 길이 정보(`lengths`)를 별도로 유지  

이 과정을 통해 텍스트가 RNN 모델에서 안정적으로 처리될 수 있도록 정제하였습니다.

---

## 모델 구조 (SimpleRNNClassifier)

PyTorch 기반의 **단일 레이어 RNN 분류 모델**입니다.

- 입력: `(batch, seq_len, 300)`
- 구성 요소:
  - `nn.RNN(hidden_dim, batch_first=True)`
  - 시퀀스 마지막 hidden state 추출
  - `Dropout(0.5)`
  - 최종 Linear 레이어를 통한 3클래스 분류
- 특징: 구조가 간결하며, 해석 가능성과 유지보수성이 높습니다.

---

## 학습 전략

모델 학습은 다음과 같은 방식으로 진행되었습니다:

- Train / Test 분리: stratified split  
- Train / Validation 분리: stratified split  
- Loss function: `CrossEntropyLoss`  
- Optimizer: `Adam`  
- Gradient Clipping 적용  
- `ReduceLROnPlateau` 기반 Learning Rate 스케줄링  
- Validation loss 기준 Early Stopping 적용  

또한 가장 성능이 우수한 모델 상태를 자동 저장하도록 구성하였습니다.

---

## 모델 평가

테스트 데이터셋을 활용하여 다음의 지표로 모델 성능을 평가하였습니다:

- Accuracy  
- Confusion Matrix  
- Classification Report  
  - Precision, Recall, F1-score  

클래스 정의는 아래와 같습니다:

- `0`: 활동 저조  
- `1`: 보통  
- `2`: 적극적 활동  

---

## 예측 데모 (Usage)

모델 학습 완료 후 CLI 형태로 문장을 입력하면 예측 결과를 확인할 수 있습니다.

```bash
분석할 문장을 입력하세요 (종료하려면 'exit' 입력): 서비스 홍보는 제가 열심히 해보겠습니다.
입력 문장 예측 결과: 적극적 활동 (2)

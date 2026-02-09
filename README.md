# Discriminative-Finetuning-Reproduction
Based on paper "Cross-Domain Image Captioning with Discriminative Finetuning"

이 레포지토리는 논문 **[Cross-Domain Image Captioning with Discriminative Finetuning (CVPR 2023)]** 을 재현한 구현체입니다.

**DiscriTune**은 사전 학습된 이미지 캡셔닝 모델(ClipCap)을 **강화학습(Reinforcement Learning)** 을 통해 미세 조정(Finetuning)하는 방법론입니다. 고정된(frozen) 이미지 검색 모델(Retriever, CLIP)이 생성된 캡션을 보고 원본 이미지를 얼마나 잘 찾아내는지(Discriminative)를 보상(Reward)으로 사용하여, 사람의 정답 캡션 없이도 구별력이 뛰어난 캡션을 생성합니다.

## 주요 특징

* **Discriminative Objective:** 캡셔닝 모델이 단순히 사람의 문장을 모방하는 것이 아니라, 이미지를 다른 이미지와 구별(Discriminate)할 수 있도록 학습합니다.


* **Reinforcement Learning:** 미분 불가능한 텍스트 생성 과정을 REINFORCE 알고리즘을 통해 최적화합니다.


* **Cross-Domain Generalization:** COCO와 같은 특정 도메인에서 학습한 후, 새로운 도메인(Conceptual Captions 등)에서도 강건한 성능을 보입니다.


* **Model Architecture:** ClipCap (CLIP Visual Encoder + Mapping Network + GPT-2) 기반으로 구현되었습니다.

## 프로젝트 구조

* **config.py**: 실험 모드(E1~E4), 하이퍼파라미터, 파일 경로 설정.
* **dataset.py**: COCO 및 Conceptual Captions 데이터셋 로딩 클래스.
* **evaluate.py**: Beam Search 캡션 생성, Retrieval P@1 측정, CIDEr 점수 계산 로직.
* **main.py**: 프로그램의 진입점(Entry point)으로 학습 및 평가 루프 실행.
* **models.py**: ClipCap(Policy), CLIP(Retriever) 모델 정의 및 DiscriTuneTrainer 클래스.
* **retrieval.py**: 이미지 임베딩 캐싱 및 Retrieval 관련 유틸리티 함수.
* **runs/**: 학습 결과 및 로그가 저장되는 디렉토리.

## 설치 및 환경 설정

### 요구 사항

* Python 3.8+
* PyTorch (CUDA 권장)
* Transformers

### 설치

필요한 라이브러리를 설치합니다.

```bash
pip install torch torchvision
pip install transformers ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install pycocoevalcap

```

## 데이터 및 가중치 준비

`config.py` 파일에서 데이터셋 및 가중치 경로를 사용자의 환경에 맞게 수정해야 합니다.

### 1. 데이터셋

* **MS COCO:** Karpathy split JSON 파일과 이미지 데이터가 필요합니다.
* **Conceptual Captions (CC):** 이미지 폴더(`train`, `val`)와 메타데이터 파일(`meta.jsonl`)이 필요합니다.

```python
# config.py 수정 예시
self.COCO_KARPATHY_JSON = "/path/to/dataset_coco.json"
self.COCO_IMAGES_ROOT = "/path/to/coco_images"
self.CC_TRAIN_DIR = "/path/to/cc/train"
self.CC_VAL_DIR = "/path/to/cc/val"

```

### 2. 사전 학습 가중치

이 프로젝트는 ClipCap의 사전 학습된 가중치(Mapping Network, GPT-2)를 로드하여 시작합니다. 해당 가중치 파일(`.pt`)을 준비하고 경로를 지정해 주세요.

```python
# config.py
self.WEIGHTS_PATH = "./weights/coco_weights.pt"

```

## 사용 방법

`main.py`를 실행하여 학습 및 평가를 진행할 수 있습니다. 실행 시 대화형 프롬프트를 통해 실험 모드를 선택합니다.

```bash
python main.py

```

### 실행 모드

`config.py`에 정의된 실험 모드는 다음과 같습니다:

| 모드 | 설명 |
| --- | --- |
| **E1** | **(Pre) COCO Test:** Finetuning 전, COCO 데이터셋에서 성능 평가 (In-domain) |
| **E2** | **(Pre) CC Val:** Finetuning 전, Conceptual Captions에서 성능 평가 (Cross-domain) |
| **TRAIN** | **DiscriTune:** Discriminative Reward를 사용하여 모델 학습 (체크포인트 저장) |
| **E3** | **(Post) COCO Test:** Finetuning 후, COCO 데이터셋에서 성능 평가 |
| **E4** | **(Post) CC Val:** Finetuning 후, Conceptual Captions에서 성능 평가 |
| **ALL** | 위 과정을 순차적으로 모두 수행 (E1 -> E2 -> TRAIN -> E3 -> E4) |

### 학습 예시

```text
E1, E2, E3, E4, TRAIN, ALL 중 입력하세요: TRAIN
[checkpoint dataset] coco, concap 중 하나를 입력하세요: coco
DiscriTune 이어서 학습할까요? (y/n): n

```

## 평가 지표

`evaluate.py`를 통해 다음 지표들을 측정합니다:

* **Retrieval P@1 (P@1@100):** 생성된 캡션을 Query로 사용하여 100개의 이미지(Target 1 + Distractors 99) 중 Target을 정확히 찾아낸 비율입니다. DiscriTune의 핵심 성능 지표입니다.


* **CIDEr:** Reference 캡션과의 유사도를 측정합니다.



## 설정

`config.py`에서 주요 하이퍼파라미터를 조정할 수 있습니다.

* `EPOCHS`: 학습 에폭 (기본값: COCO 20, CC 2).
* `BATCH_SIZE`: 배치 크기 (기본값: 100).
* `LEARNING_RATE`: 학습률 (기본값: 1e-7).
* `REWARD_TYPE`: 보상 함수 종류 (`log_prob` 사용 권장).
* `MAX_TOKEN_LEN`: 생성 캡션 최대 길이 (기본값: 40).

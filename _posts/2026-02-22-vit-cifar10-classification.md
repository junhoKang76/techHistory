---
title: "ViT로 CIFAR-10 이미지 분류 학습하기"
description: "Vision Transformer가 이미지를 패치 단위로 처리하는 방식과 CIFAR-10 학습 과정에서 확인한 증강, 스케줄러, 하이퍼파라미터 튜닝 결과를 정리했습니다."
date: 2026-02-22
category: AI Engineering
permalink: /2026-02-vit-cifar10-classification/
image: /assets/images/vit-cifar10-classification/vit-architecture-overview.png
---

Transformer는 자연어 처리에서 출발했지만, 이미지를 작은 패치의 시퀀스로 바꾸면 이미지 분류에도 적용할 수 있습니다. 이 아이디어를 대표적으로 보여준 모델이 Vision Transformer, 즉 ViT입니다.

이 글에서는 CIFAR-10 이미지 분류를 ViT 구조로 학습하면서 확인한 흐름을 요약합니다. 실제 실습 코드는 아래 노트북에서 확인할 수 있습니다.

[TF01-Cifar10-Classification.ipynb](https://github.com/mkmkchoi/ai-model-study/blob/main/ipynb/TF01-Cifar10-Classification.ipynb)

## ViT가 이미지를 입력으로 받는 방식

ViT의 핵심은 이미지를 문장처럼 다루는 것입니다. 문장을 토큰으로 나누듯이, 이미지를 일정한 크기의 패치로 나누고 각 패치를 하나의 입력 토큰처럼 Transformer Encoder에 넣습니다.

<figure>
  <img src="{{ '/assets/images/vit-cifar10-classification/vit-architecture-overview.png' | relative_url }}" alt="Vision Transformer 구조 개요">
  <figcaption>출처: Dosovitskiy et al., <a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16x16 Words</a>.</figcaption>
</figure>

CIFAR-10 이미지는 32x32 크기입니다. 예를 들어 `patch_size`를 16으로 잡으면 이미지는 4개의 패치로 나뉘고, 8로 잡으면 16개의 패치가 만들어집니다. 패치 크기는 입력 토큰 수와 학습 난이도를 함께 바꾸는 중요한 하이퍼파라미터입니다.

각 패치는 RGB 채널 값을 펼친 뒤 Linear Layer를 거쳐 `emb_size` 차원의 벡터로 변환됩니다. 여기에 위치 정보를 더해 Transformer가 패치의 순서를 구분할 수 있게 합니다.

<figure>
  <img src="{{ '/assets/images/vit-cifar10-classification/position-encoding-sample.png' | relative_url }}" alt="위치 인코딩 값 예시">
  <figcaption>실습 코드의 `get_position_encoding()` 함수로 생성한 위치 인코딩 예시.</figcaption>
</figure>

Transformer Encoder는 Multi-Head Attention과 MLP 블록을 반복해서 쌓은 구조입니다. 실습 코드에서는 `depth`, `num_heads`, `emb_size`, `expansion` 같은 값들이 모델 용량과 학습 속도를 결정합니다. 분류 결과는 별도로 추가한 class token 위치의 출력값을 Classification Head에 통과시켜 얻습니다.

## 사전 학습 없이 CIFAR-10으로 학습

ViT 논문은 대규모 이미지 데이터로 사전 학습한 뒤 여러 이미지 분류 데이터셋에 fine-tuning하는 방식으로 높은 성능을 보였습니다.

<figure>
  <img src="{{ '/assets/images/vit-cifar10-classification/vit-paper-table2.png' | relative_url }}" alt="ViT 논문 Table 2 일부">
  <figcaption>출처: Dosovitskiy et al., <a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16x16 Words</a>, Table 2.</figcaption>
</figure>

이번 실습은 사전 학습 모델을 가져오지 않고 CIFAR-10 데이터만으로 학습하는 조건으로 진행했습니다. 먼저 `patch_size`별로 좋은 조합을 비교했을 때는 4x4 패치 설정이 가장 높은 정확도를 보였습니다.

<figure>
  <img src="{{ '/assets/images/vit-cifar10-classification/initial-patch-size-results.png' | relative_url }}" alt="patch_size별 초기 학습 결과 표">
  <figcaption>초기 비교에서는 `patch_size=4x4`, `depth=8`, `num_heads=6`, `emb_size=48` 조합이 78.47%로 가장 높았습니다.</figcaption>
</figure>

## 데이터 증강과 AutoAugment

Transformer 계열 모델은 데이터 규모와 다양성의 영향을 크게 받습니다. 그래서 `torchvision.transforms.AutoAugmentPolicy.CIFAR10`을 사용해 학습 중 이미지 변형이 계속 달라지도록 구성했습니다.

```python
transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)
```

AutoAugment는 데이터에서 증강 정책을 탐색하는 방식으로 제안된 기법이며, PyTorch는 CIFAR-10용 정책을 torchvision에서 바로 사용할 수 있게 제공합니다.

<figure>
  <img src="{{ '/assets/images/vit-cifar10-classification/autoaugment-result.png' | relative_url }}" alt="AutoAugment 적용 후 결과 표">
  <figcaption>AutoAugment 적용 후 같은 4x4 패치 조건에서 정확도가 80.21%까지 올라갔습니다.</figcaption>
</figure>

## 튜닝에서 효과가 있었던 값들

추가 튜닝에서는 공개 구현 사례를 참고했습니다. 특히 label smoothing, cosine learning rate decay, weight decay, 모델 차원 수가 성능에 영향을 주는 값으로 보였습니다.

<figure>
  <img src="{{ '/assets/images/vit-cifar10-classification/lr-cosine-decay-slide.png' | relative_url }}" alt="Cosine learning rate decay 설명 슬라이드">
  <figcaption>출처: Stanford CS231n 2024 lecture slide, <a href="https://cs231n.stanford.edu/slides/2024/lecture_8.pdf">Lecture 8 PDF</a>.</figcaption>
</figure>

label smoothing은 정답 클래스에만 확률 1을 주지 않고, 나머지 클래스에도 작은 값을 나누어 주는 방식입니다. 모델이 정답에 과도하게 확신하지 않도록 만들어 일반화에 도움을 줄 수 있습니다.

```text
적용 전: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
적용 후: [0.0111, 0.0111, 0.9, 0.0111, 0.0111, 0.0111, 0.0111, 0.0111, 0.0111, 0.0111]
```

실험에서는 다음 방향이 상대적으로 좋았습니다.

- `learning_rate`: `0.0001 -> 0.00001` 스케줄이 가장 좋았습니다.
- `weight_decay`: `0.00005`가 가장 좋았습니다.
- `emb_size`: 384 이상에서 성능이 좋아졌지만, 학습 시간을 고려해 384를 선택했습니다.
- `num_heads`: 큰 차이는 없었고 4가 근소하게 좋았습니다.
- `depth`: 7이 가장 좋았습니다.
- `batch_size`: 정확도 차이는 작았고, 속도를 고려해 128을 사용했습니다.
- optimizer: Adam과 AdamW의 차이는 크지 않았고 Adam이 근소하게 좋았습니다.
- dropout: 이번 조건에서는 사용하지 않는 편이 더 좋았습니다.

<details>
  <summary>튜닝 상세 표 보기</summary>

  <figure>
    <img src="{{ '/assets/images/vit-cifar10-classification/tuning-learning-rate.png' | relative_url }}" alt="learning_rate 튜닝 결과 표">
    <figcaption>learning rate 스케줄 비교.</figcaption>
  </figure>

  <figure>
    <img src="{{ '/assets/images/vit-cifar10-classification/tuning-weight-decay.png' | relative_url }}" alt="weight_decay 튜닝 결과 표">
    <figcaption>weight decay 비교.</figcaption>
  </figure>

  <figure>
    <img src="{{ '/assets/images/vit-cifar10-classification/tuning-emb-size.png' | relative_url }}" alt="emb_size 튜닝 결과 표">
    <figcaption>embedding size 비교.</figcaption>
  </figure>

  <figure>
    <img src="{{ '/assets/images/vit-cifar10-classification/tuning-num-heads.png' | relative_url }}" alt="num_heads 튜닝 결과 표">
    <figcaption>attention head 수 비교.</figcaption>
  </figure>

  <figure>
    <img src="{{ '/assets/images/vit-cifar10-classification/tuning-depth.png' | relative_url }}" alt="depth 튜닝 결과 표">
    <figcaption>encoder depth 비교.</figcaption>
  </figure>
</details>

튜닝을 누적했을 때 정확도는 87.56%까지 올라갔습니다. 이후 epoch을 늘리고 learning rate가 더 천천히 줄어들도록 조정했을 때 600 epoch에서 90.30%를 기록했습니다.

<figure>
  <img src="{{ '/assets/images/vit-cifar10-classification/tuning-epoch-final.png' | relative_url }}" alt="epoch 증가 후 최종 결과 표">
  <figcaption>600 epoch 학습에서 90.30% 정확도를 기록했습니다.</figcaption>
</figure>

## 정리

이번 실습에서 가장 크게 확인한 점은 ViT를 작은 이미지 데이터셋에 바로 적용할 때도 구조와 학습 전략을 함께 조정해야 한다는 것입니다. 패치 크기, 모델 차원, attention head 수, encoder depth 같은 구조적 하이퍼파라미터뿐 아니라 AutoAugment, label smoothing, learning rate decay, epoch 수가 함께 성능을 좌우했습니다.

특히 사전 학습 없이 CIFAR-10만으로 학습하는 조건에서는 초기 설정만으로는 성능이 빠르게 올라가지 않았고, 데이터 증강과 학습 스케줄을 보강한 뒤에야 90% 이상의 정확도를 확인할 수 있었습니다.

## 참고 자료

- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), arXiv:2010.11929.
- PyTorch, [AutoAugmentPolicy documentation](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.AutoAugmentPolicy.html).
- Cubuk et al., [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501), arXiv:1805.09501.
- omihub777, [ViT-CIFAR](https://github.com/omihub777/ViT-CIFAR).
- Stanford CS231n, [2024 lecture slide PDF](https://cs231n.stanford.edu/slides/2024/lecture_8.pdf).

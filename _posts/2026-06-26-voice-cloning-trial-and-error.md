---
layout: post
title: "TTS 모델에 Voice Cloning 붙이기: 삽질의 기록"
date: 2026-06-26
category: AI Engineering
description: "자체 학습한 멀티스피커 TTS 모델에 Voice Cloning을 추가하면서 겪은 시행착오를 기록했습니다."
read_time: 12분
---

자체 학습한 멀티스피커 TTS 모델에 Voice Cloning 기능을 추가하면서 겪은 시행착오를 기록했습니다. 이론대로 되지 않는 것들이 생각보다 많았고, 그 과정에서 배운 것들이 더 많았습니다.

---

## 목표: 레퍼런스 오디오만으로 목소리 복제

기존에 학습된 멀티스피커 TTS 모델은 화자 ID(정수)를 입력받아 내부 임베딩 테이블에서 화자 벡터 `g`를 조회합니다. 학습 데이터에 없는 새로운 목소리를 쓰려면 재학습이 필요하죠. Voice Cloning은 이 `g`를 짧은 레퍼런스 오디오에서 실시간으로 만들어내는 기능입니다.

<div class="diagram-wrap">
<svg width="100%" viewBox="0 0 680 200" role="img">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#667085" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <rect x="20" y="76" width="130" height="48" rx="8" fill="#edf7f5" stroke="#99d6cf" stroke-width="0.5"/>
  <text x="85" y="96" text-anchor="middle" font-size="12" font-weight="600" fill="#115e59" font-family="-apple-system,sans-serif">레퍼런스 오디오</text>
  <text x="85" y="112" text-anchor="middle" font-size="11" fill="#0f766e" font-family="-apple-system,sans-serif">3~10초</text>
  <line x1="150" y1="100" x2="178" y2="100" stroke="#667085" stroke-width="1" marker-end="url(#arr)"/>
  <rect x="180" y="64" width="170" height="72" rx="8" fill="#f0fdf4" stroke="#86efac" stroke-width="0.5"/>
  <text x="265" y="88" text-anchor="middle" font-size="12" font-weight="600" fill="#166534" font-family="-apple-system,sans-serif">Speaker Encoder</text>
  <text x="265" y="106" text-anchor="middle" font-size="11" fill="#15803d" font-family="-apple-system,sans-serif">ECAPA-TDNN</text>
  <text x="265" y="122" text-anchor="middle" font-size="11" fill="#15803d" font-family="-apple-system,sans-serif">+ Projection Layer</text>
  <line x1="350" y1="100" x2="378" y2="100" stroke="#667085" stroke-width="1" marker-end="url(#arr)"/>
  <rect x="380" y="80" width="60" height="40" rx="8" fill="#fefce8" stroke="#fde047" stroke-width="0.5"/>
  <text x="410" y="97" text-anchor="middle" font-size="12" font-weight="600" fill="#854d0e" font-family="-apple-system,sans-serif">g</text>
  <text x="410" y="112" text-anchor="middle" font-size="10" fill="#a16207" font-family="-apple-system,sans-serif">256d</text>
  <line x1="440" y1="100" x2="468" y2="100" stroke="#667085" stroke-width="1" marker-end="url(#arr)"/>
  <rect x="470" y="64" width="130" height="72" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="0.5"/>
  <text x="535" y="88" text-anchor="middle" font-size="12" font-weight="600" fill="#1e40af" font-family="-apple-system,sans-serif">TTS Generator</text>
  <text x="535" y="106" text-anchor="middle" font-size="11" fill="#1d4ed8" font-family="-apple-system,sans-serif">텍스트 입력</text>
  <text x="535" y="122" text-anchor="middle" font-size="11" fill="#1d4ed8" font-family="-apple-system,sans-serif">→ 파형 출력</text>
  <line x1="535" y1="160" x2="535" y2="138" stroke="#667085" stroke-width="1" marker-end="url(#arr)"/>
  <text x="535" y="175" text-anchor="middle" font-size="11" fill="#667085" font-family="-apple-system,sans-serif">텍스트</text>
</svg>
<p class="diagram-caption">그림 1. Voice Cloning 전체 구조 — 레퍼런스 오디오에서 화자 벡터 g를 추출해 TTS Generator의 조건으로 주입</p>
</div>

---

## 삽질의 연속

### 삽질 01 — Projection Layer의 hidden 차원이 입력보다 작았다

Speaker Encoder(256d)와 Mel 평균(80d)을 합친 336d 입력을 `336 → 256 → 256`으로 설계했는데, 첫 레이어에서 바로 병목이 생겼습니다. 두 이질적인 분포의 정보를 융합하려면 중간 공간이 충분히 넓어야 합니다.

```python
# Before: 병목 발생
336 → 256 → 256

# After: 충분한 융합 공간 확보
336 → 512 → 512 → 256
```

> 💡 **교훈** Projection이 단순 차원 맞춤이 아닌 두 분포의 융합 역할을 하므로, hidden은 입력보다 넓어야 합니다.

---

### 삽질 02 — TTS 모델이 외부 g를 받는 인자가 없었다

Projection이 만든 `g`를 Generator에 넘겼더니 loss가 전혀 줄지 않았습니다. 원인을 추적해보니 실제 코드의 `forward()`가 화자 ID(sid)로만 `g`를 생성하는 구조였고, 외부 주입 인자 자체가 없었습니다.

```python
# 문제: 외부 g 주입 불가
def forward(self, x, x_lengths, y, y_lengths, sid=None):
    g = self.emb_g(sid).unsqueeze(-1)   # sid로만 생성

# 해결: g 인자 추가 (2줄 수정)
def forward(self, x, x_lengths, y, y_lengths, sid=None, g=None):
    if g is None:
        g = self.emb_g(sid).unsqueeze(-1)   # 하위 호환 유지
```

> 💡 **교훈** 오픈소스 모델을 활용할 때는 반환값과 인자 구조를 반드시 직접 확인해야 합니다.

---

### 삽질 03 — 반환값 구조 불일치로 엉뚱한 텐서로 loss 계산

사용한 구현체는 KL loss가 *duration KL + audio KL* 두 단계로 분리된 독특한 반환값 구조를 갖고 있었습니다. 일반적인 구조로 언패킹하다 보니 에러 없이 학습이 진행되면서도 loss가 전혀 움직이지 않는 상황이 발생했습니다.

```python
# 반환값 구조를 먼저 출력해서 확인
for i, v in enumerate(outputs):
    if isinstance(v, tuple):
        print(f"[{i}] tuple, len={len(v)}")
    else:
        print(f"[{i}] shape={v.shape}")
```

> 💡 **교훈** 에러 없이 loss만 안 줄 때 — 반환값 구조 불일치가 가장 흔한 원인입니다.

---

### 삽질 04 — Generator와 Discriminator를 모두 freeze했더니 학습 자체가 불가

Projection만 학습시키려고 Generator/Discriminator 모두 freeze했습니다. 그랬더니 loss가 전혀 움직이지 않았습니다.

<div class="diagram-wrap">
<svg width="100%" viewBox="0 0 680 140" role="img">
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#667085" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <rect x="20" y="46" width="120" height="48" rx="8" fill="#f0fdf4" stroke="#86efac" stroke-width="1"/>
  <text x="80" y="66" text-anchor="middle" font-size="12" font-weight="600" fill="#166534" font-family="-apple-system,sans-serif">Projection</text>
  <text x="80" y="82" text-anchor="middle" font-size="11" fill="#15803d" font-family="-apple-system,sans-serif">학습 중</text>
  <line x1="140" y1="70" x2="168" y2="70" stroke="#667085" stroke-width="1" marker-end="url(#arr2)"/>
  <rect x="170" y="46" width="140" height="48" rx="8" fill="#f9fafb" stroke="#d1d5db" stroke-width="1" stroke-dasharray="4 2"/>
  <text x="240" y="66" text-anchor="middle" font-size="12" fill="#9ca3af" font-family="-apple-system,sans-serif">Generator</text>
  <text x="240" y="82" text-anchor="middle" font-size="11" fill="#9ca3af" font-family="-apple-system,sans-serif">🔒 freeze</text>
  <line x1="310" y1="70" x2="338" y2="70" stroke="#667085" stroke-width="1" marker-end="url(#arr2)"/>
  <rect x="340" y="46" width="150" height="48" rx="8" fill="#f9fafb" stroke="#d1d5db" stroke-width="1" stroke-dasharray="4 2"/>
  <text x="415" y="66" text-anchor="middle" font-size="12" fill="#9ca3af" font-family="-apple-system,sans-serif">Discriminator</text>
  <text x="415" y="82" text-anchor="middle" font-size="11" fill="#9ca3af" font-family="-apple-system,sans-serif">🔒 freeze</text>
  <line x1="490" y1="70" x2="518" y2="70" stroke="#ef4444" stroke-width="1" marker-end="url(#arr2)"/>
  <rect x="520" y="46" width="140" height="48" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="0.5"/>
  <text x="590" y="66" text-anchor="middle" font-size="12" fill="#991b1b" font-family="-apple-system,sans-serif">학습 신호</text>
  <text x="590" y="82" text-anchor="middle" font-size="11" fill="#dc2626" font-family="-apple-system,sans-serif">❌ 전달 안 됨</text>
</svg>
<p class="diagram-caption">그림 2. Generator/Discriminator 모두 freeze 시 Projection에 유효한 학습 신호가 전달되지 않음</p>
</div>

Discriminator가 freeze되면 Projection이 만든 새로운 `g` 분포를 고정된 기준으로만 판단하므로 유효한 gradient가 흐르지 않습니다.

> 💡 **해결** Generator만 freeze하고 Discriminator는 같이 학습합니다. Discriminator가 새로운 g 분포에 함께 적응해야 유효한 학습 신호가 만들어집니다.

---

### 삽질 05 — Phase 전환 직후 출력이 완전히 붕괴됐다

1단계 학습(임베딩 모방)에서 어느 정도 품질이 나왔는데, 2단계(GAN 학습)로 전환하자마자 전혀 알아들을 수 없는 소리가 나왔습니다.

| 원인 | 설명 |
|---|---|
| D가 너무 빠르게 강해짐 | 새로운 g 기반 출력을 즉시 "가짜"로 판단 → gradient 폭발 |
| lr 불균형 | 새로 학습 시작하는 D가 이미 수렴한 Projection보다 빠르게 강해짐 |
| gradient 방향 충돌 | 1단계 loss 방향과 GAN loss 방향이 초반에 충돌 |

**해결: Warmup 구간 도입**

전환 직후 ~1,000 steps는 Mel loss만 사용하고 Discriminator 학습을 지연합니다. lr도 이전 단계의 1/10로 낮춥니다.

---

### 삽질 06 — Cross-lingual: 영어 화자로 한국어를 발음하면 억양이 섞인다

영어 화자의 레퍼런스로 한국어를 합성하면 영어식 억양이 섞입니다. Speaker Embedding에는 음색(원하는 것)과 억양/리듬(원하지 않는 것)이 함께 담겨 있기 때문입니다.

<div class="diagram-wrap">
<svg width="100%" viewBox="0 0 680 170" role="img">
  <defs>
    <marker id="arr3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#667085" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <rect x="20" y="62" width="130" height="46" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="0.5"/>
  <text x="85" y="82" text-anchor="middle" font-size="12" font-weight="600" fill="#1e40af" font-family="-apple-system,sans-serif">Speaker</text>
  <text x="85" y="98" text-anchor="middle" font-size="11" fill="#1d4ed8" font-family="-apple-system,sans-serif">Embedding</text>
  <line x1="150" y1="85" x2="178" y2="85" stroke="#667085" stroke-width="1" marker-end="url(#arr3)"/>
  <rect x="180" y="62" width="120" height="46" rx="8" fill="#f0fdf4" stroke="#86efac" stroke-width="0.5"/>
  <text x="240" y="82" text-anchor="middle" font-size="12" font-weight="600" fill="#166534" font-family="-apple-system,sans-serif">Projection</text>
  <text x="240" y="98" text-anchor="middle" font-size="11" fill="#15803d" font-family="-apple-system,sans-serif">timbre emb.</text>
  <line x1="300" y1="75" x2="340" y2="50" stroke="#667085" stroke-width="1" marker-end="url(#arr3)"/>
  <rect x="342" y="30" width="120" height="40" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="0.5"/>
  <text x="402" y="48" text-anchor="middle" font-size="12" font-weight="600" fill="#1e40af" font-family="-apple-system,sans-serif">TTS 합성</text>
  <text x="402" y="62" text-anchor="middle" font-size="11" fill="#1d4ed8" font-family="-apple-system,sans-serif">음색 조건</text>
  <line x1="300" y1="95" x2="340" y2="118" stroke="#ef4444" stroke-width="1" stroke-dasharray="4 2" marker-end="url(#arr3)"/>
  <rect x="342" y="100" width="100" height="40" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="0.5"/>
  <text x="392" y="118" text-anchor="middle" font-size="11" font-weight="600" fill="#991b1b" font-family="-apple-system,sans-serif">GRL</text>
  <text x="392" y="132" text-anchor="middle" font-size="10" fill="#dc2626" font-family="-apple-system,sans-serif">gradient 역전</text>
  <line x1="442" y1="120" x2="470" y2="120" stroke="#ef4444" stroke-width="1" stroke-dasharray="4 2" marker-end="url(#arr3)"/>
  <rect x="472" y="100" width="120" height="40" rx="8" fill="#fefce8" stroke="#fde047" stroke-width="0.5"/>
  <text x="532" y="118" text-anchor="middle" font-size="11" font-weight="600" fill="#854d0e" font-family="-apple-system,sans-serif">언어 분류기</text>
  <text x="532" y="132" text-anchor="middle" font-size="10" fill="#a16207" font-family="-apple-system,sans-serif">lang_acc → 0.33</text>
</svg>
<p class="diagram-caption">그림 3. Gradient Reversal Layer — 언어 분류기가 언어를 맞추려 할수록 Projection에는 언어 정보를 담지 말라는 신호가 역전되어 전달</p>
</div>

`lang_acc`가 랜덤 수준(1/언어 수)에 수렴하면 언어 정보가 성공적으로 제거된 것입니다.

---

## 최종적으로 안정된 학습 파이프라인

<div class="stage-flow">
  <div class="stage-item">
    <div class="stage-line"><div class="stage-dot">0</div><div class="stage-connector"></div></div>
    <div class="stage-content">
      <h4>새 화자 임베딩 테이블 학습</h4>
      <p>기존 TTS 모델의 화자 임베딩 테이블에 새 화자 행을 추가. 나머지 파라미터는 freeze한 채 새 화자 행만 gradient mask로 학습. Optimizer state도 함께 마이그레이션 필요.</p>
    </div>
  </div>
  <div class="stage-item">
    <div class="stage-line"><div class="stage-dot">1</div><div class="stage-connector"></div></div>
    <div class="stage-content">
      <h4>Projection 초기화 (임베딩 모방)</h4>
      <p>Stage 0에서 학습된 임베딩 벡터를 정답으로, Cosine + MSE loss로 Projection 초기화. cosine similarity > 0.95 수렴 확인.</p>
    </div>
  </div>
  <div class="stage-item">
    <div class="stage-line"><div class="stage-dot">2</div><div class="stage-connector"></div></div>
    <div class="stage-content">
      <h4>GAN 학습 (Warmup 포함)</h4>
      <p>Generator freeze, Discriminator 학습 가능. 전환 직후 ~1,000 steps Warmup (Mel loss만, lr 1/10). 이후 전체 GAN loss 적용.</p>
    </div>
  </div>
  <div class="stage-item">
    <div class="stage-line"><div class="stage-dot">3</div><div class="stage-connector"></div></div>
    <div class="stage-content">
      <h4>언어 중립성 강화 (선택)</h4>
      <p>Cross-lingual 품질이 부족할 때 Adversarial Projection 추가. lang_acc 모니터링으로 언어 정보 제거 여부 확인.</p>
    </div>
  </div>
  <div class="stage-item">
    <div class="stage-line"><div class="stage-dot">4</div></div>
    <div class="stage-content" style="padding-bottom:0">
      <h4>전체 파인튜닝 (선택)</h4>
      <p>Speaker Encoder 상위 레이어 + Generator 언프리즈. lr 1e-5 수준 유지.</p>
    </div>
  </div>
</div>

---

## 데이터 구성

새 화자당 2초짜리 발화 1,000개(약 33분)는 Stage 0 학습에 충분합니다.

| Stage | 데이터 | 목적 |
|---|---|---|
| 0, 1 | VCTK + LibriTTS → + AIHub 한국어 | Projection 학습 |
| 3 | VoxPopuli, CommonVoice (언어 레이블) | 언어 중립성 |

배치 내에 같은 화자의 발화가 최소 2개씩 포함되어야 일관성 loss 계산이 가능합니다. 전체 화자를 shuffle해서 섞으면 화자 간 임베딩 경계가 더 명확하게 형성됩니다.

---

## 배운 것들

**기술적으로** — Speaker Embedding에는 음색과 억양이 함께 담겨 있고, 분리가 생각보다 어렵습니다. GAN 학습에서 어떤 파라미터를 freeze하느냐가 학습 가능 여부 자체를 결정합니다. Loss가 줄지 않을 때 gradient 흐름을 먼저 확인하는 것이 가장 빠른 디버깅이었습니다.

**프로세스적으로** — 각 단계를 작게 나누고 실제 출력을 귀로 들으며 진행해야 합니다. 이론적으로 맞는 구조가 실제로는 동작하지 않는 경우가 많았고, 오픈소스 모델은 반환값 구조부터 직접 확인하는 습관이 중요했습니다.

아직 완성된 시스템이 아니고 계속 개선 중입니다. 비슷한 작업을 해보신 분들의 경험이 궁금합니다.

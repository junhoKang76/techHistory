---
title: "Brity Meeting × Voice Cloning TTS — 기술 탐색 리포트"
date: 2025-06-28
categories: [TTS, VoiceCloning, BrityMeeting, OnDevice]
tags: [VibeVoice, MOSS-TTS, SpeakerCloning, EdgeAI]
---

> Brity Meeting · 기술 트렌드 게시판

# Brity Meeting의 목소리를 지키는 기술 — Voice Cloning TTS 탐색기

TTS 기반 실시간 통역에서 발화자의 목소리를 유지하는 것이 왜 중요한지,  
그리고 On-device 구동을 위한 최신 경량 모델 두 가지를 기술적으로 분석합니다.

| | |
|---|---|
| 참조 모델 | **VibeVoice** (MS Research, arXiv 2508.19205, 2025.08) |
| 참조 모델 | **MOSS-TTS Nano** (OpenMOSS / Fudan Univ., 2025~) |
| 분류 | 기술 트렌드 |

---

## 목차

1. [왜 Voice Cloning인가](#1-왜-voice-cloning인가)
2. [Voice Cloning TTS 발전사](#2-voice-cloning-tts-발전사)
3. [대부분의 기법은 GPU 없이 돌아가지 않는다](#3-대부분의-기법은-gpu-없이-돌아가지-않는다)
4. [Brity Meeting의 On-device 요건](#4-brity-meeting의-on-device-요건)
5. [두 모델 개요](#5-두-모델-개요-vibevoice-vs-moss-tts-nano)
6. [VibeVoice 세부 구조](#6-vibevoice-세부-구조)
7. [MOSS-TTS Nano 세부 구조](#7-moss-tts-nano-세부-구조)
8. [설계 철학 비교](#8-설계-철학-비교)
9. [사용 시나리오별 추천](#9-사용-시나리오별-추천)
10. [결론 및 향후 계획](#10-결론-및-향후-계획)

---

## 1. 왜 Voice Cloning인가

Brity Meeting의 실시간 통역 기능은 발화자 A가 영어로 말하면 한국어 음성을 합성해 전달합니다.  
그런데 내용은 맞더라도 **완전히 다른 목소리의 통역이 흘러나오면 회의 몰입도(presence)가 순간 끊깁니다.**  
"이게 A가 한 말이 맞나?" 하는 어색함이 생기죠.

발화자의 음색·억양·템포를 최대한 유지한 채 목표 언어로 발화해 주는 **Voice Cloning TTS**가 필요한 이유입니다.

**Brity Meeting 실시간 통역 파이프라인**

```
발화자 (영어 발화)
  → ASR : 음성 → 텍스트
  → MT  : 영어 → 한국어
  → TTS + Voice Cloning : 발화자 음색 유지
  → 청취자 (한국어 수신)
```

> 📌 실시간 파이프라인 — 각 단계의 지연이 누적되므로 TTS 지연 최소화가 핵심

---

## 2. Voice Cloning TTS 발전사

Voice Cloning 기술은 지난 10년간 급속도로 발전했습니다.  
수백 시간 분량의 녹음이 필요하던 시대에서, 이제는 10초 샘플만으로 그 사람의 목소리를 재현할 수 있습니다.

**`~2013` — HMM / 통계적 TTS**  
화자별 수백 시간 녹음 데이터 필요. 새 화자 추가 = 처음부터 재학습.

**`2017` — Tacotron (Google)**  
딥러닝 기반 E2E TTS 등장. 텍스트 → 멜스펙트로그램 → 음성. 자연스러움 급상승.

**`2019–2021` — VITS / YourTTS / SV2TTS**  
Zero-shot 화자 클로닝 개념 도입. 소수의 샘플(수 분)만으로 새 화자 생성 가능.

**`2023` — VALL-E / CosyVoice2 / XTTS-v2**  
LLM 기반 음성 생성 등장. 3초 프롬프트로 고품질 클로닝. 다국어 지원 시작.

**`2025` — VibeVoice / MOSS-TTS (경량화 시도)**  
GPU 필수라는 한계를 돌파하려는 시도. On-device 구동을 목표로 한 모델 등장.

---

## 3. 대부분의 기법은 GPU 없이 돌아가지 않는다

최신 Voice Cloning TTS의 고품질화는 모델의 거대화로 이어졌습니다.

- **Speaker Encoder** `GPU 권장`  
  ECAPA-TDNN 등의 화자 인코더가 Voice Prompt를 고차원 임베딩으로 압축. 수백만 파라미터 행렬 연산 필요.

- **AR 언어 모델 (LLM)** `GPU 강력 권장`  
  Qwen, LLaMA 기반의 수십억 파라미터 LM이 음성 토큰을 자동회귀적으로 생성. 토큰당 순방향 패스 1회.

- **Diffusion / Flow Matching** `GPU 필수`  
  노이즈에서 음향 특성을 반복적으로 제거. 매 생성 스텝마다 20~1000회의 반복 연산 필요.

- **Audio Tokenizer (RVQ)** `GPU 권장`  
  EnCodec, DAC 등 고해상도 오디오를 이산 코드북 토큰으로 변환. 실시간 처리 시 연산량 상당.

> ⚠️ Diffusion 계열은 추론 시에도 수십~수백 step의 반복 연산이 필요해, GPU 없이는 실시간 동작 자체가 불가능합니다. 단 하나의 문장을 생성하는 데 CPU에서 수분이 걸릴 수 있습니다.

---

## 4. Brity Meeting의 On-device 요건

Brity Meeting은 기업 보안 환경에서 동작하는 솔루션입니다.  
음성 데이터를 외부 클라우드로 보내기 어려운 환경이 많기 때문에, **단말 내 처리(On-device)** 가 필수 조건입니다.

**동시에 충족해야 하는 4가지 조건**

| 조건 | 설명 |
|---|---|
| CPU 실시간 | GPU 없이 단말에서 동작 |
| 한국어 지원 | 다국어 필수 |
| 저지연 | 실시간 회의에 대응 |
| 코드 접근 가능 | 커스터마이징 가능한 오픈소스 |

위에서 설명한 대부분의 최신 Voice Cloning TTS는 이 조건을 충족하지 못합니다.  
그래서 최근 공개된 **경량화 버전 두 가지**에 주목하게 되었습니다.

---

## 5. 두 모델 개요: VibeVoice vs. MOSS-TTS Nano

| 항목 | VibeVoice | MOSS-TTS Nano |
|---|---|---|
| 개발사 | Microsoft Research | MOSI.AI / Fudan Univ. |
| 발표 | 2025.08 (arXiv 2508.19205) | 2025.06~ 지속 릴리즈 |
| 모델 크기 | ~3B (1.5B LLM 기준 전체) | 0.1B LM + 1.6B CAT |
| 지원 언어 | ❌ 영어·중국어만 | ✅ 10개+ (한국어 포함) |
| CPU 실시간 | ❌ 불가 (GPU 필수) | ✅ 가능 (설계 목표) |
| Voice Cloning | In-context (10초+ 프롬프트) | Zero-shot, 파인튜닝 선택 |
| 라이선스 | ❌ MIT (코드 비공개 전환) | ✅ 완전 오픈소스 |
| 학술 성과 | ICLR 2026 Oral 채택 | Tech Report 공개 |

---

## 6. VibeVoice 세부 구조

### 핵심 아이디어: 연속 공간에서 다음 토큰 예측

VibeVoice의 가장 큰 차별점은 **오디오를 이산(discrete) 코드북 토큰이 아닌 연속(continuous) 벡터로 모델링**한다는 점입니다.  
기존 VALL-E 계열이 "어떤 코드북 index인가"를 예측했다면, VibeVoice는 "다음 음향 벡터가 무엇인가"를 Diffusion으로 예측합니다.

**VibeVoice 전체 파이프라인**

```
[텍스트 입력]         [Voice Prompt: 10초+ 음성]
      │                        │
      ▼                        ▼
 ┌─────────────────────────────────────┐
 │         7.5 Hz Dual Tokenizer       │
 │  ┌─────────────────┐ ┌───────────┐  │
 │  │ Semantic Tok.   │ │Acoustic   │  │
 │  │ (ASR 프록시 학습) │ │Tok. (VAE) │  │
 │  │ "무슨 말인가"    │ │"누구 목소리"│ │
 │  └─────────────────┘ └───────────┘  │
 └──────────────┬──────────────────────┘
                │ 연속 벡터
                ▼
       ┌─────────────────┐
       │  Qwen2.5 LLM    │
       │  (1.5B / 7B)    │
       │  AR 생성         │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │  Diffusion Head  │
       │  노이즈 제거 반복  │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │  VAE Decoder    │
       │  파형 복원       │
       └────────┬────────┘
                │
                ▼
           [오디오 출력]
```

### 이중 토크나이저: Semantic + Acoustic

두 토크나이저는 역할이 완전히 다릅니다.

**Semantic Tokenizer** — 텍스트 입력 처리  
음성인식(ASR)을 프록시 태스크로 삼아 학습. 음성의 내용적 의미와 언어 정보를 포착합니다. *"무슨 말인가"* 를 담당.

**Acoustic Tokenizer** — Voice Prompt 처리  
VAE(Variational Autoencoder) 구조. 음색·억양·호흡 패턴 등 화자 정체성 정보를 연속 벡터로 압축합니다. *"누구의 목소리인가"* 를 담당.

> 💜 **핵심: 왜 7.5 Hz인가?**  
> 기존 EnCodec은 75 Hz × 8 코드북 → 30초 음성이 18,000 토큰 → LLM 처리 불가.  
> VibeVoice의 7.5 Hz는 80배 압축 → 30초 = 225 토큰.  
> 90분 음성도 ~40,500 토큰으로 LLM 컨텍스트에 들어갑니다.

| 수치 | 설명 |
|---|---|
| **18,000** | EnCodec 기준 30초 음성 토큰 수 |
| **225** | VibeVoice 기준 30초 음성 토큰 수 |
| **80×** | 압축률 (품질 유지) |

### Diffusion Head: 연속 공간에서 음향 생성

LLM이 다음 시점의 의미적 벡터를 예측하면, Diffusion Head가 그것을 실제 음향 특성 벡터로 변환합니다.

```
[순수 노이즈 z_T]
  → Denoise (LLM 조건 입력) step T→T-1
  → Denoise (LLM 조건 입력) step T-1→…
  → … 반복 …
  → [목표 음향 벡터 z_0]
```

> 💜 **왜 이산 토큰보다 유리한가?**  
> 이산 토큰은 코드북 내 가장 가까운 값으로 반올림 → 미묘한 억양·감정 손실.  
> 연속 벡터는 반올림 없이 표현 → 표현력이 훨씬 높습니다.  
> 이것이 VibeVoice의 음질이 다른 모델보다 자연스럽다고 평가받는 기술적 이유입니다.

**✅ VibeVoice 장점**

- 90분 연속 생성 — 현존 오픈소스 최장 컨텍스트
- Diffusion 연속 표현 → 최고 수준의 자연스러움
- ICLR 2026 Oral 채택, ElevenLabs·Gemini TTS 초과 품질 보고
- In-context 화자 클로닝 10초만으로 가능
- 자발적 노래·감정 표현 Emergent 능력 관찰

**❌ VibeVoice 한계**

- 영어·중국어만 지원 — 한국어 없음
- 코드 비공개 전환 (2025.9) — 재현 어려움
- 7B 모델 가중치 삭제 (딥페이크 우려)
- GPU 필수 — On-device 불가
- 배경음악 비제어 생성 불안정성

---

## 7. MOSS-TTS Nano 세부 구조

### 핵심 아이디어: 작게, 빠르게, 그래도 품질은

MOSS-TTS Nano는 설계 목표 자체가 VibeVoice와 정반대입니다.  
**"어떻게 하면 GPU 없이 CPU에서 실시간으로 돌릴 수 있을까"** 를 먼저 정하고,  
그 제약 안에서 최대한 좋은 품질을 내는 방향으로 설계되었습니다.

**MOSS-TTS Nano 전체 파이프라인**

```
[텍스트 입력]    [Voice Prompt: 5~30초]
      │                   │
      └──────────┬────────┘
                 ▼
       ┌─────────────────────┐
       │   CAT Tokenizer     │
       │   1.6B Causal       │
       │   Transformer       │
       │   CNN-free 구조      │
       │   RVQ 16 codebook   │
       │   → 12.5 Hz 이산 토큰 │
       └──────────┬──────────┘
                  │ 이산 토큰 (화자 정보 포함)
                  ▼
       ┌─────────────────────┐
       │   AR LM (0.1B)      │
       │   Qwen 기반 초경량    │
       │   토큰 → 토큰 자동회귀  │
       └──────────┬──────────┘
                  │ 예측된 오디오 토큰
                  ▼
       ┌─────────────────────┐
       │   CAT Decoder       │
       │   이산 토큰 → 파형    │
       │   48 kHz 스테레오    │
       └──────────┬──────────┘
                  ▼
             [오디오 출력]
```

### CAT: CNN을 쓰지 않는 이유

기존 오디오 토크나이저(EnCodec, DAC 등)는 CNN(합성곱 신경망)으로 오디오를 다운샘플링합니다.  
CNN은 오디오 처리에 자연스럽게 맞는 구조지만, 경량화와 통합 아키텍처 구축에 걸림돌이 됩니다.

CAT는 이를 **Causal Transformer만으로 대체**했습니다.

RVQ(잔차 벡터 양자화)를 16단계로 쌓아 이산 토큰을 생성합니다.

- **Codebook 1**: 전체 음질 · 언어 내용
- **Codebook 4**: 운율 · 억양 패턴
- **Codebook 8**: 음색 세부 질감
- **Codebook 16**: 미세 배음 구조

*속도 우선 상황에서는 상위 코드북(1~4)만 사용해 품질을 낮추고 추론 속도를 올리는 전략도 가능합니다.*

### 0.1B LM이 AR로 동작하는 방식

LM은 텍스트 토큰과 Voice Prompt 토큰을 컨텍스트로 보유한 채, 매 시점마다 다음 오디오 토큰을 하나씩 예측합니다.

```
[텍스트 토큰 T] + [프롬프트 P₁…Pₙ]
  → A₁ → A₂ → A₃ → … → [EOS]
         ↓
    CAT Decoder → 연속 파형
```

> 💚 **왜 AR 방식이 CPU에서 유리한가?**  
> 토큰 하나 예측 = 순방향 패스 1회.  
> VibeVoice처럼 매 스텝마다 Diffusion 반복 연산이 없습니다.  
> 첫 토큰이 생성되는 순간부터 오디오 재생을 시작(스트리밍)할 수 있어 체감 지연도 줄어듭니다.

**✅ MOSS-TTS Nano 장점**

- 0.1B LM — CPU 전용 실시간 구동 가능
- 한국어 포함 10개+ 언어 네이티브 지원
- 코드·가중치 완전 오픈소스 공개
- 48 kHz 스테레오 고음질 (v2 기준)
- SGLang 최적화 최대 16× 가속
- 파인튜닝으로 음질 향상 가능

**❌ MOSS-TTS Nano 한계**

- CAT Tokenizer 1.6B — 전체 스택 메모리 부담
- VibeVoice 대비 화자 유사도 낮음
- 실제 모바일 기기 지연 미검증
- 고품질 모델(4B)은 여전히 GPU 권장

---

## 8. 설계 철학 비교

| 설계 결정 | VibeVoice | MOSS-TTS Nano |
|---|---|---|
| 오디오 표현 | 연속 벡터 (Continuous) — 손실 없는 미세 표현 | 이산 토큰 (Discrete) — RVQ 16 코드북 |
| 생성 패러다임 | AR LLM + Diffusion Head — 매 스텝 노이즈 제거 반복 | 순수 AR (토큰→토큰) — 스텝당 순방향 패스 1회 |
| 화자 정보 처리 | Acoustic Tokenizer 분리 — 화자 벡터→LLM 조건 | CAT로 통합 토큰화 — 프롬프트 토큰 컨텍스트에 포함 |
| 토크나이저 주파수 | `7.5 Hz` 이중 구조 (Semantic + Acoustic) | `12.5 Hz` 단일 CAT (CNN-free Causal Transformer) |
| 추론 병목 | Diffusion 반복 횟수 — 스텝 수↑ = 품질↑ / 속도↓ | LM 순방향 패스 횟수 — 토큰 수에 비례 |
| 품질 vs 속도 | **품질 최우선** — 속도 타협 (GPU 전제) | **속도 최우선** — 품질 타협 (CPU 전제) |

---

## 9. 사용 시나리오별 추천

**영문 팟캐스트 / 오디오북 자동 생성**  
추천: VibeVoice  
90분 단일 세션, 4 스피커, 높은 자연스러움 → 장편 영어·중국어 콘텐츠 생산에 최적

**한국어 포함 다국어 프로덕션 서비스**  
추천: MOSS-TTS Nano  
10개+ 언어 네이티브, Code-switching → 글로벌 SaaS·앱 내 TTS 엔진으로 적합

**엣지 / CPU 환경 실시간 TTS**  
추천: MOSS-TTS Nano  
0.1B LM, GPU 불필요, 저지연 스트리밍 → IoT, 모바일, 브라우저 배포 가능

**연구 / 음성 합성 기반 기술 연구**  
추천: MOSS-TTS Nano  
완전 오픈소스, 파인튜닝 코드 공개 → 모델 개조·실험 자유도 높음

**고품질 음성 클로닝 (영어 중심)**  
추천: VibeVoice  
Diffusion 연속 공간 + 10초 프롬프트 → 표현력·억양 재현 우수

**Brity Meeting On-device 통역 TTS**  
추천: MOSS-TTS Nano  
한국어, CPU 실시간, 오픈소스 — 세 조건 동시 충족하는 유일한 후보

---

## 10. 결론 및 향후 계획

### 결론: MOSS Nano, 가능성은 있지만 아직 무겁다

#### VibeVoice

Microsoft Research의 연구 산출물. 음질·표현력 최고 수준이나 한국어 미지원, 코드 비공개, GPU 필수.  
Brity Meeting 요건과 맞지 않아 현 시점 적용 불가.

#### MOSS-TTS Nano

CPU 실시간 + 한국어 + 오픈소스 세 조건 동시 충족.  
Brity Meeting On-device 적용에 **가장 가능성이 큰 후보**.  
단, 그대로 구동하기에는 아직 꽤 무겁다.

> ⚠️ **현실적 제약 — CAT Tokenizer의 무게**  
> LM 자체는 0.1B로 가볍지만, 오디오 처리를 위한 CAT Tokenizer가 1.6B입니다.  
> 전체 스택을 단말에 상시 로딩하기에는 메모리 부담이 여전히 큽니다.  
> 실제 Galaxy 단말 및 저전력 PC에서의 RTF(Real-Time Factor) 측정이 필요합니다.

*두 모델은 경쟁보다 상호보완적 — 영어 콘텐츠 고품질 생성은 VibeVoice, 다국어·On-device 배포는 MOSS-TTS Nano*

### 향후 연구 방향

**01. CAT Tokenizer 경량화**  
INT4/INT8 양자화(Quantization), Knowledge Distillation 기법으로 1.6B Tokenizer의 크기를 줄일 수 있는지 탐색. 상위 코드북(1~4)만 사용하는 품질-속도 트레이드오프도 실험.

**02. Speaker Encoder 분리 최적화**  
화자 유사도를 유지하면서 가중치를 줄이는 경량 Speaker Encoder (ECAPA-TDNN 소형 버전) 조합 검토. 한국어 음성 데이터로 파인튜닝하는 방안도 함께 탐색.

**03. On-device 벤치마크 측정**  
Galaxy 단말 및 저전력 PC에서 실제 RTF(Real-Time Factor)를 측정해 병목 구간 파악. 목표: RTF < 1.0 (실시간 이하).

**04. Brity Meeting 파이프라인 통합 PoC**  
ASR → MT → MOSS Nano TTS 파이프라인 전체 지연(E2E latency)을 측정. 회의 몰입도를 해치지 않는 임계 지연(약 200ms) 달성 가능 여부 검증.

---

Voice Cloning TTS는 빠르게 발전 중인 분야입니다. 지금 당장 완전한 On-device 적용은 어렵더라도, 기술 방향을 미리 파악해 두면 좋은 시점이라고 생각합니다. 관심 있으신 분들의 의견이나 아이디어 공유 환영합니다.

---

**References**  
- VibeVoice — arXiv 2508.19205, Microsoft Research, 2025.08  
- MOSS-TTS / MOSS-TTS Nano — OpenMOSS Team, github.com/OpenMOSS, 2025~2026  
- CAT (Causal Audio Tokenizer) — MOSI.AI Tech Report

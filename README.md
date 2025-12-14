# Motion Retargeting Distillation Framework
## distilR²ET: R²ET 기반 Student 모델

---

## 요약 (Summary)

- **Teacher 모델(R²ET)로부터 knowledge distillation 방식으로 Student 모델을 학습**하여  
  motion retargeting 모델을 보다 경량화된 구조로 구현합니다.
- Mixamo 데이터셋을 활용해 **Teacher 추론 → Student 학습 → 정량 평가(MSE, Foot, Attractive, Repulsive)**  
  까지의 전체 파이프라인을 제공합니다.
- 단일 샘플 추론, 전체 데이터셋 평가, Seen/Unseen 분리 등  
  **연구 실험 및 재현을 위한 전체 실험 워크플로우**를 제공합니다.

---

## 개요 (Overview)

본 프로젝트는 **R²ET** Teacher 모델을 기반으로,  
Student 모델을 knowledge distillation 방식으로 학습하여 **실시간성 및 경량화** 달성을 통해,
**Motion Retargeting 분야의 Distillation Framework** 구축을 목표로 합니다.

**distillation loss**를 통해 teacher 모델의 shape-aware retargeting 능력을 학습하여
소스 캐릭터의 모션을 타겟 캐릭터의 스켈레톤 및 체형에 맞게 변환하면서 원본 모션의 의미를 유지하고
**foot contact**를 통해 물리적 사실성을 강화하도록 설계되었습니다.

**attractive / repulsive loss**를 측정하여 평가되어 Teacher 모델의 자기관통방지, 자기접촉유지 능력이 유지되었는지 평가되며,
**foot contact loss**를 측정하여 새로운 제약에 대한 fine-tuning이 잘 되었는지 평가되었습니다. 

**Inference time / model parameters**를 측정하여 모델 경량화 측면에서 추론 시간과 모델 크기에 대해 평가되었습니다. 

---

## 프로젝트 폴더 구조

```text
config/          - 학습 및 추론 설정 파일
datasets/        - !용량 문제로 인해 메일로 전달!
 ├─ mixamo/      - Mixamo 데이터셋
 └─ r2et/        - Teacher(R²ET) 추론 결과 (Student 학습용)
inference/
 ├─ teacher/     - Teacher 단일 추론 결과
 └─ student/     - Student 단일 추론 결과
metrics/
 └─ summary/     - JSON 형식 성능 평가 결과
outside-code/    - FBX/BVH 처리 및 SDF 관련 외부 코드
pretrain/        - 사전학습된 R²ET Teacher (checkpoint)
results/         - 모델 전체 추론 결과 !용량 문제로 인해 메일로 전달!
src/             - 모델 구조 및 기타 코드
work_dir/        - !용량 문제로 인해 메일로 전달!
 └─ distilr2et/  - Student 모델 학습 결과 (checkpoint)
```

### 기타 파일 및 폴더에 대하여

README에 명시되지 않은 일부 파일 및 폴더는  
프로젝트 진행 과정에서의 **시행착오(trial-and-error)** 및  
실험 기록 보존을 위해 **삭제하지 않았습니다**.

README에 기술된 파일 및 폴더만으로도  
**전체 실험 파이프라인의 실행 및 재현이 가능**하도록 구성되어 있습니다.

---

## 전체 파이프라인 한눈에 보기

```text
[1] 환경 세팅
    ├─ Conda 환경 생성
    ├─ Python 의존성 설치
    ├─ PyTorch + CUDA 설치
    └─ Blender 설치 (필수)
          ↓
[2] 데이터 준비
    ├─ Mixamo FBX 다운로드
    ├─ FBX → BVH 변환 (Blender)
    ├─ BVH → NPY 전처리
    └─ T-pose 기반 Shape 추출
          ↓
[3] (선택) 데이터 분할
    └─ Seen / Unseen Character & Motion 재구성
          ↓
[4] R²ET Teacher 추론
    └─ Student 학습용 Teacher 데이터 생성
          ↓
[5] Student 모델 학습 (distilR²ET)
          ↓
[6] 단일 샘플 추론
    ├─ Teacher 단일 추론
    └─ Student 단일 추론
          ↓
[7] 전체 데이터셋 추론
    ├─ Teacher BVH 생성
    └─ Student BVH 생성
          ↓
[8] 모델 평가 
    ├─ 경량화 평가 (Inference time + Model parameters)
    ├─ 정확도 빠른 평가 (MSE + Foot)
    └─ 정확도 전체 평가 (MSE + Foot + Attractive + Repulsive)
```

---

## Quick Start

> 아래 절차는 **전체 파이프라인 한눈에 보기**와 동일한 순서로 구성되어 있습니다.

### Step 1. 환경 세팅

#### 1-1. Conda 환경 생성

```bash
conda create -n motion-retarget python=3.9
conda activate motion-retarget
```

#### 1-2. Python 의존성 설치

```bash
pip install -r requirements.txt
```

#### 1-3. PyTorch + CUDA 설치

```bash
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 ^
  --index-url https://download.pytorch.org/whl/cu118
```

#### 1-4. Blender 설치 (필수)

Blender는 **FBX/BVH 변환 및 Shape 추출을 위해 필수**입니다.

```bat
"C:\Program Files\Blender Foundation\Blender 2.93\2.93\python\bin\python.exe" -m pip uninstall -y torch torchvision torchaudio
```

```bat
"C:\Program Files\Blender Foundation\Blender 2.93\2.93\python\bin\python.exe" -m pip install ^
  torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 ^
  --index-url https://download.pytorch.org/whl/cu118
```

---

### Step 2. 데이터 준비 (Mixamo + Blender)

```bash
blender -b -P ./datasets/fbx2bvh.py
python ./datasets/preprocess_q.py
blender -b -P ./datasets/extract_shape.py
```

> ⚠️ Mesh 주의사항  
> - 여러 Mesh가 있을 경우 **통합**하거나  
> - **Body Mesh가 가장 먼저 위치**해야 합니다.

---

### Step 3. (선택) Seen / Unseen 데이터 분할

```bash
python ./datasets/data_splitter.py [unseen_motion_ratio] [unseen_character_number] [target_character_name]
```

예시:
```bash
python ./datasets/data_splitter.py 0.2 1 Claire
```

---

### Step 4. R²ET Teacher 추론

```bash
python inference.py --config ./config/inference_cfg.yaml
```

---

### Step 5. Student 모델 학습 (distilR²ET)

```bash
python train_student.py --config ./config/train_student.yaml
```

---

### Step 6. 단일 샘플 추론

#### Teacher
```bash
python inference_bvh.py --config ./config/inference_bvh.yaml
```

#### Student
```bash
python inference_bvh_student.py --config ./config/inference_bvh_student.yaml
```

---

### Step 7. 전체 데이터셋 추론

```bash
python inference_all_bvh.py --config ./config/inference_all_bvh.yaml
python inference_all_bvh_student.py --config ./config/inference_all_bvh_student.yaml
```

> ⚠️ 해당 순서를 반드시 따라야 합니다.

---

### Step 8. 모델 평가 (JSON 출력)


#### 정확도 빠른 평가

```bash
python metrics/compare_inference_speed.py ^
  --teacher_config [teacher_model_config] ^   --student_config [student_model_config] ^   --bench_iters [n_iter] ^   --device [cpu/cuda:0] ^   --seq_len [sample_seq_len]
``

예시: cpu로 120 프레임 모션 100회 추론
```bash
python metrics/compare_inference_speed.py ^
  --teacher_config ./config/inference_bvh.yaml ^
  --student_config ./config/inference_bvh_student.yaml ^
  --bench_iters 100 ^
  --device cpu ^
  --seq_len 120
```

> 결과: ./metrics/summary/compare_inference_speed_120_cpu.json

#### 정확도 빠른 평가
```bash
python ./metrics/evaluation.py
```

> 결과: ./metrics/summary/eval_summary.json

#### 정확도 전체 평가
```bash
python ./metrics/total_evaluation.py
```

> 결과: ./metrics/summary/total_eval_summary.json

> ⚠️ 전체 평가는 최적화가 부족하여 시간이 오래 걸릴 수 있습니다.

---


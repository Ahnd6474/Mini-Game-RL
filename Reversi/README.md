# Reversi environment

외부 라이브러리 없이 사용할 수 있는 2인용 리버시 환경입니다.

## 기본 API

- 보드 값: 빈칸 `0`, 흑 `1`, 백 `-1`
- 행동: 행 우선 순서의 `0..63`, 패스는 `64`
- `observe()`: 보드, 현재 플레이어, 합법 행동과 마스크 반환
- `step(action)`: `(observation, rewards, terminated, info)` 반환
- `rewards`: 게임 중에는 양쪽 모두 `0`, 종료 시 승자 `1`, 패자 `-1`

```python
from Reversi import BLACK, WHITE, ReversiEnv

env = ReversiEnv()
while not env.terminated:
    observation = env.observe()
    action = observation.legal_actions[0]
    observation, rewards, terminated, info = env.step(action)

print(env.render())
print(rewards)
```

`observation.legal_actions`에는 둘 수 있는 칸이 없을 때 `PASS_ACTION` 하나가
들어갑니다. `env.legal_actions()`는 실제 착수 위치만 반환하므로 이 경우 빈
튜플입니다.

## 실행

저장소 루트에서 다음 명령을 실행합니다.

```powershell
python -m unittest Reversi.test_env
python -m Reversi.random_agent
```

## 병렬 학습 구조

`model.py`에는 서로 파라미터를 전혀 공유하지 않는 두 Transformer가
있습니다. 둘 다 기본적으로 64차원, 3-layer, 4-head입니다.

- `DVN`: 최종 64칸 소유권과 현재 플레이어의 승패 가치 지도학습
- `DPN`: 고정된 DVN의 negamax temporal delta를 advantage로 사용하는 PPO 학습

`batch_env.py`는 여러 게임을 `[N, 8, 8]` NumPy 배열 하나에 보관합니다.
합법 수와 뒤집기를 전체 게임에 동시에 계산하고, DPN 추론과 두 모델의
학습 minibatch만 GPU에서 수행합니다. 학습 sample은 GPU에 한 번 올린 후
회전·반사 증강도 GPU에서 실행합니다.

DPN rollout에서는 무작위 D4 변환을 먼저 적용한 관측에서 행동을 샘플링하고
그 상태의 old log-probability를 함께 저장합니다. 따라서 PPO likelihood ratio가
대칭 증강 때문에 어긋나지 않습니다. 기본값은 4 PPO epoch, clip range 0.2,
target KL 0.02, DPN learning rate `1e-4`입니다. DVN은 기존 지도학습을 그대로
유지합니다.

DVN은 기본적으로 현재 rollout과 과거 상태 reservoir를 1:1로 섞어 학습합니다.
reservoir는 최대 262,144개 상태를 균등 표본으로 유지하며 체크포인트에도 함께
저장됩니다. DPN self-play의 절반은 과거 tree benchmark 최고 정책 중 하나를
상대 색으로 사용합니다. 과거 정책의 행동은 PPO loss에서 제외하지만, 그 게임의
최종 보드와 승패는 DVN 지도학습 데이터로 사용합니다.

## 선형 평가기와 트리 기준점

본 학습 전에 64칸만 입력받는 bias 없는 선형 승패 평가기를 만들 수 있습니다.
보드는 항상 현재 플레이어 기준으로 변환하므로 색을 뒤집으면 logit도 정확히
반대가 됩니다. 랜덤 대국은 NumPy로 병렬 생성하고, 학습 중 무작위 D4 증강과
epoch별 대칭 가중치 평균을 적용합니다.

```powershell
.venv\Scripts\python.exe -m Reversi.train_linear `
  --games 16384 --epochs 12 --batch-size 8192
```

결과는 기본적으로 `Reversi/checkpoints/linear_value.json`에 저장됩니다. 이
가중치를 비트보드 negamax/alpha-beta 탐색기의 leaf 평가 함수로 사용합니다.

```powershell
.venv\Scripts\python.exe -m Reversi.tree_agent `
  --weights Reversi/checkpoints/linear_value.json --depth 4
```

탐색기는 transposition table, alpha-beta pruning, 코너 우선 move ordering을
사용합니다. `--weights`를 생략하면 내장 위치 가중치로 바로 실행됩니다.

깊이 5 트리 제출본은 [submission_tree_depth5.py](submission_tree_depth5.py)입니다.
학습된 64칸 선형 가중치, 비트보드 규칙, alpha-beta와 transposition table을
한 파일에 내장하며 외부 패키지가 필요 없습니다. 누적 CPU 4.65초 안전 한도와
항상 합법 수를 반환하는 fallback이 있어 5초 전체 제한을 넘기지 않습니다.

고정된 무작위 중반 상태에서 깊이별 속도를 재려면 다음을 실행합니다.

```powershell
.venv\Scripts\python.exe -m Reversi.benchmark_tree `
  --weights Reversi/checkpoints/linear_value.json
```

```python
import torch

from Reversi import ReversiEnv
from Reversi.model import DPN, DVN

env = ReversiEnv()
obs = env.observe()
device = torch.device("cuda")
dvn = DVN().to(device)
dpn = DPN().to(device)

policy = dpn(
    torch.tensor([obs.board], device=device),
    torch.tensor([obs.current_player], device=device),
    torch.tensor([obs.legal_action_mask], device=device),
)
```

## GPU 설치와 학습

현재 로컬 NVIDIA GPU용으로 CUDA 13.0 PyTorch wheel을 사용합니다.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
.venv\Scripts\python -m unittest discover -s Reversi -t . -v
.venv\Scripts\python.exe -m Reversi.train --iterations 100 --games 1024 --batch-size 1024
```

`--device auto`가 기본값이며 CUDA가 있으면 BF16 AMP를 자동으로 사용합니다.
체크포인트는 기본적으로 `Reversi/checkpoints/latest.pt`에 저장됩니다.

현재 RTX 5060 Laptop 8GB에서 측정한 batch 처리량 최적점은 게임 1,024개,
minibatch 1,024입니다. PPO 기본 설정은 DVN 1 epoch, DPN 4 epoch이므로
단일 policy epoch 벤치마크보다 처리량이 낮습니다. 다른 GPU에서 다시 측정하려면:

```powershell
.venv\Scripts\python.exe -m Reversi.benchmark
```

고정 검증셋, 트리 기준 대국, JSONL 지표, SVG 학습 곡선까지 포함한 실험은:

```powershell
.venv\Scripts\python.exe -m Reversi.experiment `
  --iterations 30 --games 1024 --eval-every 5
```

기존 PPO 체크포인트에서 replay/policy-pool 학습을 이어가려면:

```powershell
.venv\Scripts\python.exe -m Reversi.experiment `
  --resume Reversi/runs/ppo_30/selected.pt `
  --run-dir Reversi/runs/ppo_replay_pool `
  --iterations 15 --entropy-weight 0.02
```

결과는 기본적으로 `Reversi/runs/latest`에 저장됩니다. 트리 평가는 고정된
오프닝마다 DPN이 흑과 백을 한 번씩 맡으므로 선후공 편향을 줄입니다.

## 제출 코드 생성

각 학습 iteration이 끝날 때 다음 두 파일을 자동으로 갱신합니다.

- `Reversi/checkpoints/latest.pt`: DVN/DPN과 optimizer를 포함한 학습 체크포인트
- `Reversi/checkpoints/submission.py`: DPN 가중치를 내장한 단일 제출 소스

제출 소스는 PyTorch, NumPy, 별도 모델 파일을 사용하지 않습니다. FP16
가중치와 DPN Transformer의 최소 실행 코드를 한 파일에 넣으며, 실행 시에는
Python 표준 라이브러리만으로 가중치를 복원하고 모델을 추론합니다.

기존 체크포인트에서 다시 만들 수도 있습니다.

```powershell
.venv\Scripts\python.exe -m Reversi.export_submission `
  --checkpoint Reversi/checkpoints/latest.pt `
  --output Reversi/submission.py
```

제출 환경과 동일하게 site-packages 없이 확인하려면 생성 파일을
`python -S Reversi/submission.py`로 실행할 수 있습니다.

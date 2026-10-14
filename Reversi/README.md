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
- `DPN`: 고정된 DVN의 negamax temporal delta로 65개 행동 policy 학습

`batch_env.py`는 여러 게임을 `[N, 8, 8]` NumPy 배열 하나에 보관합니다.
합법 수와 뒤집기를 전체 게임에 동시에 계산하고, DPN 추론과 두 모델의
학습 minibatch만 GPU에서 수행합니다. 학습 sample은 GPU에 한 번 올린 후
회전·반사 증강도 GPU에서 실행합니다.

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

현재 RTX 5060 Laptop 8GB에서 측정한 기본 처리량 최적점은 게임 1,024개,
minibatch 1,024, DVN/DPN 각 1 epoch입니다. 다른 GPU에서 다시 측정하려면:

```powershell
.venv\Scripts\python.exe -m Reversi.benchmark
```

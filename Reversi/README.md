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

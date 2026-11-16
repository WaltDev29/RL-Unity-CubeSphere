# Unity-Python 물리 기반 강화학습(RL) 플랫폼

Unity의 물리 엔진(PhysX)을 활용한 3D 시뮬레이션 환경 서버와 Python 기반 강화학습(SAC / TD3) 클라이언트 시스템으로 구성된 통합 강화학습 플랫폼입니다.

구체(Sphere) 에이전트에 연속적인 **물리적 Force**를 가하여, 바닥에서 떨어지지 않고 목표 정육면체(Cube)에 안전하고 신속하게 접촉하도록 학습합니다.

---

## 📌 주요 특징

1. **Unity 물리 환경 및 고속 스텝 동기화**
   * 순간이동 방식이 아닌 `Rigidbody.AddForce` 기반의 현실적인 관성 물리 제어
   * `Physics.simulationMode = Script` 및 `Physics.Simulate()`를 이용한 수동 물리 진행: Python과 Unity 간 네트워크 대기 시간 중 물리 시간이 흐르지 않아 완벽한 재현성 및 무제한 고속 학습 지원
2. **독립적 TCP 소켓 통신 시스템**
   * 패키지 버전 종속성이 큰 ML-Agents 대신 고성능 4바이트 길이 접두(Length-prefixed) JSON TCP 소켓 프로토콜 구현
   * 환경 중립적 인터페이스(`IRLEnvironment`)로 설계되어 향후 다른 Unity RL 환경에도 즉시 재사용 가능
3. **PyTorch 기반 연속 행동 강화학습 (SAC & TD3)**
   * **SAC (Soft Actor-Critic):** 최대 엔트로피 프레임워크 및 자동 온도 조절($\alpha$)이 포함된 주 알고리즘 (안정적 탐색 및 고효율 샘플링)
   * **TD3 (Twin Delayed DDPG):** 비교 실험용 2차 알고리즘 지원
   * 위치 불변성 및 위험 인지를 결합한 **6차원 관측 공간(Observation Space)** 최적화
   * 보상 해킹(제자리 진동)을 차단하는 **잠재 함수 기반 접근 보상(Potential-based Reward Shaping)**
4. **실시간 모니터링 & 제어 대시보드 (PySide6 + pyqtgraph)**
   * GUI와 학습 프로세스가 완전히 분리된 멀티프로세스 아키텍처: GUI 지연이나 강제 종료가 학습에 전혀 영향을 주지 않음
   * 에피소드 보상, 이동 평균, Loss, 성공률, 에피소드 길이 실시간 렌더링
   * 학습 시작, 일시정지, 재개, 중단, 이전 체크포인트 기반 이어서 학습, 모델 평가(Evaluation) 지원
   * **런타임 시뮬레이션 속도 조절(SpeedSync):** 학습 중에도 GUI에서 실시간(1x) ↔ 최고속(무제한) 즉시 전환 가능
5. **안정적인 모델 관리 & 환경설정**
   * `.env` 단일 출처 기반의 모든 하이퍼파라미터 및 환경 변수 중앙 관리 (`.env.example` 제공)
   * Best Model(주기적 결정론적 평가 기반), Final Model, 정기 체크포인트 자동 보존
   * 예외 발생 또는 비정상 종료 시 긴급 체크포인트 보존 로직 내장

---

## 🛠️ 기술 스택 및 요구 사양

| 구분 | 버전 / 기술 | 비고 |
|---|---|---|
| **OS** | Windows 10 / 11 | - |
| **Unity** | Unity 2022.3.62f3 (LTS) | Built-in Render Pipeline |
| **Python** | Python 3.12 (Anaconda 가상환경 `RL`) | - |
| **Deep Learning** | PyTorch 2.11.0+cu128 | NVIDIA GPU (RTX 3080 권장) 및 CPU 지원 |
| **GUI** | PySide6, pyqtgraph | 비동기 멀티프로세스 |
| **기타 라이브러리** | gymnasium, numpy, matplotlib, python-dotenv, pytest | - |

---

## 📁 프로젝트 구조

```text
261114_assign/
├── Assets/
│   ├── Editor/SceneBuilder.cs       # Unity 에디터 메뉴 및 배치 빌드 스크립트
│   ├── Materials/                   # 에이전트, 타깃, 바닥 머티리얼
│   ├── Scenes/RLEnv.unity           # 강화학습 메인 씬
│   └── Scripts/
│       ├── Env/
│       │   ├── CubeSphereEnv.cs     # 물리 시뮬레이션, 관측, 보상, 초기화 로직
│       │   └── SphereContactSensor.cs # 구체 충돌 감지 센서
│       └── Net/
│           ├── EnvParams.cs         # Unity-Python 공유 환경 파라미터 구조체
│           ├── Protocol.cs          # 통신 메시지 규격 및 IRLEnvironment 인터페이스
│           └── TcpEnvServer.cs      # 고성능 논블로킹 TCP 소켓 서버
├── python/
│   ├── gui/
│   │   ├── app.py                   # PySide6 대시보드 메인 UI
│   │   └── runner.py                # 학습/평가 백그라운드 프로세스 관리
│   ├── rl/
│   │   ├── agents/                  # SAC, TD3, TwinQ, GaussianActor 구현체
│   │   ├── envs/                    # Gymnasium 호환 클라이언트 및 세션 매니저
│   │   ├── train/                   # Trainer, Evaluator, Rollout 오케스트레이션
│   │   ├── utils/                   # 체크포인트, 로거, 그래프 시각화 모듈
│   │   ├── cli.py                   # CLI 진입점 (train, eval, plot, compare)
│   │   └── config.py                # .env 로드 및 무결성 검증
│   └── tests/                       # 단위 테스트 및 가상 서버(FakeUnityServer)
├── .env.example                     # 환경 설정 템플릿 파일
├── .gitignore                       # Unity 및 Python Git 제외 설정
└── README.md                        # 본 문서
```

---

## 🚀 시작하기

### 1. Python 환경 구축

Anaconda 프롬프트 또는 PowerShell에서 실행합니다:

```powershell
# 가상환경 생성 및 활성화
conda create -n RL python=3.12 -y
conda activate RL

# PyTorch (CUDA 지원 버전) 설치
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 필수 라이브러리 설치
pip install numpy matplotlib gymnasium PySide6 pyqtgraph python-dotenv pytest

# 환경 설정 파일 복사 (.env 생성)
Copy-Item .env.example .env
```

### 2. Unity 빌드 준비

프로젝트 루트에 Unity 에디터가 설치된 상태에서 배치 명령어로 씬과 실행 파일을 자동 빌드할 수 있습니다:

```powershell
# 씬 생성 (Assets/Scenes/RLEnv.unity)
& "C:\Program Files\Unity\Hub\Editor\2022.3.62f3\Editor\Unity.exe" -batchmode -nographics -quit -projectPath . -executeMethod RLPlatform.EditorTools.SceneBuilder.BuildSceneBatch

# 독립 실행형 플레이어 빌드 (Builds/RLEnv/RLEnv.exe)
& "C:\Program Files\Unity\Hub\Editor\2022.3.62f3\Editor\Unity.exe" -batchmode -nographics -quit -projectPath . -executeMethod RLPlatform.EditorTools.SceneBuilder.BuildPlayer
```
*(또는 Unity Editor에서 프로젝트를 열고 상단 메뉴의 `RL > Build Scene`, `RL > Build Windows Player`를 클릭해도 됩니다.)*

---

## 🎮 실행 방법

### 방법 1. GUI 대시보드로 실행 (권장)

```powershell
conda activate RL
cd d:\course\unity\261114_assign\python
python -m gui.app
```

* **학습 시작:** `학습 시작` 버튼 클릭 시 `.env`의 설정대로 Unity가 자동 백그라운드 실행되며 실시간 학습이 진행됩니다.
* **관찰 모드:** 기본 설정(`.env`의 `UNITY_HEADLESS=0`) 시 Unity 창이 함께 팝업되어 구체의 움직임을 직접 눈으로 확인할 수 있습니다.
* **속도 조절:** GUI 하단의 스핀박스 또는 프리셋 버튼(`실시간 10`, `5배 50`, `최고속 0`)을 통해 학습 도중 언제든 시뮬레이션 속도를 조절할 수 있습니다.
* **추가 학습:** `중단된 run 이어서 학습…` 버튼으로 이전 체크포인트 폴더를 지정해 중단된 에피소드부터 계속 학습할 수 있습니다.
* **평가 실행:** 학습 완료 후 `models/best.pt`를 선택하고 `평가 실행`을 누르면 무작위 탐색 없이 최적화된 정책으로 에이전트의 주행을 테스트합니다.

### 방법 2. CLI 명령어로 실행

```powershell
conda activate RL
cd d:\course\unity\261114_assign\python

# 1. 신규 학습 실행
python -m rl.cli train --episodes 1500

# 2. TD3 알고리즘으로 비교 학습 실행
python -m rl.cli train --algorithm td3 --episodes 1500

# 3. 중단된 체크포인트에서 이어서 학습 (Resume)
python -m rl.cli train --resume ..\runs\train_sac_2026xxxx_xxxxxx --episodes 2000

# 4. 학습된 모델 평가 (20 에피소드)
python -m rl.cli eval --model ..\runs\train_sac_2026xxxx_xxxxxx\models\best.pt --episodes 20

# 5. 여러 실험 결과 그래프 비교 시각화
python -m rl.cli compare ..\runs\run_sac ..\runs\run_td3 --out comparison.png
```

---

## 🧠 상태 공간 및 보상 체계

### Observation Space (10차원, 정규화)
| Index | 상태 내용 | 정규화 수식 | 설계 의도 |
|:---:|---|---|---|
| **0 ~ 1** | 목표 큐브 상대 위치 ($x, z$) | $(p_{\text{target}} - p_{\text{sphere}}) / \text{floor\_size}$ | 목표의 방향과 거리 정보를 동시에 제공하며, 절대 좌표 불변성 보장 |
| **2 ~ 3** | 구체 선형 속도 ($v_x, v_z$) | $v / \text{max\_speed}$ | 물리 관성 제어 및 목표 도달 전 감속 제동 타이밍 인지 |
| **4 ~ 5** | 구체 절대 위치 ($x, z$) | $p_{\text{sphere}} / (\text{floor\_size} / 2)$ | 바닥 경계선 접근 시 추락 위험 인지 |
| **6 ~ 7** | 장애물 벽 표면 상대 위치 ($w_x, w_z$) | $(p_{\text{closest\_wall}} - p_{\text{sphere}}) / \text{floor\_size}$ | 벽과의 최단 거리 및 충돌 위험 인지 |
| **8 ~ 9** | 우회 통로(좌/우 통로 중 최근접) 상대 벡터 | $(p_{\text{opening}} - p_{\text{sphere}}) / \text{floor\_size}$ | 벽에 가로막혔을 때 우회할 탈출구 방향 안내 |

### Reward Function
$$r_t = w_{\text{approach}} \cdot \max(0, d_{\text{best}} - d_t) - c_{\text{step}} - c_{\text{action}} \|a_t\|^2 + R_{\text{terminal}}$$

* **역대 최단 거리 갱신 보상 ($w_{\text{approach}}=1.0$):** 에피소드 내에서 달성한 최소 거리 신기록을 갱신한 양만큼만 보상 지급. 뒤로 물러났다가 다시 전진하며 보상을 반복 취득하는 행위(보상 해킹/뱅뱅 도는 현상) 원천 차단
* **스텝 시간 페널티 ($c_{\text{step}}=0.01$):** 최단 경로/시간 내 도달 유도
* **행동 크기 페널티 ($c_{\text{action}}=0.001$):** 불필요한 과도한 Force 입력 억제
* **성공 보상 ($R_{\text{success}}=+10.0$):** 큐브와 물리적 접촉 시 부여 및 에피소드 종료
* **추락 페널티 ($R_{\text{fall}}=-10.0$):** 바닥 경계 밖으로 떨어질 시 부여 및 에피소드 종료

---

## 📊 결과물 확인

모든 결과물은 `runs/<run_id>/` 디렉터리에 자동으로 저장됩니다:
* `models/best.pt`: 결정론적 주기 평가에서 가장 높은 성공률을 기록한 최고 성능 모델 가중치
* `models/final.pt`: 학습 종료 시점의 최종 모델
* `checkpoints/latest.pt`: Replay Buffer 및 옵티마이저가 포함된 전체 재개용 체크포인트
* `plots/*.png`: 누적 보상, 이동 평균 보상, 손실(Loss), 성공률(Success Rate), 스텝 길이 시각화 차트
* `episodes.csv`: 에피소드별 로우 데이터 (Pandas 등을 통한 추가 분석용)

---

## 🧪 테스트 실행

Unity 환경 없이 Python 통신 프로토콜 및 알고리즘 단위 동작을 검증하려면:

```powershell
cd d:\course\unity\261114_assign\python
python -m pytest tests -q
```
*(가상 Unity 소켓 서버를 통해 14개의 단위 테스트가 20여 초 만에 전원 통과합니다.)*

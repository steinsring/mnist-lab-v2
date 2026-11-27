# 과제 - 신경망을 이용한 손글씨 숫자 인식 

딥러닝 프레임워크 없이 NumPy로 2층 신경망의 핵심 계산을 직접 구현하는 교육용 과제입니다.

목표는 MNIST 정확도 경쟁이 아닙니다. 다음 흐름을 코드, 수식, 배열 shape으로 설명하고 핵심 forward/backward 계산을 직접 작성하는 것이 목표입니다.

```text
입력 -> Affine -> ReLU -> Affine -> logits -> Softmax + Cross Entropy
     <-   dx   <-  dx  <-   dx   <- gradient

Forward -> Loss -> Backward -> Update
```

세부 과제 안내는 함께 배포된 과제 문서 「과제 - 신경망을 이용한 손글씨 숫자 인식」을 먼저 읽으세요.

## 빠른 시작

```bash
conda create -n mnist-nn python=3.12 -y
conda activate mnist-nn
pip install -r requirements.txt
python download_mnist.py

pytest tests/ -v
python tools/selfcheck.py --day 1
```

처음 전체 테스트가 실패하는 것은 정상입니다. 현재 Day의 TODO를 직접 구현하고 해당 테스트부터 통과시키세요.

모든 명령은 프로젝트 루트(`README.md`가 있는 폴더)에서 실행합니다.

### Mac

1. [Miniconda](https://docs.conda.io/en/latest/miniconda.html)를 설치합니다. Apple Silicon은 `arm64`, Intel Mac은 `x86_64` 설치 파일을 선택합니다.
2. 터미널을 새로 열고 위 "빠른 시작" 명령을 순서대로 실행합니다.

### Windows

1. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) Windows 설치 파일을 설치합니다.
2. 시작 메뉴에서 **Anaconda Prompt (miniconda3)** 를 열고 프로젝트 폴더로 이동합니다.
   ```bat
   cd C:\path\to\mnist-lab
   ```
3. 위 "빠른 시작" 명령을 순서대로 실행합니다. PowerShell에서 `conda activate`가 동작하지 않으면 `conda init powershell` 실행 후 터미널을 다시 엽니다.

### Colab

`mnist_lab.ipynb`를 Colab에서 열고 맨 위 환경설정 셀을 실행하면 저장소 URL과 GitHub token을 묻습니다. 저장소 URL은 `https://github.com/USERNAME/mnist-lab`처럼 입력하고, 공개 저장소라면 token 입력란에서 Enter를 누릅니다.

### 데이터

`python download_mnist.py`가 MNIST를 `data/mnist.npz`에 저장합니다. 노트북이나 `load_mnist()`를 처음 실행할 때도 파일이 없으면 자동으로 내려받습니다. `data/mnist.npz`는 `.gitignore`에 포함되어 있으므로 commit하지 않습니다.

## 직접 구현하는 범위

| Day | 파일 | 구현 대상 |
| ---: | --- | --- |
| 1 | `src/layers.py` | `Affine.forward` |
| 1 | `src/activations.py` | `ReLU.forward` |
| 1 | `src/losses.py` | `SoftmaxWithLoss.forward` |
| 2 | `src/losses.py` | `SoftmaxWithLoss.backward` |
| 2 | `src/activations.py` | `ReLU.backward` |
| 2 | `src/layers.py` | `Affine.backward` |
| 3 | `src/optimizers.py` | `SGD.update` |
| 3 | `src/training.py` | `train()`의 Forward, Loss, Backward, Update 연결 |

`TwoLayerNet`, 데이터 처리, 미니배치, Adam, BatchNorm, Dropout, gradient check, 평가와 그래프 코드는 제공됩니다. 제공 코드를 다시 작성할 필요는 없지만 핵심 함수가 어디서 호출되는지는 읽어야 합니다.

## 3일 진행

| Day | 주제 | 주요 검증 |
| ---: | --- | --- |
| 1 | Forward, logits, loss | 작은 배열 값, shape, 초기 loss |
| 2 | Backward와 chain rule | backward 실제 값, gradient check |
| 3 | SGD와 end-to-end 학습 | Synthetic Smoke, 전체 테스트, MNIST baseline 1회 |

- 정확도 임계값과 learning-rate·optimizer·hidden-size 비교는 필수가 아닙니다.
- 테스트를 빨리 통과했다면 추가 실험보다 수식, shape, 중간값, gradient 흐름을 먼저 설명하세요.
- 실제 MNIST에서는 고정 baseline을 한 번 실행해 loss 감소와 관찰 정확도만 기록합니다.

## AI 활용

AI는 수식과 shape 설명, NumPy API 확인, 자신이 작성한 코드의 오류 메시지 해석에 사용할 수 있습니다. TODO 함수 전체 구현이나 테스트 정답 코드를 생성해 붙여 넣는 용도로 사용하지 않습니다. 받은 설명은 작은 배열, 단위 테스트, gradient check 중 하나로 직접 검증하고 `LEARNING_LOG.md`와 `REPORT.md`에 기록합니다.

## 완료 기준

```bash
python tools/selfcheck.py --all
```

다음을 모두 완료하면 과제를 마친 것입니다.

- 전체 테스트와 gradient check 통과
- Synthetic Smoke에서 loss 감소와 parameter 변화 확인
- MNIST baseline 1회 실행과 loss 변화 기록
- `REPORT.md`와 AI·참고 기록 작성
- 핵심 계산을 코드 없이 수식과 shape으로 설명

높은 정확도나 여러 설정 실험은 완료 기준이 아닙니다.

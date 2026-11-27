# -*- coding: utf-8 -*-
"""일차별 TODO, 테스트, 개념 질문을 한 번에 확인하는 제공 도구."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DAYS = {
    1: {
        "title": "Forward, logits, loss",
        "targets": [
            "tests/test_affine.py::TestAffineForward",
            "tests/test_relu.py::TestReLUForward",
            "tests/test_softmax_with_loss.py::TestSoftmaxWithLossForward",
        ],
        "todos": [
            "layers.Affine.forward",
            "activations.ReLU.forward",
            "losses.SoftmaxWithLoss.forward",
        ],
        "questions": [
            "Affine.forward가 입력 x를 저장해야 하는 이유는 무엇인가?",
            "logits와 확률은 어떻게 다른가?",
            "모든 logits가 같을 때 loss가 ln(10)인 이유는 무엇인가?",
        ],
    },
    2: {
        "title": "Backward와 chain rule",
        "targets": [
            "tests/test_softmax_with_loss.py",
            "tests/test_relu.py",
            "tests/test_affine.py",
            "tests/test_gradient_check.py",
        ],
        "todos": [
            "losses.SoftmaxWithLoss.backward",
            "activations.ReLU.backward",
            "layers.Affine.backward",
        ],
        "questions": [
            "dW, db, dx의 shape는 각각 무엇과 같아야 하는가?",
            "ReLU backward에서 gradient가 0이 되는 위치는 어디인가?",
            "gradient check는 어떤 두 gradient를 비교하는가?",
        ],
    },
    3: {
        "title": "SGD와 end-to-end 학습",
        "targets": ["tests/test_sgd.py", "tests/test_training_smoke.py"],
        "todos": ["optimizers.SGD.update", "training.train의 네 단계 연결"],
        "questions": [
            "SGD가 파라미터 배열을 제자리에서 갱신해야 하는 이유는 무엇인가?",
            "Forward, Loss, Backward, Update가 이 순서인 이유는 무엇인가?",
            "loss 감소가 일반화 성능까지 보장하지 않는 이유는 무엇인가?",
        ],
    },
}


def run_tests(targets):
    command = [sys.executable, "-m", "pytest", *targets, "-q", "--no-header", "--tb=no"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    failures = [line for line in lines if line.startswith("FAILED")]
    for line in failures[:8]:
        print("  " + line.replace("FAILED ", "실패: ").split(" - ")[0])
    if len(failures) > 8:
        print(f"  ... 외 {len(failures) - 8}개")
    if not lines:
        print("  => pytest를 실행하지 못했습니다. pip install -r requirements.txt를 확인하세요.")
        if result.stderr.strip():
            print("  " + result.stderr.strip().splitlines()[-1])
        return False
    print(f"  => {lines[-1]}")
    if failures:
        first_failure = failures[0].replace("FAILED ", "").split(" - ")[0]
        print(f"\n  첫 실패를 자세히 보려면: pytest {first_failure} -x -q")
    return result.returncode == 0


def print_flow_trace():
    try:
        import numpy as np

        from network import TwoLayerNet

        model = TwoLayerNet(hidden_size=16, seed=0)
        value = np.zeros((2, 784))
        print("\n[forward shape trace]")
        print(f"input          {value.shape}")
        for layer in model.layers:
            value = layer.forward(value)
            print(f"{type(layer).__name__:14s} {value.shape}")
        return True
    except NotImplementedError as error:
        print(f"\n[shape trace] 아직 구현되지 않은 부분: {error}")
        return False
    except Exception as error:
        print(f"\n[shape trace] 실패: {type(error).__name__}: {error}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if args.all:
        print("=" * 64)
        print("전체 테스트와 forward shape trace")
        print("=" * 64)
        tests_ok = run_tests(["tests/"])
        trace_ok = print_flow_trace()
        raise SystemExit(0 if tests_ok and trace_ok else 1)

    if not args.day or args.day not in DAYS:
        parser.error("--day 1~3 또는 --all을 지정하세요.")

    day = DAYS[args.day]
    print("=" * 64)
    print(f"Day {args.day} - {day['title']}")
    print("=" * 64)
    print("\n[오늘의 TODO]")
    for todo in day["todos"]:
        print(f"  - {todo}")
    print("\n[테스트]")
    ok = run_tests(day["targets"])
    print("\n[코드를 보지 않고 답하세요]")
    for index, question in enumerate(day["questions"], 1):
        print(f"  {index}. {question}")
    trace_ok = print_flow_trace() if args.trace else True
    print("\n" + "-" * 64)
    if ok:
        print("테스트와 질문 설명이 모두 되면 오늘 핵심 목표를 완료했습니다.")
    else:
        print("실패한 값과 shape를 작은 배열로 다시 계산해 보세요.")

    raise SystemExit(0 if ok and trace_ok else 1)


if __name__ == "__main__":
    main()

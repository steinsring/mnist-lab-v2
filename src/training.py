# -*- coding: utf-8 -*-
"""
학습 루프, 검증 도구, 평가, 시각화.

[학생 구현 대상]
- train() 안의 TODO 1~4 (Forward -> Loss -> Backward -> Update 연결)

[제공 코드 - 구현하지 않습니다]
- iterate_minibatches, gradient_check, evaluate, plot_loss_history
"""

import time

import numpy as np


def iterate_minibatches(x, y, batch_size, shuffle=True, rng=None):
    """
    (제공 코드) 데이터를 섞어 미니배치 단위로 잘라 내보내는 generator.

    Args:
        x: (N, 784), y: (N,)
        batch_size: 한 배치의 샘플 수
        shuffle: True면 매번 순서를 섞습니다.
        rng: numpy Generator. None이면 새로 만듭니다.

    Yields:
        (x_batch, y_batch)
    """
    if rng is None:
        rng = np.random.default_rng()

    indices = rng.permutation(len(x)) if shuffle else np.arange(len(x))
    for start in range(0, len(x), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]


def train(model, criterion, optimizer, x_train, y_train,
          epochs=10, batch_size=128, seed=0, verbose=True):
    """
    미니배치 학습 루프.

    반복문과 데이터 처리는 이미 작성되어 있습니다.
    학생이 채워야 할 것은 신경망 학습의 핵심 네 단계를 연결하는 부분입니다.

        Forward -> Loss -> Backward -> Update

    Args:
        model: TwoLayerNet
        criterion: SoftmaxWithLoss
        optimizer: SGD 또는 Adam
        epochs: 전체 데이터를 몇 번 반복할지
        batch_size: 미니배치 크기

    Returns:
        loss_history: epoch별 평균 손실 리스트
    """
    # 아래 한 줄은 구현을 시작할 때 삭제하세요.
    raise NotImplementedError("train()의 TODO 1~4를 구현하세요.")

    rng = np.random.default_rng(seed)
    loss_history = []
    start_time = time.time()

    for epoch in range(epochs):
        batch_losses = []

        for x_batch, y_batch in iterate_minibatches(x_train, y_train, batch_size, rng=rng):
            # TODO 1 (Forward): model에 x_batch를 학습 모드로 통과시켜 logits를 얻으세요.
            #                   힌트: model.forward(..., train=True)
            logits = None

            # TODO 2 (Loss): criterion으로 logits와 y_batch의 손실을 계산해 loss에 담으세요.
            loss = None

            # TODO 3 (Backward): criterion.backward()로 출력층 gradient를 만든 뒤
            #                    model.backward()에 넘기세요. 이 호출로 model.grads가 채워집니다.

            # TODO 4 (Update): optimizer로 model.params를 model.grads만큼 갱신하세요.

            batch_losses.append(loss)

        epoch_loss = float(np.mean(batch_losses))
        loss_history.append(epoch_loss)

        if verbose:
            elapsed = time.time() - start_time
            print(f"epoch {epoch + 1:3d}/{epochs}  loss {epoch_loss:.4f}  ({elapsed:.1f}s)")

    return loss_history


def gradient_check(model, criterion, x, y, num_checks=8, eps=1e-5, tol=1e-6,
                   atol=1e-8, seed=0, verbose=True):
    """
    (제공 코드) 학생이 구현한 backward가 수학적으로 맞는지 검사합니다.

    파라미터 하나를 아주 조금(eps) 올렸다 내렸다 하면서 손실 변화를 직접 측정한 값
    (수치 미분)과, backward가 계산한 gradient(해석적 미분)를 비교합니다.
    두 값이 거의 같다면 backward 구현이 옳다는 강한 증거입니다.

        (loss(param + eps) - loss(param - eps)) / (2 * eps)  ==  grads[key]

    Args:
        num_checks: 파라미터별로 무작위로 몇 개 위치를 검사할지
        tol: 허용 상대 오차
        atol: 두 gradient의 차이가 이 값보다 작으면 같다고 본다.
              gradient가 원래 0인 파라미터는 상대 오차를 계산해도 의미가 없기 때문이다.
              (예: BatchNorm 바로 앞의 bias는 평균이 제거되므로 gradient가 정확히 0이다)

    Returns:
        (ok, max_relative_error)

    주의: Dropout은 매번 다른 mask를 뽑아 손실이 흔들리므로 검사할 수 없습니다.
          use_dropout=False인 모델로 검사하세요.
    """
    if getattr(model, "use_dropout", False):
        raise ValueError(
            "gradient_check는 Dropout이 켜진 모델에서 사용할 수 없습니다. "
            "use_dropout=False로 만든 모델로 검사하세요."
        )

    rng = np.random.default_rng(seed)

    criterion.forward(model.forward(x, train=True), y)
    model.backward(criterion.backward())
    analytic = {key: np.array(value, copy=True) for key, value in model.grads.items()}

    max_rel_error = 0.0
    for key, param in model.params.items():
        for _ in range(num_checks):
            index = tuple(int(rng.integers(0, size)) for size in param.shape)
            original = param[index]

            param[index] = original + eps
            loss_plus = criterion.forward(model.forward(x, train=True), y)

            param[index] = original - eps
            loss_minus = criterion.forward(model.forward(x, train=True), y)

            param[index] = original

            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic_value = analytic[key][index]

            abs_diff = abs(numeric - analytic_value)
            if abs_diff <= atol:
                # 둘 다 사실상 0이면 상대 오차는 의미가 없으므로 통과로 본다.
                rel_error = 0.0
            else:
                rel_error = abs_diff / (abs(numeric) + abs(analytic_value))

            max_rel_error = max(max_rel_error, rel_error)

            if verbose and rel_error > tol:
                print(f"  [{key}]{index} 수치={numeric:+.8f} 해석적={analytic_value:+.8f} "
                      f"상대오차={rel_error:.2e}")

    ok = max_rel_error <= tol
    if verbose:
        status = "통과" if ok else "실패"
        print(f"gradient check {status}: 최대 상대 오차 {max_rel_error:.2e} (허용 {tol:.0e})")

    return ok, max_rel_error


def evaluate(model, x, y):
    """(제공 코드) 정확도(%)와 총 파라미터 수를 반환합니다."""
    logits = model.predict(x)
    accuracy = float(np.mean(np.argmax(logits, axis=1) == y) * 100)
    total_params = sum(param.size for param in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history, label=None):
    """(제공 코드) epoch별 손실 곡선을 그립니다."""
    import matplotlib.pyplot as plt

    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.3)
    if label:
        plt.legend()
    plt.show()

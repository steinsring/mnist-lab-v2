# -*- coding: utf-8 -*-
"""
학습 루프 스모크 테스트. (Day 3)

MNIST 다운로드 없이, 작은 인공 데이터로 "정말 학습이 되는지"만 몇 초 안에 확인합니다.
테스트가 통과하면 Forward -> Loss -> Backward -> Update 연결이 올바르다는 뜻입니다.

실행:
    pytest tests/test_training_smoke.py -v
"""

import numpy as np

from losses import SoftmaxWithLoss
from network import TwoLayerNet
from optimizers import SGD, Adam
from training import train


def make_learnable_data(n_samples=1000, seed=0):
    """
    MNIST를 흉내 낸 작은 인공 데이터.

    클래스마다 "켜지는 픽셀" 패턴을 정해 두고 노이즈를 더해 샘플을 만듭니다.
    - 입력은 MNIST와 같이 0~1 범위이고 대부분의 픽셀이 0에 가깝습니다.
    - 10개 클래스가 고르게 나타나며, 학습하면 충분히 맞출 수 있습니다.
    """
    rng = np.random.default_rng(seed)

    patterns = np.zeros((10, 784))
    for class_id in range(10):
        on_pixels = rng.choice(784, size=40, replace=False)
        patterns[class_id, on_pixels] = 1.0

    y = rng.integers(0, 10, n_samples)
    x = np.clip(patterns[y] + rng.normal(0.0, 0.2, (n_samples, 784)), 0.0, 1.0)
    return x, y


class TestTrainingSmoke:

    def test_returns_loss_history_per_epoch(self):
        """train()은 epoch별 평균 손실 리스트를 반환해야 한다."""
        x, y = make_learnable_data()
        model = TwoLayerNet(seed=0)
        history = train(model, SoftmaxWithLoss(), Adam(lr=0.001), x, y,
                        epochs=3, batch_size=128, verbose=False)
        assert isinstance(history, list)
        assert len(history) == 3
        assert all(np.isfinite(value) and value > 0 for value in history)

    def test_first_loss_is_near_log_num_classes(self):
        """학습 시작 직후의 손실은 ln(10) = 2.30 근처여야 한다."""
        x, y = make_learnable_data()
        model = TwoLayerNet(seed=0)
        history = train(model, SoftmaxWithLoss(), SGD(lr=0.0), x, y,
                        epochs=1, batch_size=128, verbose=False)
        assert 2.0 < history[0] < 2.8, (
            f"lr=0으로 학습해도 손실이 {history[0]:.3f}입니다. "
            "초기 손실은 ln(10)=2.303 근처여야 합니다. forward를 확인하세요."
        )

    def test_loss_decreases_with_adam(self):
        """Adam으로 몇 epoch만 돌려도 손실이 뚜렷하게 감소해야 한다."""
        x, y = make_learnable_data()
        model = TwoLayerNet(seed=0)
        history = train(model, SoftmaxWithLoss(), Adam(lr=0.001), x, y,
                        epochs=5, batch_size=128, verbose=False)
        assert history[-1] < history[0] * 0.7, (
            f"손실이 충분히 감소하지 않았습니다 ({history[0]:.3f} -> {history[-1]:.3f}). "
            "backward 연결이나 optimizer 업데이트를 확인하세요."
        )

    def test_loss_decreases_with_sgd(self):
        """SGD로도 손실이 감소해야 한다. in-place 갱신이 되어 있지 않으면 실패한다."""
        x, y = make_learnable_data()
        model = TwoLayerNet(seed=0)
        history = train(model, SoftmaxWithLoss(), SGD(lr=0.1), x, y,
                        epochs=5, batch_size=128, verbose=False)
        assert history[-1] < history[0] * 0.9, (
            f"손실이 감소하지 않았습니다 ({history[0]:.3f} -> {history[-1]:.3f}). "
            "SGD.update가 params를 제자리에서 갱신하는지 확인하세요."
        )

    def test_parameters_actually_change(self):
        """학습 후 모델 파라미터가 실제로 바뀌어야 한다."""
        x, y = make_learnable_data()
        model = TwoLayerNet(seed=0)
        before = {key: value.copy() for key, value in model.params.items()}
        train(model, SoftmaxWithLoss(), SGD(lr=0.1), x, y,
              epochs=1, batch_size=128, verbose=False)
        for key in before:
            assert not np.allclose(model.params[key], before[key]), (
                f"{key}가 학습 후에도 그대로입니다. optimizer 업데이트를 확인하세요."
            )

    def test_model_can_fit_training_data(self):
        """의도적으로 쉬운 인공 패턴을 학습해 전체 연결이 동작하는지 확인한다.

        이 임계값은 MNIST 성능 목표가 아니라 구현 smoke 검증에만 사용합니다.
        """
        x, y = make_learnable_data()
        model = TwoLayerNet(seed=0)
        train(model, SoftmaxWithLoss(), Adam(lr=0.001), x, y,
              epochs=5, batch_size=128, verbose=False)
        accuracy = np.mean(np.argmax(model.predict(x), axis=1) == y) * 100
        assert accuracy > 80.0, (
            f"인공 패턴 학습 정확도가 {accuracy:.1f}%입니다. "
            "Forward -> Loss -> Backward -> Update 연결을 확인하세요."
        )

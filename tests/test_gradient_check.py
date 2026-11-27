# -*- coding: utf-8 -*-
"""
Gradient check 테스트. (Day 2의 최종 관문)

Affine.backward, ReLU.backward, SoftmaxWithLoss.backward가 모두 맞아야 통과합니다.
수치 미분으로 직접 측정한 gradient와 여러분의 backward가 계산한 gradient를 비교합니다.

실행:
    pytest tests/test_gradient_check.py -v -s
"""

import numpy as np

from losses import SoftmaxWithLoss
from network import TwoLayerNet
from training import gradient_check


class TestGradientCheck:

    def test_backward_matches_numerical_gradient(self):
        """모든 파라미터에서 해석적 gradient와 수치 미분이 일치해야 한다."""
        rng = np.random.default_rng(0)
        model = TwoLayerNet(hidden_size=16, seed=0)
        criterion = SoftmaxWithLoss()
        x = rng.standard_normal((8, 784))
        y = rng.integers(0, 10, 8)

        ok, max_rel_error = gradient_check(model, criterion, x, y, num_checks=6, tol=1e-6)

        assert ok, (
            f"gradient check 실패 (최대 상대 오차 {max_rel_error:.2e}). "
            "출력에 표시된 파라미터의 backward를 다시 확인하세요. "
            "W2/b2에서 틀리면 SoftmaxWithLoss나 두 번째 Affine, "
            "W1/b1에서만 틀리면 ReLU나 첫 번째 Affine이 원인일 가능성이 높습니다. "
            "모든 파라미터에서 1e-5 수준으로만 조금 넘는다면 SoftmaxWithLoss.forward에서 "
            "log(확률 + 1e-7) 대신 np.clip으로 하한을 두었는지 확인하세요."
        )

    def test_backward_matches_with_real_scale_input(self):
        """MNIST처럼 0~1 범위의 입력에서도 gradient가 일치해야 한다."""
        rng = np.random.default_rng(1)
        model = TwoLayerNet(hidden_size=32, seed=1)
        criterion = SoftmaxWithLoss()
        x = rng.random((16, 784))
        y = rng.integers(0, 10, 16)

        ok, max_rel_error = gradient_check(model, criterion, x, y, num_checks=4, tol=1e-6)

        assert ok, f"gradient check 실패 (최대 상대 오차 {max_rel_error:.2e})"

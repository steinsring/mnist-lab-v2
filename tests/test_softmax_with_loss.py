# -*- coding: utf-8 -*-
"""
SoftmaxWithLoss 테스트. (Day 1: forward / Day 2: backward)

실행:
    pytest tests/test_softmax_with_loss.py -v
"""

import numpy as np

from losses import SoftmaxWithLoss


class TestSoftmaxWithLossForward:
    """Day 1에서 통과시킵니다."""

    def test_probabilities_sum_to_one(self):
        """각 샘플의 클래스 확률 합은 1이어야 한다."""
        criterion = SoftmaxWithLoss()
        logits = np.random.randn(4, 10)
        criterion.forward(logits, np.array([0, 1, 2, 3]))
        assert criterion.y_prob is not None, "확률을 self.y_prob에 저장해야 합니다."
        np.testing.assert_allclose(criterion.y_prob.sum(axis=1), np.ones(4), rtol=1e-10)

    def test_probabilities_in_valid_range(self):
        """확률은 0 이상 1 이하여야 한다."""
        criterion = SoftmaxWithLoss()
        criterion.forward(np.random.randn(4, 10) * 5, np.array([0, 1, 2, 3]))
        assert (criterion.y_prob >= 0).all() and (criterion.y_prob <= 1).all()

    def test_uniform_logits_give_log_num_classes(self):
        """모든 logits가 같으면 확률이 1/10이므로 손실은 ln(10) = 2.302585 가 된다.

        학습 시작 직후의 손실이 이 값 근처인지 확인하는 것이 가장 값싼 검증 방법이다.
        log(0)을 피하려고 더한 아주 작은 값(예: 1e-7)으로 생기는 오차는 허용한다.
        """
        criterion = SoftmaxWithLoss()
        loss = criterion.forward(np.zeros((8, 10)), np.random.randint(0, 10, 8))
        np.testing.assert_allclose(loss, np.log(10), rtol=1e-6)

    def test_returns_scalar(self):
        """손실은 배치 전체를 요약한 스칼라 하나여야 한다."""
        criterion = SoftmaxWithLoss()
        loss = criterion.forward(np.zeros((4, 3)), np.array([0, 1, 2, 0]))
        assert np.isscalar(loss) or np.asarray(loss).shape == ()
        np.testing.assert_allclose(loss, np.log(3), rtol=1e-6)

    def test_numerically_stable_for_large_logits(self):
        """큰 logits에서도 overflow 없이 정상 값이 나와야 한다. (max-shift 필요)"""
        criterion = SoftmaxWithLoss()
        loss = criterion.forward(np.array([[1000.0, 1001.0, 1002.0]]), np.array([2]))
        assert np.isfinite(loss), "max-shift 없이 exp를 계산하면 overflow가 발생합니다."
        np.testing.assert_allclose(loss, 0.40760596, rtol=1e-6)

    def test_loss_decreases_when_prediction_improves(self):
        """정답 클래스의 logit이 커지면 손실은 작아져야 한다."""
        criterion = SoftmaxWithLoss()
        y = np.array([0])
        bad = criterion.forward(np.array([[0.0, 0.0, 0.0]]), y)
        good = criterion.forward(np.array([[5.0, 0.0, 0.0]]), y)
        assert good < bad


class TestSoftmaxWithLossBackward:
    """Day 2에서 통과시킵니다."""

    def test_backward_value(self):
        """dx = (확률 - 정답 one-hot) / batch_size 여야 한다."""
        criterion = SoftmaxWithLoss()
        criterion.forward(np.zeros((2, 3)), np.array([0, 1]))
        dx = criterion.backward()
        third = 1.0 / 3.0
        expected = np.array([[third - 1.0, third, third],
                             [third, third - 1.0, third]]) / 2.0
        np.testing.assert_allclose(dx, expected, rtol=1e-10)

    def test_backward_divides_by_batch_size(self):
        """batch가 커지면 샘플 하나가 gradient에 기여하는 크기는 작아져야 한다."""
        criterion = SoftmaxWithLoss()
        criterion.forward(np.zeros((1, 3)), np.array([0]))
        dx_small = criterion.backward()
        criterion.forward(np.zeros((4, 3)), np.zeros(4, dtype=int))
        dx_large = criterion.backward()
        np.testing.assert_allclose(dx_large[0], dx_small[0] / 4.0, rtol=1e-10)

    def test_backward_sums_to_zero(self):
        """확률의 합과 one-hot의 합이 같으므로 dx의 전체 합은 0이어야 한다."""
        criterion = SoftmaxWithLoss()
        criterion.forward(np.random.randn(6, 10), np.random.randint(0, 10, 6))
        assert abs(criterion.backward().sum()) < 1e-10

    def test_backward_shape(self):
        """dx는 logits와 같은 shape여야 한다."""
        criterion = SoftmaxWithLoss()
        logits = np.random.randn(4, 10)
        criterion.forward(logits, np.random.randint(0, 10, 4))
        assert criterion.backward().shape == logits.shape

    def test_backward_does_not_modify_stored_probabilities(self):
        """y_prob를 직접 수정하면 안 된다. 복사한 배열을 수정해야 한다."""
        criterion = SoftmaxWithLoss()
        criterion.forward(np.zeros((2, 3)), np.array([0, 1]))
        before = criterion.y_prob.copy()
        criterion.backward()
        np.testing.assert_allclose(criterion.y_prob, before,
                                   err_msg="self.y_prob가 변경되었습니다. copy()를 사용하세요.")

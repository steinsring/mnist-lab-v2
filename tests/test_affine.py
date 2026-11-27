# -*- coding: utf-8 -*-
"""
Affine layer 테스트. (Day 1: forward / Day 2: backward)

실행:
    pytest tests/test_affine.py -v
    pytest tests/test_affine.py -v -k "backward"
"""

import numpy as np

from layers import Affine


def make_layer():
    """모든 테스트가 공유하는 작은 예제. 손으로 계산할 수 있는 크기입니다."""
    W = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    b = np.array([1.0, 0.0, -1.0])
    x = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    return Affine(W, b), W, b, x


class TestAffineForward:
    """Day 1에서 통과시킵니다."""

    def test_forward_value(self):
        """x @ W + b 의 계산 결과가 손계산 값과 일치해야 한다."""
        layer, _, _, x = make_layer()
        out = layer.forward(x)
        expected = np.array([[10.0, 12.0, 14.0],
                             [20.0, 26.0, 32.0]])
        np.testing.assert_allclose(out, expected)

    def test_forward_shape(self):
        """(batch, input_dim) @ (input_dim, output_dim) -> (batch, output_dim)."""
        W = np.random.randn(784, 128)
        b = np.zeros(128)
        out = Affine(W, b).forward(np.random.randn(4, 784))
        assert out.shape == (4, 128)

    def test_forward_bias_is_added_to_every_row(self):
        """bias는 broadcasting으로 배치의 모든 샘플에 더해져야 한다."""
        W = np.zeros((2, 3))
        b = np.array([1.0, 2.0, 3.0])
        out = Affine(W, b).forward(np.random.randn(5, 2))
        np.testing.assert_allclose(out, np.tile(b, (5, 1)))

    def test_forward_saves_input_for_backward(self):
        """forward가 입력을 저장하지 않으면 backward에서 dW를 만들 수 없다."""
        layer, _, _, x = make_layer()
        layer.forward(x)
        assert layer.x is not None, "forward에서 self.x에 입력을 저장해야 합니다."
        np.testing.assert_allclose(layer.x, x)


class TestAffineBackward:
    """Day 2에서 통과시킵니다. shape뿐 아니라 값까지 검사합니다."""

    def test_backward_dW_value(self):
        """dW는 입력과 dout으로 만들어지며 손계산 값과 일치해야 한다."""
        layer, _, _, x = make_layer()
        layer.forward(x)
        dout = np.array([[1.0, 0.0, -1.0],
                         [2.0, 1.0, 0.0]])
        layer.backward(dout)
        expected_dW = np.array([[7.0, 3.0, -1.0],
                                [10.0, 4.0, -2.0]])
        np.testing.assert_allclose(layer.dW, expected_dW)

    def test_backward_db_value(self):
        """db는 batch 방향으로 합산한 값이어야 한다."""
        layer, _, _, x = make_layer()
        layer.forward(x)
        dout = np.array([[1.0, 0.0, -1.0],
                         [2.0, 1.0, 0.0]])
        layer.backward(dout)
        np.testing.assert_allclose(layer.db, np.array([3.0, 1.0, -1.0]))

    def test_backward_dx_value(self):
        """dx는 이전 층으로 전달할 gradient이며 손계산 값과 일치해야 한다."""
        layer, _, _, x = make_layer()
        layer.forward(x)
        dout = np.array([[1.0, 0.0, -1.0],
                         [2.0, 1.0, 0.0]])
        dx = layer.backward(dout)
        expected_dx = np.array([[-2.0, -2.0],
                                [4.0, 13.0]])
        np.testing.assert_allclose(dx, expected_dx)

    def test_backward_shapes(self):
        """dW, db, dx의 shape는 각각 W, b, x와 같아야 한다."""
        W = np.random.randn(784, 128)
        b = np.zeros(128)
        layer = Affine(W, b)
        x = np.random.randn(4, 784)
        layer.forward(x)
        dx = layer.backward(np.random.randn(4, 128))
        assert layer.dW.shape == W.shape
        assert layer.db.shape == b.shape
        assert dx.shape == x.shape

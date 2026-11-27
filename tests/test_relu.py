# -*- coding: utf-8 -*-
"""
ReLU 테스트. (Day 1: forward / Day 2: backward)

실행:
    pytest tests/test_relu.py -v
"""

import numpy as np

from activations import ReLU


class TestReLUForward:
    """Day 1에서 통과시킵니다."""

    def test_forward_keeps_positive_and_zeros_the_rest(self):
        """양수는 그대로, 음수와 0은 0이 되어야 한다."""
        x = np.array([[-1.0, 2.0],
                      [0.0, -3.0]])
        out = ReLU().forward(x)
        np.testing.assert_allclose(out, np.array([[0.0, 2.0],
                                                  [0.0, 0.0]]))

    def test_forward_does_not_modify_input(self):
        """입력 배열을 직접 수정하면 안 된다. 복사한 배열을 수정해 반환한다."""
        x = np.array([[-1.0, 2.0],
                      [0.0, -3.0]])
        original = x.copy()
        ReLU().forward(x)
        np.testing.assert_allclose(x, original,
                                   err_msg="입력 x가 변경되었습니다. x.copy()를 사용하세요.")

    def test_forward_shape(self):
        """ReLU는 shape를 바꾸지 않는다."""
        x = np.random.randn(4, 128)
        assert ReLU().forward(x).shape == x.shape


class TestReLUBackward:
    """Day 2에서 통과시킵니다."""

    def test_backward_blocks_gradient_where_input_was_not_positive(self):
        """forward 때 x <= 0 이었던 위치는 gradient가 0, 나머지는 그대로 통과해야 한다."""
        relu = ReLU()
        x = np.array([[-1.0, 2.0],
                      [0.0, -3.0]])
        relu.forward(x)
        dout = np.array([[5.0, 7.0],
                         [9.0, 11.0]])
        dx = relu.backward(dout)
        np.testing.assert_allclose(dx, np.array([[0.0, 7.0],
                                                 [0.0, 0.0]]))

    def test_backward_shape(self):
        """dx는 dout과 같은 shape여야 한다."""
        relu = ReLU()
        x = np.random.randn(4, 128)
        relu.forward(x)
        dout = np.random.randn(4, 128)
        assert relu.backward(dout).shape == dout.shape

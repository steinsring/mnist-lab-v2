# -*- coding: utf-8 -*-
"""
SGD optimizer 테스트. (Day 3)

실행:
    pytest tests/test_sgd.py -v
"""

import numpy as np

from optimizers import SGD


class TestSGD:

    def test_update_value(self):
        """param = param - lr * grad 가 정확히 계산되어야 한다."""
        params = {"W": np.array([[1.0, 2.0],
                                 [3.0, 4.0]])}
        grads = {"W": np.ones((2, 2))}
        SGD(lr=0.1).update(params, grads)
        np.testing.assert_allclose(params["W"], np.array([[0.9, 1.9],
                                                          [2.9, 3.9]]))

    def test_update_all_keys(self):
        """params에 있는 모든 파라미터가 갱신되어야 한다."""
        params = {"W1": np.ones((2, 2)), "b1": np.ones(2), "W2": np.ones((2, 3))}
        grads = {"W1": np.ones((2, 2)), "b1": np.ones(2), "W2": np.ones((2, 3))}
        SGD(lr=0.5).update(params, grads)
        for key in params:
            np.testing.assert_allclose(params[key], np.full(params[key].shape, 0.5),
                                       err_msg=f"{key}가 갱신되지 않았습니다.")

    def test_update_is_in_place(self):
        """
        레이어는 params dict와 같은 배열 객체를 공유한다.
        새 배열을 만들어 대입하면 레이어의 가중치는 바뀌지 않아 학습이 진행되지 않는다.

        아래처럼 구현하면 이 테스트가 실패한다.
            params[key] = params[key] - self.lr * grads[key]
        """
        shared = np.array([[1.0, 2.0],
                           [3.0, 4.0]])
        params = {"W": shared}
        grads = {"W": np.ones((2, 2))}

        SGD(lr=0.1).update(params, grads)

        assert params["W"] is shared, (
            "params[key]에 새 배열을 대입했습니다. 배열 내용을 제자리에서 바꿔야 합니다. (-= 사용)"
        )
        np.testing.assert_allclose(shared, np.array([[0.9, 1.9],
                                                     [2.9, 3.9]]))

    def test_learning_rate_is_used(self):
        """lr 값이 실제로 이동 거리에 반영되어야 한다."""
        params_small = {"W": np.zeros(3)}
        params_large = {"W": np.zeros(3)}
        grads = {"W": np.ones(3)}
        SGD(lr=0.01).update(params_small, grads)
        SGD(lr=1.0).update(params_large, grads)
        np.testing.assert_allclose(params_small["W"], np.full(3, -0.01))
        np.testing.assert_allclose(params_large["W"], np.full(3, -1.0))

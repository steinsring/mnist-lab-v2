# -*- coding: utf-8 -*-
"""
파라미터 업데이트 규칙(optimizer) 모음.

[학생 구현 대상]
- SGD.update

[제공 코드 - 구현하지 않습니다]
- Adam
  학습 루프 smoke 검증에 사용할 수 있도록 제공합니다. 직접 구현하지 않습니다.
"""

import numpy as np


class SGD:
    """
    확률적 경사하강법(Stochastic Gradient Descent).

    가장 단순한 optimizer로, 각 파라미터를 gradient 반대 방향으로 lr만큼 이동시킵니다.

        param = param - lr * grad
    """

    def __init__(self, lr=0.1):
        """Args: lr: 한 번 업데이트할 때 gradient에 곱할 학습률."""
        self.lr = lr

    def update(self, params, grads):
        """
        params dict의 모든 파라미터를 갱신합니다.

        Args:
            params: {"W1": ndarray, "b1": ndarray, ...}
            grads:  params와 같은 key를 갖는 gradient dict

        반드시 지켜야 할 규칙 - 제자리(in-place) 갱신:
            레이어(Affine 등)는 params dict와 **같은 배열 객체**를 공유합니다.
            따라서 아래처럼 새 배열을 만들어 대입하면 레이어는 예전 배열을 계속 보게 되고,
            레이어 출력이 바뀌지 않아 학습 loss가 줄어들지 않습니다.

                params[key] = params[key] - self.lr * grads[key]   # 이렇게 하면 안 됩니다

            배열 내용을 직접 바꾸는 방식으로 구현하세요. (예: -= 연산자)
        """
        # TODO: params의 모든 key에 대해 gradient 반대 방향으로 제자리 갱신하세요.
        raise NotImplementedError("SGD.update를 구현하세요.")


class Adam:
    """
    Adam Optimizer. (제공 코드 - 구현하지 않습니다)

    gradient의 이동평균(m)과 제곱의 이동평균(v)을 함께 사용해 파라미터마다
    학습률을 조절합니다. 이번 과제에서는 내부 수식을 구현하지 않습니다.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        if not self.m:
            for key, param in params.items():
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)

        self.t += 1
        bias1 = 1.0 - self.beta1 ** self.t
        bias2 = 1.0 - self.beta2 ** self.t

        for key in params:
            grad = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad * grad)

            m_hat = self.m[key] / bias1
            v_hat = self.v[key] / bias2
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

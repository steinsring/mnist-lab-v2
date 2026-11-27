# -*- coding: utf-8 -*-
"""
MNIST 분류용 2층 신경망. (전체 제공 코드 - 구현하지 않습니다)

구조는 아래로 고정되어 있습니다.

    784 -> Affine -> 128 -> ReLU -> Affine -> 10

이 파일을 읽고 다음을 이해하면 충분합니다.
- params dict에 어떤 파라미터가 들어 있는지
- 레이어가 params와 같은 배열을 공유한다는 점
- forward가 확률이 아니라 logits를 반환한다는 점 (확률과 손실은 SoftmaxWithLoss가 담당)
- backward가 각 레이어의 gradient를 grads dict로 모아 준다는 점
"""

import numpy as np

from activations import ReLU
from layers import Affine, BatchNorm, Dropout


class TwoLayerNet:
    """
    입력 784 -> Affine(hidden_size) -> [BatchNorm] -> ReLU -> [Dropout] -> Affine(10)

    BatchNorm과 Dropout은 제공 코드이며 기본값은 사용하지 않음(off)입니다.

    Attributes:
        params: 학습되는 파라미터 dict. optimizer에 그대로 넘깁니다.
        grads: backward가 채우는 gradient dict. params와 같은 key를 갖습니다.
        layers: forward 순서대로 담긴 레이어 리스트.
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10,
                 use_batchnorm=False, use_dropout=False, dropout_ratio=0.2, seed=0):
        """
        Args:
            hidden_size: 은닉층 뉴런 수. 필수 과제에서는 기본값 128을 사용합니다.
            use_batchnorm: True면 첫 Affine 뒤에 BatchNorm을 넣습니다.
            use_dropout: True면 ReLU 뒤에 Dropout을 넣습니다.
            seed: 가중치 초기화 난수 seed. 같은 seed면 항상 같은 초기값이 나옵니다.
        """
        rng = np.random.default_rng(seed)
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        # He 초기화: ReLU를 쓰는 신경망에서 널리 쓰이는 초기화 방식입니다.
        # 표준정규분포에 sqrt(2 / 입력차원)을 곱해 층을 지날 때 값의 크기가 유지되도록 합니다.
        self.params = {
            "W1": rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size),
            "b1": np.zeros(hidden_size),
            "W2": rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size),
            "b2": np.zeros(output_size),
        }
        if use_batchnorm:
            self.params["gamma1"] = np.ones(hidden_size)
            self.params["beta1"] = np.zeros(hidden_size)

        # 레이어는 params dict의 배열을 그대로 넘겨받아 공유합니다.
        # 따라서 optimizer가 params를 제자리에서 갱신하면 레이어의 가중치도 함께 바뀝니다.
        self.affine1 = Affine(self.params["W1"], self.params["b1"])
        self.affine2 = Affine(self.params["W2"], self.params["b2"])

        self.layers = [self.affine1]
        self.batchnorm1 = None
        self.dropout1 = None

        if use_batchnorm:
            self.batchnorm1 = BatchNorm(self.params["gamma1"], self.params["beta1"])
            self.layers.append(self.batchnorm1)

        self.layers.append(ReLU())

        if use_dropout:
            self.dropout1 = Dropout(drop_ratio=dropout_ratio, rng=np.random.default_rng(seed))
            self.layers.append(self.dropout1)

        self.layers.append(self.affine2)

        # grads는 첫 backward 이전에도 params와 같은 key를 갖도록 0으로 초기화합니다.
        self.grads = {key: np.zeros_like(value) for key, value in self.params.items()}

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 0~1로 정규화된 이미지
            train: BatchNorm/Dropout의 동작 모드. 학습 중에는 True, 추론 시에는 False.

        Returns:
            (batch_size, 10) logits. 확률이 아니라 점수입니다.
            확률과 손실은 SoftmaxWithLoss가 계산합니다.
        """
        out = x
        for layer in self.layers:
            # BatchNorm과 Dropout만 train 인자를 받습니다. Affine과 ReLU는 받지 않습니다.
            if getattr(layer, "train_mode_layer", False):
                out = layer.forward(out, train=train)
            else:
                out = layer.forward(out)
        return out

    def backward(self, dout):
        """
        Args:
            dout: (batch_size, 10) SoftmaxWithLoss.backward()가 만든 logits에 대한 gradient

        Side effects:
            self.grads를 각 레이어의 gradient로 채웁니다.
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        self.grads["W1"] = self.affine1.dW
        self.grads["b1"] = self.affine1.db
        self.grads["W2"] = self.affine2.dW
        self.grads["b2"] = self.affine2.db
        if self.batchnorm1 is not None:
            self.grads["gamma1"] = self.batchnorm1.dgamma
            self.grads["beta1"] = self.batchnorm1.dbeta

        return dout

    def predict(self, x):
        """
        추론 모드로 logits를 계산합니다.

        softmax는 값의 순서를 바꾸지 않으므로, 어떤 숫자인지 고를 때는
        logits에 바로 argmax를 적용해도 결과가 같습니다.
        """
        return self.forward(x, train=False)

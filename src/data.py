# -*- coding: utf-8 -*-
"""
MNIST 데이터 로드 유틸리티. (제공 코드 - 구현하지 않습니다)

데이터 파일이 로컬에 없으면 공개 URL에서 내려받아 `data/`에 저장합니다.
"""

import os
import urllib.request

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def load_mnist(data_dir="data"):
    """
    MNIST 손글씨 숫자 데이터셋을 로드합니다.

    data/mnist.npz가 있으면 로컬 파일을 사용하고, 없으면 다운로드 후 저장합니다.

    Returns:
        (x_train, y_train), (x_test, y_test)
        - x: (N, 784) float32, 0~1로 정규화됨
        - y: (N,) int, 0~9 레이블
    """
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, "mnist.npz")

    if not os.path.isfile(local_path):
        # 다운로드가 중간에 끊겨도 깨진 mnist.npz가 남지 않도록 임시 파일에 받은 뒤 이름을 바꿉니다.
        partial_path = local_path + ".part"
        urllib.request.urlretrieve(MNIST_URL, partial_path)
        os.replace(partial_path, local_path)

    with np.load(local_path) as data:
        x_train = data["x_train"].astype(np.float32).reshape(-1, 784) / 255.0
        x_test = data["x_test"].astype(np.float32).reshape(-1, 784) / 255.0
        y_train = data["y_train"]
        y_test = data["y_test"]

    return (x_train, y_train), (x_test, y_test)

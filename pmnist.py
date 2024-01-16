"""
Permuted MNIST Dataset
"""
from typing import Any, Optional, Callable, List, Tuple
import random

import torchvision
from PIL import Image

mnist_image_size = 28


class PermutedMNIST(torchvision.datasets.MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        permutation: Optional[List[int]] = None,
    ):
        if permutation is None:
            permutation = list(range(mnist_image_size * mnist_image_size))
            random.shuffle(permutation)

        self.permutation = permutation

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        orig_shape = img.shape

        img = Image.fromarray(img.view(-1)[self.permutation].view(orig_shape).numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

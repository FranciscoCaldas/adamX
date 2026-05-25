from .cifar10 import get_cifar10_loaders
from .mnist import get_mnist_loaders

__all__ = ["get_cifar10_loaders", "get_mnist_loaders"]

try:
	from .ogbg import download_ogbg, get_ogbg_molpcba_loaders
except ImportError:
	download_ogbg = None
	get_ogbg_molpcba_loaders = None
else:
	__all__.append("download_ogbg")
	__all__.append("get_ogbg_molpcba_loaders")

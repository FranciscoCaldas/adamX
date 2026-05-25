from .simple_mlp import SimpleMLP
from .improved_cnn import ImprovedCNN
from .cifar_net import CifarNet

__all__ = ["SimpleMLP", "ImprovedCNN", "CifarNet"]

try:
	from .gnn import GNN
except ImportError:
	GNN = None
else:
	__all__.append("GNN")

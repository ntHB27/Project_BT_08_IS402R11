"""
Fed-LCCF Client Machine Learning Subsystem.

Provides on-device Matrix Factorization with pointwise SGD, adaptive L2
regularization, user privacy enforcement, and Cold-Start handling
(Phase 1 observed mean initialization and Phase 3 Leiden centroid anchoring).
"""

from fedlccf_prototype.client_ml.local_mf import (
    DEFAULT_LATENT_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_ITEMS,
    DEFAULT_NUM_USERS,
    DEFAULT_LAMBDA_ANCHOR,
    ADAPTIVE_LAMBDA_MAP,
    MatrixFactorization,
)
from fedlccf_prototype.client_ml.client_agent import (
    FederatedClientAgent,
)

__all__ = [
    "DEFAULT_LATENT_DIM",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_NUM_ITEMS",
    "DEFAULT_NUM_USERS",
    "DEFAULT_LAMBDA_ANCHOR",
    "ADAPTIVE_LAMBDA_MAP",
    "MatrixFactorization",
    "FederatedClientAgent",
]

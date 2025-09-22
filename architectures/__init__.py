import warnings

from .elastic_mae import ElasticMae, SimpleAMEGlimpseElasticMae, DivideFourGlimpseElasticMae, StamlikeGlimpseElasticMae, \
    StamlikeSaliencyGlimpseElasticMae, DivideFourSaliencyGlimpseElasticMae, HybridElasticMae
from .rl_glimpse import ReconstructionRlMAE, ClassificationRlMAE, SegmentationRlMAE

# Suppress timm FutureWarning about deprecated imports
warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated",
    category=FutureWarning,
    module="timm.*",
)

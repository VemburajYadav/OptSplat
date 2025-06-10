from typing import Optional

from .encoder import Encoder
from .encoder_costvolume import EncoderCostVolume, EncoderCostVolumeCfg
from .encoder_depth_transformer import EncoderMultiViewDepthTransformer, EncoderMultiViewDepthTransformerCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_costvolume import EncoderVisualizerCostVolume

ENCODERS = {
    "costvolume": (EncoderCostVolume, EncoderVisualizerCostVolume),
    "depth_transformer": (EncoderMultiViewDepthTransformer, EncoderVisualizerCostVolume),
}

def get_encoder_costvolume(cfg: EncoderCostVolumeCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer

def get_encoder_depth_transformer(cfg: EncoderMultiViewDepthTransformerCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer


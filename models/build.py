from .model_zoo import TimmRegressor


def build_model(config):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm
        
    if model_type.startswith('baevit'):
        model_name = config.MODEL.NAME
        last_feature_dim = 576
        model = TimmRegressor(model_name, last_feature_dim, False, config.DATA.IMG_SIZE, config)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

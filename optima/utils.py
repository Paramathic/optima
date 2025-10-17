def skip_layers(layers, skip_attention=True, skip_mlp=False):
    """
    Skip certain layers in the model.
    layers: list, the layers to skip
    skip_attention: bool, whether to skip attention layers
    skip_mlp: bool, whether to skip MLP layers
    """
    if skip_attention:
        keys = list(layers.keys())
        for key in keys:
            if "attn" in key:
                del layers[key]
    if skip_mlp:
        keys = list(layers.keys())
        for key in keys:
            if "mlp" in key:
                del layers[key]
    return layers

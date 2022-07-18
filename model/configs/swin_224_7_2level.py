# inherited from swin_224_7_15
# cross attn and High level norm off

class EncoderConfig:    
    encoder_name = 'swin'
    load_pretrained = True
    
    img_size = 224
    window_size = 7

    patch_size = 4
    in_chans = 3
    embed_dim = 96

    depths = [2, 2, 6]
    num_heads = [3, 6, 12]

    low_level_idx = 0
    high_level_idx = 2
    high_level_after_block = True
    low_level_after_block = True

    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.
    attn_drop_rate = 0.
    drop_path_rate = 0.1
    
    norm_layer = 'layer'
    high_level_norm = False
    low_level_norm = True
    
    ape = False
    patch_norm = True
    use_checkpoint = False


class ASPPConfig:
    aspp_name = 'swin'
    load_pretrained = False
    cross_attn = 'CBAM' # set to None to disable
    
    depth = 2
    num_heads = 3
    start_window_size = 7 ## Only the first two pyramid level
    
    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.
    attn_drop_rate = 0.
    drop_path_rate = 0.1
    
    norm_layer = 'layer'
    aspp_norm = False
    aspp_activation = 'relu' # set to None in order to deactivate
    aspp_dropout = 0.1
    
    downsample = None
    use_checkpoint = False
    

class DecoderConfig:
    decoder_name = 'swin'
    load_pretrained = True
    extended_load = False

    window_size = EncoderConfig.window_size
    
    num_classes = 9
    
    low_level_idx = EncoderConfig.low_level_idx
    high_level_idx = EncoderConfig.high_level_idx
    
    depth = 2
    last_layer_depth = 6
    num_heads = 3
    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.
    attn_drop_rate = 0.
    drop_path_rate = 0.1
    norm_layer = 'layer'
    decoder_norm = True
    
    use_checkpoint = False
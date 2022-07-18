class EncoderConfig:    
    encoder_name = 'resnet'
    load_pretrained = True
    
    img_size = 224

    low_level_idx = 0
    high_level_idx = 2

class ASPPConfig:
    aspp_name = 'swin'
    load_pretrained = False
    cross_attn = None # set to None to disable
    
    depth = 2
    num_heads = 3
    start_window_size = 2
    
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

    window_size = 7
    
    num_classes = 9
    
    low_level_idx = 0
    high_level_idx = 2
    
    depth = 2
    last_layer_depth = 6
    num_heads = 4
    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.
    attn_drop_rate = 0.
    drop_path_rate = 0.1
    norm_layer = 'layer'
    decoder_norm = True
    
    use_checkpoint = False
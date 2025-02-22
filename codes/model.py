
from modules.unet import UNetModel
from generative.networks.nets import VQVAE
from config import config

myUnet = UNetModel(
    image_size=config.image_size / config.r,
    model_channels=128,
    in_channels=8,
    out_channels=8,
    num_res_blocks=8,
    num_heads=8,
    attention_resolutions=(64, 32, 16, 8),
    num_heads_upsample=-1,
    num_head_channels=-1,
    resblock_updown=True,
    channel_mult=(1, 1, 2, 2, 4, 4),
    use_scale_shift_norm=True,
    use_new_attention_order=True
)

myVQGANModel = VQVAE(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256, 512),
    num_res_channels=512,
    num_res_layers=2,
    downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1),),
    upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    num_embeddings=1024,
    embedding_dim=4,
)

if __name__ == "__main__":
    print("Number of model parameters:", sum([p.numel() for p in myUnet.parameters()]))
    print("Number of model parameters:", sum([p.numel() for p in myVQGANModel.parameters()]))

from dataclasses import dataclass


@dataclass
class config():
    use_server = True
    test_epoch_interval = 10
    image_size = 1024
    r = 8

    # VQGAN
    vae_epoch_number = 300
    vae_batch_size = 4
    milestones_g = [200]
    milestones_d = [200]
    initial_learning_rate_g = 1e-4
    initial_learning_rate_d = 5e-4

    # lcm
    batch_size = 4
    epoch_number = 600
    initial_learning_rate = 1e-4
    milestones = [300, 400, 500]
    num_train_timesteps = 1000
    beta_start = 0.00085
    beta_end = 0.012
    offset_noise = True
    offset_noise_coefficient = 0.1
    output_feature_map = True
    clip_sample = True
    num_infer_timesteps = 50
    alpha = 3
    video = False

    # SZCH
    clip_rate = 0.025
    initial_clip_sample_range_g = 2
    initial_clip_sample_range_l = 3.5
    # JSRT
    # clip_rate = 0.025
    # initial_clip_sample_range_g = 1.7
    # initial_clip_sample_range_l = 3


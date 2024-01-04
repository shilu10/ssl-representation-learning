
config = {}

# params need for model architecture and dataloader

# model architecture names
networks = {}
networks['generator_name'] = 'ContextEncoderGenerator'
networks['discriminator_name'] = 'ContextEncoderDiscriminator'
config['networks'] = networks

# dataloader
dataloader = {}
dataloader['name'] = 'ContextPredictionDataLoader'
config['dataloader'] = dataloader

# optimizer
optimizer = {}
optimizer['generator_type'] = "Adam"
optimizer['discriminator_type'] = "Adam"
optimizer['generator_lr'] = 0.001 
optimizer['generator_lr'] = 0.001

optimizer['use_lr_scheduler'] = False

config['optimizer'] = optimizer

# criterion(loss function)
criterion = {}
criterion['adversial_type'] = "binary_crossentropy"
criterion['reconstruction_type'] = "mean_squared_error"
config['criterion'] = criterion


# pretext type specific args
model = {}
model['bottleneck_dim'] = 4098
model['img_size'] = 128
model['mask_area'] = 0.24
model['resolution'] = 0.03 
model['max_pattern_size'] = 10_000
model['r_weight'] = 0.999 # recon weight
model['overlap'] = 0
model['overlap_weight_multiplier'] = 10
model['random_masking'] = False
model['num_classes'] = 10 # just random to maintain consistency in configs


config['model'] = model

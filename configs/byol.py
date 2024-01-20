
config = {}

# params need for model architecture and dataloader

model = {}
model['img_size'] = 96 
model['algorithm_type'] = "BYOL"
config['model'] = model 

# model architecture names
networks = {}
networks['encoder_type'] = 'ResNet18'
networks['projectionhead_1_type'] = 'ProjectionHead'
networks['projectionhead_2_type'] = 'ProjectionHead'
config['networks'] = networks 


# dataloader
dataloader = {}
dataloader['type'] = 'Common'
dataloader['augmentations_type'] = 'Byol'
config['dataloader'] = dataloader

# optimizer
optimizer = {}
optimizer['type'] = "Adam"
optimizer['use_lr_scheduler'] = False

config['optimizer'] = optimizer

# criterion(loss function)
criterion = {}
criterion['type'] = "BYOL"
config['criterion'] = criterion


# pretext type specific args
augmentations = {}
augmentations['scales'] = (0.08, 1.0)
augmentations['ratio'] = (0.75, 1.3333333333333333)
augmentations['brightness'] = 0.8
augmentations['contrast'] = 0.8 
augmentations['saturation'] = 0.8
augmentations['hue'] = 0.2
augmentations['kernel_size'] = int(96 * 0.1)

config['augmentations'] = augmentations

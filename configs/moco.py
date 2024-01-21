
config = {}

# params need for model architecture and dataloader

model = {}
model['img_size'] = 96 
model['algorithm_type'] = "MOCO"
model['m'] = 0.999
model['version'] = "v1"
model['temp'] = 0.07
model['queue_len'] = 65536
model['feature_dims'] = 128  # consider a num_classes
config['model'] = model 

# model architecture names
networks = {}
networks['encoder_type'] = 'ResNet18'
networks['projectionhead_type'] = 'ProjectionHead'
config['networks'] = networks 


# dataloader
dataloader = {}
dataloader['type'] = 'Common'
dataloader['augmentations_type'] = 'MOCOV1'
config['dataloader'] = dataloader

# optimizer
optimizer = {}
optimizer['type'] = "Adam"
optimizer['use_lr_scheduler'] = False

config['optimizer'] = optimizer

# criterion(loss function)
criterion = {}
criterion['type'] = "InfoNCE"
config['criterion'] = criterion


# pretext type specific args
augmentations = {}
augmentations['scales'] = (0.2, 1.0)
augmentations['ratio'] = (0.75, 1.3333333333333333)
augmentations['brightness'] = 0.4
augmentations['contrast'] = 0.4
augmentations['saturation'] = 0.4
augmentations['hue'] = 0.4
augmentations['color_jitter_prob'] = 1.0  # random apply prob, 1.0=apply color jitter to all images
augmentations['grayscale_prob'] = 0.2 

config['augmentations'] = augmentations

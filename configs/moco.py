
config = {}

# params need for model architecture and dataloader

model = {}
model['img_size'] = 96 
model['algorithm_type'] = "MoCo"
model['m'] = 0.999
model['version'] = "v1"
model['temp'] = 0.07
model['queue_len'] = 65536
model['hidden_dims'] = 128
model['projection_dims'] = 128  # num_classes
config['model'] = model 

# model architecture names
networks = {}
networks['encoder_type'] = 'ResNet50'
config['networks'] = networks 

# dataloader
dataloader = {}
dataloader['type'] = 'Common'
dataloader['transform_type'] = 'MoCoV1'
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
transformations = {}
transformations['scales'] = (0.2, 1.0)
transformations['ratio'] = (0.75, 1.3333333333333333)
transformations['brightness'] = 0.4
transformations['contrast'] = 0.4
transformations['saturation'] = 0.4
transformations['hue'] = 0.4
transformations['color_jitter_prob'] = 1.0  # random apply prob, 1.0=apply color jitter to all images
transformations['grayscale_prob'] = 0.2 

config['transformations'] = transformations

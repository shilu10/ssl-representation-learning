
config = {}

# params need for model architecture and dataloader

model = {}
model['img_size'] = 96 
model['algorithm_type'] = "BYOL"
model['projection_dims'] = 128 
model['hidden_dims'] = 256
model['m'] = 0.99
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
dataloader['transform_type'] = 'BYOL'
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
transformations = {}
transformations['scales'] = (0.08, 1.0)
transformations['ratio'] = (0.75, 1.3333333333333333)
transformations['brightness'] = 0.8
transformations['contrast'] = 0.8 
transformations['saturation'] = 0.8
transformations['hue'] = 0.2
transformations['color_jitter_prob'] = 0.8    # random apply prob
transformations['grayscale_prob'] = 0.2 		# ranodm apply prob
transformations['kernel_size'] = int(96 * 0.1)

config['transformations'] = transformations

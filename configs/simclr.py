
config = {}

# params need for model architecture and dataloader

model = {}
model['img_size'] = 96 
model['algorithm_type'] = "SimCLR"
model['hidden_dims'] = 256
model['projection_dims'] = 128
config['model'] = model 

# model architecture names
networks = {}
networks['encoder_type'] = 'ResNet18'
config['networks'] = networks 

# dataloader
dataloader = {}
dataloader['type'] = 'Common'
dataloader['augmentations_type'] = 'SimCLR'
config['dataloader'] = dataloader

# optimizer
optimizer = {}
optimizer['type'] = "Adam"
optimizer['use_lr_scheduler'] = False

config['optimizer'] = optimizer

# criterion(loss function)
criterion = {}
criterion['type'] = "NTXent"
criterion['tau'] = 1.0
config['criterion'] = criterion


# pretext type specific args
s = 1
transformations = {}
transformations['scales'] = (0.2, 1.0)
transformations['ratio'] = (0.75, 1.3333333333333333)
transformations['brightness'] = 0.8 * s
transformations['contrast'] = 0.8 * s
transformations['saturation'] = 0.8 * s
transformations['hue'] = 0.2 * s
transformations['color_jitter_prob'] = 0.8  # random apply prob, 1.0=apply color jitter to all images
transformations['grayscale_prob'] = 0.2 

config['transformations'] = transformations



config = {}

model = {}
model['patch_dim'] = 15
model['gap'] = 2
model['img_size'] = 96 
model['num_classes'] = 8

config['model'] = model

networks = {}
networks['type'] = 'AlexNetContextPrediction'
config['networks'] = networks

dataloader = {}
dataloader['type'] = 'ContextPredictionDataLoader'
config['dataloader'] = dataloader

# optimizer
optimizer = {}
optimizer['type'] = "Adam"
optimizer['lr'] = 0.001 
optimizer['use_lr_scheduler'] = False
config['optimizer'] = optimizer

# loss function
criterion = {}
criterion['type'] = 'sparse_categorical_crossentropy'
config['criterion'] = criterion



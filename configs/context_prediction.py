config = {}

model = {}
model['patch_size'] = 15
model['gap'] = 2
model['img_size'] = 96 
model['num_classes'] = 8

config['model'] = model

networks = {}
networks['type'] = 'AlexNetContextPrediction'
config['networks'] = networks

dataloader = {}
dataloader['name'] = 'ContextPredictionDataLoader'
config['dataloader'] = dataloader

# optimizer
optimizer = {}
optimizer['name'] = "Adam"
optimizer['lr'] = 0.001 
optimizer['use_lr_scheduler'] = False
config['optimizer'] = optimizer

# loss function
criterion = {}
criterion['name'] = 'sparse_categorical_crossentropy'
config['criterion'] = criterion



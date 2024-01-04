config = {}

model = {}
model['rotations'] = [0, 90, 180, 270]
model['use_all_rotation'] = False 
model['num_classes'] = 4

config['model'] = model

networks = {}
networks['type'] = 'AlexNetRotationPrediction'
config['networks'] = networks

dataloader = {}
dataloader['name'] = 'RotationPredictionDataLoader'
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



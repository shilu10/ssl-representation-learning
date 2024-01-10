config = {}


model = {}
model['grid_size'] = (3, 3)
model['num_classes'] = 10
model['permutation_path'] = 'permutation_max_10.npy'

config['model'] = model

networks = {}
networks['type'] = 'AlexNetJigSaw'
config['networks'] = networks

dataloader = {}
dataloader['type'] = 'JigSawDataLoader'
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

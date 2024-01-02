batch_size   = 128

config = {}

config['grid_size'] = (3, 3)
config['num_classes'] = 10 
config['permutation_path'] = 'permutation_max_10.npy'

networks = {}
networks['type'] = 'AlexNetContextPrediction'
config['networks'] = networks

dataloader = {}
dataloader['name'] = 'ContextPredictionDataLoader'
config['dataloader'] = dataloader

# optimizer
model_training = {}
model_training['optimizer'] = "Adam"
model_training['loss'] = 'CrossEntropyLoss'
model_training['metrics'] = ['prec1', 'prec5']
config['model_training'] = model_training


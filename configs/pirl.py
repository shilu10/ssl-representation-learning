
config = {}

# params need for model architecture and dataloader

model = {}
model['img_size'] = 96 
model['algorithm_type'] = "PIRL"
model['projection_dims'] = 128  # num_classes
model['pretext_task_type'] = "JigSaw"
config['model'] = model 

# model architecture names
networks = {}
networks['encoder_type'] = 'ResNet50'
networks['generic_type'] = "GenericTask"
networks['transformed_type'] = "JigSawTask"
config['networks'] = networks 

# dataloader
dataloader = {}
dataloader['type'] = 'PIRL'
config['dataloader'] = dataloader

# optimizer
optimizer = {}
optimizer['type'] = "Adam"
optimizer['use_lr_scheduler'] = False

config['optimizer'] = optimizer

# criterion(loss function)
criterion = {}
criterion['type'] = "NCE"
config['temp'] = 1.0
config['criterion'] = criterion

# mmory bank
memory_bank = {}
memory_bank['weight'] = 0.5
memory_bank["datapath"] = ".stl10/unlabeled_images/"

config['memory_bank'] = memory_bank

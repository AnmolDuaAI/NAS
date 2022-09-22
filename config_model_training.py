

############################ CONFIG FILE FOR MODEL SEARCHING ##############
# Lets Try this for cifar dataset
data = "./data/"

save = "./ciphar_nas/"

train_prop = 0.5

seed = 2
batch_size = 64
lr = 0.025
lr_min = 0.001
arch_lr = 3e-4
arch_weight_decay = 1e-3

momentum = 0.9
weight_decay = 3e-4
gpu_device_id = 0
epochs = 50
init_channels = 16
layers = 8
model_path = "./saved_checkpoints/"
cutout = False
cutout_length = 16
drop_path_prob = 0.3
report_freq = 10

unrolled = True



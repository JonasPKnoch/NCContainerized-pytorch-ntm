"""Copy Task NTM model."""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
import numpy as np

from ntm.aio import EncapsulatedNTM


# This is ugly but it's fine
def dataloader(num_batches, batch_size, data_path):
    file = open(data_path, "r")
    state = 0
    input_batch=[]
    output_batch=[]
    inputs = []
    outputs = []
    batch_count = 0

    file.readline()
    while True:
        line = file.readline()
        if not line:
            break
        if "INPUT" in line:
            input_batch.append(inputs)
            output_batch.append(outputs)
            inputs = []
            outputs = []
            if len(input_batch) == batch_size:
                batch_count += 1
                torch_input = torch.from_numpy(np.stack(input_batch, 1))
                torch_output = torch.from_numpy(np.stack(output_batch, 1))
                input_batch = []
                output_batch = []
                yield batch_count, torch_input.float(), torch_output.float()
                if batch_count == num_batches:
                    break
            
            state = 0
            continue

        if "OUTPUT" in line:
            state = 1
            continue

        row = np.fromstring(line, dtype=float, sep=" ")
        if state == 0:
            inputs.append(row)
        if state == 1:
            outputs.append(row)
    file.close()
    


@attrs
class NCTestCopyTaskParams(object):
    name = attrib(default="copy-task")
    controller_size = attrib(default=100, convert=int)
    controller_layers = attrib(default=1,convert=int)
    num_heads = attrib(default=1, convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=20, convert=int)
    num_batches = attrib(default=50000, convert=int) #Must be 50000
    batch_size = attrib(default=1, convert=int)
    data_path = attrib(default="./tasks/data/copy_task_train.txt", convert=str)
    test_data_path = attrib(default="./tasks/data/copy_task_test.txt", convert=str)
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)


#
# To create a network simply instantiate the `:class:CopyTaskModelTraining`,
# all the components will be wired with the default values.
# In case you'd like to change any of defaults, do the following:
#
# > params = CopyTaskParams(batch_size=4)
# > model = CopyTaskModelTraining(params=params)
#
# Then use `model.net`, `model.optimizer` and `model.criterion` to train the
# network. Call `model.train_batch` for training and `model.evaluate`
# for evaluating.
#
# You may skip this alltogether, and use `:class:CopyTaskNTM` directly.
#

@attrs
class NCTestCopyTaskModelTraining(object):
    params = attrib(default=Factory(NCTestCopyTaskParams))
    net = attrib()
    dataloader = attrib()
    test_dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()
    sequence_width = 8

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        net = EncapsulatedNTM(self.sequence_width + 1, self.sequence_width,
                              self.params.controller_size, self.params.controller_layers,
                              self.params.num_heads,
                              self.params.memory_n, self.params.memory_m)
        return net

    @dataloader.default
    def default_dataloader(self):
        return dataloader( self.params.num_batches, self.params.batch_size, self.params.data_path)
    
    @test_dataloader.default
    def default_test_dataloader(self):
        return dataloader( self.params.num_batches, self.params.batch_size, self.params.test_data_path)

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)

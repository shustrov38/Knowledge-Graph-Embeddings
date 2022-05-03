import io
import sys

from tqdm import tqdm
import time

from typing import List, Tuple, Set
import random

import numpy as np
import torch

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False);


class train_kit:
    
    SAVE_PATH = './models/saves'
    TORCH_FILE_EXT = '.pth'
    
    
    def __init__(self, path: str, train: Set[Tuple[int, int, int]], valid: Set[Tuple[int, int, int]]=set()):
        self.path = path
        
        entities = set()       
        relations = set()
     
        for l, o, r in train | valid:
            entities.add(l)
            relations.add(o)
            entities.add(r)
        
        self.C = np.array(list(sorted(entities)))
        self.R = np.array(list(sorted(relations)))
        self.D = train
        self.V = valid if valid else None
    
    def train(self, network: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch_count: int, batches_per_epoch: int, \
                  save_epochs: List[int]=[], write_log: bool=False, file: io.TextIOWrapper=sys.stdout) -> List[float]:
        assert file is not None
        
        accuracies = []
        best_loss = None
        batch_size = len(self.D) // batches_per_epoch

        D = torch.IntTensor(list(self.D))
        D = torch.utils.data.TensorDataset(D)
        
        for epoch in tqdm(list(range(epoch_count)), desc='epoch'):
            if write_log:
                print(f'epoch #{epoch + 1}', file=file)
            start = time.time()
            epoch_losses = []

            for batch_index, batch in enumerate(torch.utils.data.DataLoader(D, batch_size=batch_size, shuffle=True, drop_last=True)):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.zeros(1)

                if type(batch) == list: 
                    batch = batch[0]
                
                for normal in batch:
                    negative = self.get_negative(normal)
                    loss += network(normal, negative)

                batch_mean_loss = float(loss.item()) / len(batch)
                epoch_losses.append(batch_mean_loss)

                if write_log and (batch_index + 1) % 50 == 0:
                    print(f'\tbatch #{batch_index + 1}, mean_loss={batch_mean_loss}', file=file)

                loss.backward()
                optimizer.step()

            end = time.time()
            epoch_mean_loss = float(sum(epoch_losses)) / len(epoch_losses)

            elapsed_time = end - start
            literal = 's'
            if elapsed_time > 60.0:
                elapsed_time /= 60.0
                literal = 'm'

            if self.V:
                accuracy = self.validate(network)
                accuracies.append(accuracy)
                if write_log:
                    print(f'\taccuracy={accuracy}', file=file)
            if write_log:
                print(f'\tloss={epoch_mean_loss}, time={elapsed_time}{literal}', file=file)
            
            if epoch in save_epochs:
                torch.save(network.state_dict(), f'{self.SAVE_PATH}/{self.path}-{epoch}{self.TORCH_FILE_EXT}')
            torch.save(network.state_dict(), f'{self.SAVE_PATH}/{self.path}{self.TORCH_FILE_EXT}')

        if write_log:
            print('finish', file=file)
        
        return accuracies
    
    def __get_link(self, network: torch.nn.Module, l: int, r: int) -> int:
        min_energy = None
        o_with_min_energy = 0
        
        for o in self.R:
            normal = torch.IntTensor([l, o, r])
            negative = self.get_negative(normal)
            cur_energy = network(normal, negative).detach().clone().detach().numpy()[0]
            if min_energy is None or cur_energy < min_energy:
                min_energy = cur_energy
                o_with_min_energy = o
            
        return o_with_min_energy
    
    def validate(self, network: torch.nn.Module) -> float:
        good_count = 0
        for l, o, r in self.V:
            good_count += (o == self.__get_link(network, l, r))
            
        accuracy = good_count / len(self.V) * 100
            
        return accuracy 
    
    def get_negative(self, normal: torch.Tensor) -> torch.Tensor:
        negative = normal.tolist()
        while True:
            k = np.random.randint(0, 3)
            _negative = negative.copy()
            c = random.choice(self.C)
            r = random.choice(self.R)
            _negative[k] = (c, r)[k == 1]
            if tuple(_negative) not in self.D:
                return torch.IntTensor(_negative)
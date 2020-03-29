from collections import OrderedDict
from collections import namedtuple
from itertools import product
from termcolor import colored


class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run',params.keys())
        
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs




import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import time
import json

from itertools import product
from collections import namedtuple
from collections import OrderedDict


class RunManager():

    def __init__(self):

        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):

        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')
        # images, _ = next(iter(self.loader))
        # grid = torchvision.utils.make_grid(images)
        # self.tb.add_image('images', grid)
        # self.tb.add_graph(self.network, images)

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        ## Add the various values to tensorboard
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

       

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration

        for k,v in self.run_params._asdict().items(): 
            results[k] = v
        
        self.print_interactive(results)

    
    def print_interactive(self,dictval):
        
        for k,v in  dictval.items():

            key = colored(k,'blue', attrs=['bold'] )

            if type(v) == float:
                v = round(v,3)

            color = 'green'

            if k == 'accuracy':
                if v < 0.5:
                    color = 'red'

            value = colored( v, color, attrs=['bold'])

            print('{} : {}'.format(key,value))

        print()


            

    
    def track_loss(self,loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self.get_num_correct(preds, labels)


    @torch.no_grad()
    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    

    def save_results(self, fileName):
    
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

    
    def save_model(self,path):
        
        torch.save(self.network, path)


    

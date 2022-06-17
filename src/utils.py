import os
from datetime import datetime
from torch import nn

# logging
class Logger():
    def __init__(self, mode, log_dir, custom_name=''):
        assert mode in ['custom']
        self.mode = mode
        
        # create log file
        now = datetime.now()
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        logfname = 'log_{}_{}.txt'.format(custom_name, now.strftime("%m-%d-%Y_%H:%M:%S"))
        self.logfname = os.path.join(log_dir, logfname)
        print(self.logfname)
        
        with open(self.logfname, 'w') as fp: # create file
            pass
        
        # log intro message
        start_msg = 'beginning {} on {}.\n'.format(self.mode, now.strftime("%m/%d/%Y, %H:%M:%S"))

            
        if mode == 'custom':
            start_msg += '--------------------------\n'
            start_msg += 'custom log.\n'
        
        self.write_msg(start_msg)
        print(start_msg)
        
    def write_msg(self, msg, print_=True):
        if print_:
            print(msg)
            
        if not msg.endswith('\n'):
            msg += '\n'
            
        log_f = open(self.logfname, 'a')
        log_f.write(msg)
        log_f.close()
        
        return
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_netinput_onechannel(net, model):
    # fix nets to take one channel as input
    name = model.__name__
    if 'resnet' in name:
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif 'vgg' in name:
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    elif 'squeezenet' in name:
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))
    elif 'densenet' in name:
        net.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
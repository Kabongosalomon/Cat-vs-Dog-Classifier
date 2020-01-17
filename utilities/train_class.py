import numpy as np
# import custom_loss_print
import torch

# reporter = custom_loss_print.LossPrettifier(show_percentage=True)

class LossPrettifier(object):
    
    STYLE = {
        'green' : '\033[32m',
        'red'   : '\033[91m', 
        'bold'  : '\033[1m', 
    }
    STYLE_END = '\033[0m'
    
    def __init__(self, show_percentage=False):
        
        self.show_percentage = show_percentage
        self.color_up = 'green'
        self.color_down = 'red'
        self.loss_terms = {}
    
    def __call__(self, epoch=None, **kwargs):
        
        if epoch is not None:
            print_string = f'Epoch {epoch: 5d} '
        else:
            print_string = ''

        for key, value in kwargs.items():
            
            pre_value = self.loss_terms.get(key, value)
            
            if value > pre_value:
                indicator  = '▲'
                show_color = self.STYLE[self.color_up]
            elif value == pre_value:
                indicator  = ''
                show_color = ''
            else:
                indicator  = '▼'
                show_color = self.STYLE[self.color_down]
            
            if self.show_percentage:
                show_value = 0 if pre_value == 0 \
                             else (value - pre_value) / float(pre_value)
                key_string = f'| {key}: {show_color}{value:3.4f}({show_value:+3.4%}) {indicator}'
            else: 
                key_string = f'| {key}: {show_color}{value:.4f} {indicator}'
            
            # Trim some long outputs
            key_string_part = key_string[:32]
            print_string += key_string_part+f'{self.STYLE_END}\t'
            
            self.loss_terms[key] = value
            
        print(print_string)
        
reporter = LossPrettifier(show_percentage=True)

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss)) 
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss +=((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
            
        # print training/validation statistics 
        
        reporter(epoch=epoch, LossA = train_loss, LossB = valid_loss)
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model
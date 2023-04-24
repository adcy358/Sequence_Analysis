import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np 

def save_checkpoint(model_state, ckpt_filename): 
    print("-> Saving checkpoint") 
    torch.save(model_state, ckpt_filename)      
    
def load_checkpoint(checkpoint, model, optimizer): 
    print("-> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict']) 
    optimizer.load_state_dict(checkpoint['optimizer']) 
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history'] 
    
    
def train(dataset, model, args, DEVICE='cuda', ckpt_filename='checkpoints.tar', load_model=False, save_epochs=10, lr=0.001): 
    """
        Train Loop       
    """
    
    model.train() 
    dataloader = DataLoader(dataset, batch_size=args.batch_size) 
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    
    # WARNING: everytime we set load_model=False, it overwrites the previously saved file.
    if load_model: 
        load_checkpoint(torch.load(ckpt_filename), model, optimizer) 
        
    mean_loss = []
    loss_history = []
    
    for epoch in range(args.max_epochs):   
        # save checkpoint   
        if epoch % save_epochs == 0 and epoch != 0: 
            checkpoint = {
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'epoch': epoch, 
                'loss_history': loss_history,
            }
            save_checkpoint(checkpoint, ckpt_filename) 
   
        loop = tqdm(dataloader, leave=True)
        state_h, state_c = model.init_state(args.sequence_length) 
    
        for batch, (x, y) in enumerate(loop): 
            optimizer.zero_grad() 
            y_pred, (state_h, state_c) = model(x, (state_h, state_c)) 
            loss = criterion(y_pred.transpose(1,2),y)
            mean_loss.append(loss.item())
            state_h = state_h.detach()
            state_c = state_c.detach() 
            loss.backward() 
            optimizer.step() 
            
            # update progress bar
            loop.set_postfix(loss=loss.item())
        
        avg_loss = sum(mean_loss)/len(mean_loss)
        loss_history.append(avg_loss)
        print(f"\033[34m EPOCH {epoch + 1}: \033[0m Mean loss {avg_loss:.3f}")
    
    if args.max_epochs != 0:    
        checkpoint = {
                    'state_dict': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss_history': loss_history,
                }
        save_checkpoint(checkpoint, ckpt_filename)
        
    return loss_history
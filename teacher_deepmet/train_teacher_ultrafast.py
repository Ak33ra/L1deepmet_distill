import json
import os.path as osp
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph, knn_graph
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from tqdm import tqdm
import argparse
import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate
import warnings
warnings.simplefilter('ignore')
from time import strftime, gmtime
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--data', default='/hildafs/projects/phy230010p/share/NanoAOD/data4L1/data_ttbar',
                    help="Name of the data folder")
parser.add_argument('--ckpts', default='/hildafs/projects/phy230010p/share/NanoAOD/Znunu/ckpts_znunu_Sep17_teacher_ultrafast',
                    help="Name of the ckpts folder")
parser.add_argument('--batch_size', type=int, default=2048,
                    help="Batch size for training (default: 2048)")
parser.add_argument('--max_epochs', type=int, default=5,
                    help="Maximum number of epochs (default: 5)")
parser.add_argument('--num_workers', type=int, default=8,
                    help="Number of data loading workers (default: 8)")
parser.add_argument('--use_amp', action='store_true', default=True,
                    help="Use automatic mixed precision training")
parser.add_argument('--gradient_accumulation', type=int, default=1,
                    help="Gradient accumulation steps")

scale_momentum = 128

def train(model, device, optimizer, scheduler, loss_fn, dataloader, epoch, scaler, use_amp, grad_accum_steps):
    model.train()
    loss_avg_arr = []
    loss_avg = utils.RunningAverage()
    
    # Pre-compute deltaR for efficiency
    deltaR = 0.4
    
    with tqdm(total=len(dataloader)) as t:
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device, non_blocking=True)
            
            # Prepare data
            x_cont = data.x[:,:8]
            x_cat = torch.zeros((data.x.shape[0], 3), dtype=torch.long, device=device)
            
            # Compute phi and edge_index
            phi = torch.atan2(data.x[:,1], data.x[:,0])
            etaphi = torch.stack([data.x[:,3], phi], dim=1)
            
            # Use knn_graph instead of radius_graph for faster computation
            # Or limit max_num_neighbors more aggressively
            edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, 
                                     loop=False, max_num_neighbors=128)  # Reduced from 255
            edge_index = to_undirected(edge_index)
            
            # Mixed precision training
            if use_amp:
                with autocast():
                    result = model(x_cont, x_cat, edge_index, data.batch)
                    loss = loss_fn(result, data.x, data.y, data.batch)
                    loss = loss / grad_accum_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            else:
                result = model(x_cont, x_cat, edge_index, data.batch)
                loss = loss_fn(result, data.x, data.y, data.batch)
                loss = loss / grad_accum_steps
                loss.backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            # Update metrics
            loss_avg_arr.append(loss.item() * grad_accum_steps)
            loss_avg.update(loss.item() * grad_accum_steps)
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    
    if loss_avg_arr:
        avg_loss = np.mean(loss_avg_arr)
        scheduler.step(avg_loss)
        return avg_loss
    else:
        return 0.0

def evaluate_fast(model, device, loss_fn, dataloader, metrics, deltaR=0.4):
    """Faster evaluation without gradient computation"""
    model.eval()
    loss_avg_arr = []
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device, non_blocking=True)
            x_cont = data.x[:,:8]
            x_cat = torch.zeros((data.x.shape[0], 3), dtype=torch.long, device=device)
            phi = torch.atan2(data.x[:,1], data.x[:,0])
            etaphi = torch.stack([data.x[:,3], phi], dim=1)
            edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, 
                                    loop=False, max_num_neighbors=128)
            edge_index = to_undirected(edge_index)
            
            result = model(x_cont, x_cat, edge_index, data.batch)
            loss = loss_fn(result, data.x, data.y, data.batch)
            loss_avg_arr.append(loss.item())
    
    return {'loss': np.mean(loss_avg_arr) if loss_avg_arr else 0.0}

if __name__ == '__main__':
    args = parser.parse_args()
    
    print(f"=== ULTRA FAST TRAINING CONFIG ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Num workers: {args.num_workers}")
    print(f"Mixed precision: {args.use_amp}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    
    # Set up data loaders with more workers
    dataloaders = data_loader.fetch_dataloader(data_dir=args.data, 
                                               batch_size=args.batch_size,
                                               validation_split=.25)
    
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']
    
    # Add persistent workers and pin memory for faster data transfer
    train_dl.num_workers = args.num_workers
    train_dl.pin_memory = True
    test_dl.num_workers = args.num_workers
    test_dl.pin_memory = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enable cudnn benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    norm = torch.tensor([1./scale_momentum, 1./scale_momentum, 1./scale_momentum, 
                        1., 1., 1., 1., 1.]).to(device)
    
    model = net.Net(8, 3, norm).to(device)
    
    # Use larger learning rate with bigger batch size
    base_lr = 0.001 * (args.batch_size / 64) ** 0.5  # Scale with sqrt of batch size ratio
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=base_lr * 10,
        epochs=args.max_epochs,
        steps_per_epoch=len(train_dl) // args.gradient_accumulation,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    first_epoch = 0
    max_epochs = args.max_epochs
    best_validation_loss = 10e7
    deltaR = 0.4

    loss_fn = net.loss_fn_response_tune
    metrics = net.metrics

    model_dir = args.ckpts
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup mixed precision scaler
    scaler = GradScaler() if args.use_amp else None

    if args.restore_file is not None:
        restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
        ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
        first_epoch = ckpt['epoch']
        print('Restarting training from epoch', first_epoch)
        with open(osp.join(model_dir, 'metrics_val_best.json')) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)['loss']
    
    # Create log file
    if first_epoch == 0:
        loss_log = open(model_dir+'/loss.log', 'w')
        loss_log.write('# Ultra fast training log ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
        loss_log.write('epoch, loss, val_loss\n')
        loss_log.flush()
    else:
        loss_log = open(model_dir+'/loss.log', 'a')

    # Time estimates
    batches_per_epoch = len(train_dl) // args.gradient_accumulation
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Estimated time per epoch: {batches_per_epoch * 0.5 / 60:.1f} minutes")
    print(f"Total estimated time: {batches_per_epoch * 0.5 * args.max_epochs / 60:.1f} minutes")

    for epoch in range(first_epoch+1, max_epochs+1):
        print(f'\n=== Epoch {epoch}/{max_epochs} === Time: {strftime("%Y-%m-%d %H:%M:%S", gmtime())}')
        print(f'Best loss so far: {best_validation_loss:.2f}')
        
        # Training
        train_loss = train(model, device, optimizer, scheduler, loss_fn, 
                          train_dl, epoch, scaler, args.use_amp, args.gradient_accumulation)
        print(f'Training loss: {train_loss:.2f}')
        
        # Quick evaluation every epoch
        test_metrics = evaluate_fast(model, device, loss_fn, test_dl, metrics, deltaR)
        val_loss = test_metrics['loss']
        print(f'Validation loss: {val_loss:.2f}')
        
        loss_log.write(f'{epoch}, {train_loss:.4f}, {val_loss:.4f}\n')
        loss_log.flush()

        val_metrics = {'epoch': epoch, 'loss': val_loss}
        is_best = val_loss < best_validation_loss
        
        # Save checkpoint
        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(),
            'scaler_dict': scaler.state_dict() if scaler else None
        }, is_best=is_best, checkpoint=model_dir)

        if is_best:
            best_validation_loss = val_loss
            with open(osp.join(model_dir, 'metrics_val_best.json'), 'w') as f:
                json.dump(val_metrics, f, indent=4)

        with open(osp.join(model_dir, 'metrics_val_last.json'), 'w') as f:
            json.dump(val_metrics, f, indent=4)
    
    loss_log.close()
    print("\nULTRA FAST TRAINING COMPLETED!")
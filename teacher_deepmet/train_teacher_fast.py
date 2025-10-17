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

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--data', default='/hildafs/projects/phy230010p/share/NanoAOD/data4L1/data_ttbar',
                    help="Name of the data folder")
parser.add_argument('--ckpts', default='/hildafs/projects/phy230010p/share/NanoAOD/Znunu/ckpts_znunu_Sep17_teacher_fast',
                    help="Name of the ckpts folder")
parser.add_argument('--batch_size', type=int, default=512,
                    help="Batch size for training (default: 512)")
parser.add_argument('--max_epochs', type=int, default=10,
                    help="Maximum number of epochs (default: 10)")
parser.add_argument('--num_workers', type=int, default=4,
                    help="Number of data loading workers (default: 4)")

scale_momentum = 128

def train(model, device, optimizer, scheduler, loss_fn, dataloader, epoch):
    model.train()
    loss_avg_arr = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:
        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            sample_weight = None
            x_cont = data.x[:,:8]
            x_cat = torch.zeros((data.x.shape[0], 3), dtype=torch.long, device=data.x.device)
            phi = torch.atan2(data.x[:,1], data.x[:,0])
            etaphi = torch.cat([data.x[:,3][:,None], phi[:,None]], dim=1)        
            edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, loop=False, max_num_neighbors=255)
            edge_index = to_undirected(edge_index)
            result = model(x_cont, x_cat, edge_index, data.batch)
            loss = loss_fn(result, data.x, data.y, data.batch)
            loss.backward()
            optimizer.step()
            loss_avg_arr.append(loss.item())
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    if loss_avg_arr:
        avg_loss = np.mean(loss_avg_arr)
        scheduler.step(avg_loss)
        return avg_loss
    else:
        return 0.0

if __name__ == '__main__':
    args = parser.parse_args()
    
    print(f"Training with batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Num workers: {args.num_workers}")

    dataloaders = data_loader.fetch_dataloader(data_dir=args.data, 
                                               batch_size=args.batch_size,
                                               validation_split=.25)
    
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    norm = torch.tensor([1./scale_momentum, 1./scale_momentum, 1./scale_momentum, 1., 1., 1., 1., 1.]).to(device)   
    
    model = net.Net(8, 3, norm).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, cycle_momentum=False)
    first_epoch = 0
    max_epochs = args.max_epochs
    best_validation_loss = 10e7
    deltaR = 0.4
    deltaR_dz = 0.3

    loss_fn = net.loss_fn_response_tune
    metrics = net.metrics

    model_dir = args.ckpts
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    if args.restore_file is not None:
        restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
        ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
        first_epoch = ckpt['epoch']
        print('Restarting training from epoch',first_epoch)
        with open(osp.join(model_dir, 'metrics_val_best.json')) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)['loss']
    
    if first_epoch == 0:
        loss_log = open(model_dir+'/loss.log', 'w')
        loss_log.write('# loss log for training starting in '+strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
        loss_log.write('epoch, loss, val_loss\n')
        loss_log.flush()
    else:
        loss_log = open(model_dir+'/loss.log', 'a')

    print(f"Estimated time per epoch: {(746250/args.batch_size) * 1.0 / 3600:.2f} hours")
    print(f"Total estimated time: {(746250/args.batch_size) * 1.0 * args.max_epochs / 3600:.2f} hours")

    for epoch in range(first_epoch+1, max_epochs+1):

        print(f'Epoch: {epoch} Time: {strftime("%Y-%m-%d %H:%M:%S", gmtime())} Best loss: {best_validation_loss}')
        if '_last_lr' in scheduler.state_dict():
            print('Learning rate:', scheduler.state_dict()['_last_lr'][0])

        # compute number of batches in one epoch (one full pass over the training set)
        train_loss = train(model, device, optimizer, scheduler, loss_fn, train_dl, epoch)

        # Evaluate for one epoch on validation set
        test_metrics = evaluate(model, device, loss_fn, test_dl, metrics, deltaR)
        val_loss = test_metrics['loss']
        
        loss_log.write(f'{epoch}, {train_loss:.4f}, {val_loss:.4f}\n')
        loss_log.flush()

        val_metrics = {'epoch': epoch, 'loss': val_loss}
        is_best = val_loss < best_validation_loss
        
        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                               'scheduler_dict': scheduler.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, save to json file in the model directory
        if is_best:
            best_validation_loss = val_loss
            best_json_path = osp.join(model_dir, 'metrics_val_best.json')
            with open(best_json_path, 'w') as f:
                json.dump(val_metrics, f, indent=4)

        # Save latest val metrics in a json file in the model directory
        last_json_path = osp.join(model_dir, 'metrics_val_last.json')
        with open(last_json_path, 'w') as f:
            json.dump(val_metrics, f, indent=4)
    
    loss_log.close()
    print("Training completed!")
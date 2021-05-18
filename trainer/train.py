import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim import lr_scheduler

import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import os

import hypertune
from google.cloud import storage

from trainer.UNet import UNet
from trainer.CNN1 import CNN1

import trainer.utils as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name',
        type=str,
        default='test_model.pth',
        help='model name')
    parser.add_argument(
        '--lr-scheduler',
        type=str,
        default='OneCycleLR',
        help='learning rate scheduler')
    parser.add_argument(
        '--net',
        type=str,
        required=True,
        help='network')
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args

def save_model(args, name=None):
    """Saves the model to Google Cloud Storage
    Args:
      args: contains name for saved model.
    """
    scheme = 'gs://'
    bucket_name = args.job_dir[len(scheme):].split('/')[0]

    prefix = '{}{}/'.format(scheme, bucket_name)
    bucket_path = args.job_dir[len(prefix):].rstrip('/')
    
    if bucket_path and name is None:
        model_path = '{}/{}'.format(bucket_path, args.model_name)
    else:
        model_path = '{}/{}'.format(bucket_path, name)

    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(name)

# make data loaders
# make optimizer
# contain training loop
# declare hyper params
def fit(args):
    train_set, val_set = utils.load_FMADatasets('training', 'validation')
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    model = None
    if args.net == 'UNet':
        model = UNet()
    elif args.net == 'CNN1':
        model = CNN1()
        
    model = model.to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    scheduler = None
    if args.lr_scheduler == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(
            optim, 
            max_lr=args.learning_rate * 10, 
            epochs=args.num_epochs,
            steps_per_epoch=len(train_loader)
        )
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optim, 'min')
        
    loss_function = torch.nn.CrossEntropyLoss()
    
    # Keep track of metrics
    total_train_loss, total_val_loss = [], []
    total_train_acc, total_val_acc = [], []
    
    # Keep track of saved model names
    model_names = []
    
    print('         Train Loss    Val Loss    Train Acc     Val Acc')
    for epoch in range(args.num_epochs):
        epoch_train_acc = 0.
        epoch_val_acc = 0.
        epoch_train_loss = 0.
        epoch_val_loss = 0.
        
        model.train()
        for xs, ys in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            
            optim.zero_grad()
            probs = model(xs)
            # Turns out nn.Softmax() takes ints (0-7), not 1-hot encoded vectors, oops.
            ys = torch.max(ys, 1)[1]
            loss = loss_function(probs, ys)
            
            loss.backward()
            optim.step()
            if scheduler is not None:
                if args.lr_scheduler == 'OneCycleLR':
                    scheduler.step()
            
            # Track metrics
            epoch_train_loss += loss.item()
            preds = probs.cpu().argmax(1)
            epoch_train_acc += (preds == ys.cpu()).sum().item() 
            
        epoch_train_loss /= len(train_loader)
        epoch_train_acc /= len(train_loader.dataset)
        
        model.eval()
        with torch.no_grad():
            for xs, ys in val_loader:
                xs, ys = xs.to(device), ys.to(device)
                probs = model(xs)
                ys = torch.max(ys, 1)[1]
                loss = loss_function(probs, ys)
                
                # Track metrics
                epoch_val_loss += loss.item()
                preds = probs.cpu().argmax(1)
                epoch_val_acc += (preds == ys.cpu()).sum().item()
                       
        epoch_val_loss /= len(val_loader)
        epoch_val_acc /= len(val_loader.dataset)
        
        if scheduler is not None and args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler.step(epoch_val_loss)
        
        total_train_loss.append(epoch_train_loss)
        total_val_loss.append(epoch_val_loss)
        total_train_acc.append(epoch_train_acc)
        total_val_acc.append(epoch_val_acc)

        print(f'{epoch:5d}{epoch_train_loss:14.4f}{epoch_val_loss:12.4f}{epoch_train_acc:12.4f}{epoch_val_acc:12.4f}')
        
        if epoch_val_acc > 0.48 and epoch_train_acc > 0.45:
            model_name = f'model_intermediate{epoch_val_acc:0.4f}_B{args.batch_size}_LR{args.learning_rate}.pth'
            torch.save(model.state_dict(), model_name)
            model_names.append(model_name)
        
        # Use Hypertune to report metrics for optimization
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
              hyperparameter_metric_tag='epoch_val_acc',
              metric_value=epoch_val_acc,
              global_step=epoch
        )
    # Save Model
    model_name = f'model_B{args.batch_size}_LR{args.learning_rate}.pth'
    torch.save(model.state_dict(), model_name)
    model_names.append(model_name)
    
    # Save diagrams
    image_name = f'metrics_B{args.batch_size}_LR{args.learning_rate}.png'
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(total_train_loss, label='train')
    ax1.plot(total_val_loss, label='val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss per Epoch')
    ax1.legend()
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(total_train_acc, label='train')
    ax2.plot(total_val_acc, label='val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy per Epoch')
    ax2.legend()
    
    plt.savefig(image_name)
    
    if args.job_dir:
        for model_name in model_names:
            save_model(args, name=model_name)
        save_model(args, name=image_name)

if __name__=='__main__':
    args = get_args()
    fit(args)
from math import ceil
import torch
from torch.linalg import matrix_rank
from tqdm import tqdm
import wandb
import numpy as np

from data_alg import get_data
from model import GPTBase

def main(args: dict):
    wandb.init(project="grokking", config=args)
    config = wandb.config
    device = torch.device(config.device)

    # Define time scales
    wandb.define_metric("step")
    wandb.define_metric("epoch")

    # Define metrics
    wandb.define_metric("train/accuracy", step_metric='step')
    wandb.define_metric("train/loss", step_metric='step')
    wandb.define_metric("val/accuracy", step_metric='epoch')
    wandb.define_metric("val/loss", step_metric='epoch')
    for layer_idx in range(config.n_layer):
        wandb.define_metric(f"rank/layer-{layer_idx}", step_metric='epoch')
    # wandb.define_metric("val/CKA", step_metric='epoch')

    train_loader, val_loader = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size
        )
    model = GPTBase(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor = 0.1, total_iters=9
    )

    num_epochs = ceil(config.num_steps / len(train_loader))

    for epoch in tqdm(range(num_epochs)):
        train(model, train_loader, optimizer, scheduler, device, config.num_steps)
        evaluate(model, val_loader, device, epoch)

def train(model, train_loader, optimizer, scheduler, device, num_steps):
    # Set model to training mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    # Loop over each batch from the training set
    for batch in train_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Zero gradient buffers
        optimizer.zero_grad()
        
        # Forward pass
        output = model(inputs, get_logits=True)
        logits = output['logits'][:, -1, :]
        loss = criterion(logits, labels)
        acc = (torch.argmax(logits, dim=1) == labels).sum() / len(labels)
        
        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        scheduler.step()

        metrics = {
            "train/accuracy": acc,
            "train/loss": loss,
            "step": wandb.run.step
        }
        wandb.log(metrics)

        # Finish training at maximum gradient updates
        if wandb.run.step == num_steps:
            return

def evaluate(model, val_loader, device, epoch):
    # Set model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    running_loss = 0.

    # Loop over each batch from the validation set
    for batch in val_loader:
        
        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch
        
        # Forward pass
        rank_dict = {idx:[] for idx in range(model.config.n_layer)}
        with torch.no_grad():
            output = model(inputs, get_logits=True, return_layer_rep=True)
            logits = output['logits'][:, -1, :]
            loss = criterion(logits, labels)
            correct += (torch.argmax(logits, dim=1) == labels).sum()
            running_loss += loss.item() * len(labels)
            # get feature rank
            layer_reps = output['layer_reps']
            cur_rank_dict = {layer_idx: matrix_rank(torch.cov(rep.flatten(start_dim=0, end_dim=1).T)).item()
                    for layer_idx, rep in enumerate(layer_reps)}
            for k in rank_dict.keys():
                rank_dict[k].append(cur_rank_dict[k])
            
    
    acc = correct / len(val_loader.dataset)
    loss = loss / len(val_loader.dataset)

    metrics = {
        "val/accuracy": acc,
        "val/loss": loss,
        "epoch": epoch
    }
    for layer_idx in range(model.config.n_layer):
        metrics[f"rank/layer-{layer_idx}"] = np.mean(rank_dict[layer_idx])
    wandb.log(metrics, commit=False)

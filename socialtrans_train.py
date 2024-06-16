import importlib
import torch
import numpy as np
from social_trans import SocialTrans
from data import Dataloader
from utils import seed, get_rng_state, ADE_FDE
import torch.nn as nn
import torch.optim as optim
import random
import argparse

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='+', default=[])
parser.add_argument("--test", nargs='+', default=[])
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)

if __name__ == "__main__":
    settings = parser.parse_args()
    print('settings.config', settings.config)

    # Load configuration
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Device setup
    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    settings.device = torch.device(settings.device)

    # Seed setup
    seed(settings.seed)
    init_rng_state = get_rng_state(settings.device)
    rng_state = init_rng_state

    # Prepare datasets
    kwargs = dict(
        batch_first=False,
        frameskip=settings.frameskip,
        ob_horizon=config.OB_HORIZON,
        pred_horizon=config.PRED_HORIZON,
        device=settings.device,
        seed=settings.seed
    )

    train_data = None
    if settings.train:
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.train))]
        else:
            inclusive = None
        train_dataset = Dataloader(
            settings.train, **kwargs, inclusive_groups=inclusive,
            flip=True, rotate=True, scale=True,
            batch_size=config.BATCH_SIZE, shuffle=True, batches_per_epoch=config.EPOCH_BATCHES
        )
        train_data = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate_fn,
            batch_sampler=train_dataset.batch_sampler
        )
        batches = train_dataset.batches_per_epoch

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model, loss function, and optimizer setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ztrans = SocialTrans(x_in=4, neighbor_in=3, ob_T=config.PRED_HORIZON, pred_T=8).to(device)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(ztrans.parameters(), lr=0.0001)

    # Training parameters
    num_epochs = 800
    ztrans.train()
    ade = 100000
    fde = 100000
    lossss = 100000

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_ade = 0.0
        total_fde = 0.0
        num = 0

        for x, y, nei in train_data:
            num += 1
            optimizer.zero_grad()
            y = y.to(device)
            pred = ztrans(x.to(device), nei[:config.PRED_HORIZON].to(device))

            if len(pred.size()) == 4:
                loss = torch.sqrt(loss_fn(pred, y.unsqueeze(0).repeat(config.PRED_SAMPLES, 1, 1, 1)))
                losstrans = ztrans.loss(pred, y.unsqueeze(0).repeat(config.PRED_SAMPLES, 1, 1, 1))
            else:
                loss = torch.sqrt(loss_fn(pred, y))
                losstrans = ztrans.loss(pred, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ade += losstrans['ade'].item()
            total_fde += losstrans['fde'].item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        avg_loss = total_loss / len(train_data)
        avg_ade = total_ade / len(train_data)
        avg_fde = total_fde / len(train_data)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss}, Average ADE: {avg_ade}, Average FDE: {avg_fde}')
        print('---------------------------------------------------------------------')

        if avg_loss < lossss:
            ade = avg_ade
            fde = avg_fde
            lossss = avg_loss
            print('save:------', ade, fde, lossss)

            state_best = {
                'model': ztrans.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ade': ade,
                'fde': fde,
                'loss': lossss
            }

        state = {
            'model': ztrans.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ade': ade,
            'fde': fde,
            'loss': lossss
        }

    torch.save(state, 'trans.pth')
    torch.save(state_best, 'trans_best.pth')

    print("Training finished and model saved!")

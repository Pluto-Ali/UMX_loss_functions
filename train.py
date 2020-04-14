import argparse
import model
import data
import torch
import time
from pathlib import Path
import tqdm
import json
import utils
# import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
from torch.utils.data.sampler import SubsetRandomSampler
import torchaudio


tqdm.monitor_interval = 0

def SISDR(s, s_hat):
    """Computes the Scale-Invariant SDR as in [1]_.
    References
    ----------
    .. [1] Le Roux, Jonathan, et al. "SDR–half-baked or well done?." ICASSP 2019-2019 IEEE International Conference on
    Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019.

    Parameters:
        s: list of targets of any shape
        s_hat: list of corresponding estimates of any shape
    """
    s = torch.stack(s).view(-1)
    EPS = torch.finfo(s.dtype).eps
    s_hat = torch.stack(s_hat).view(-1)
    a = (torch.dot(s_hat, s) * s) / ((s ** 2).sum() + EPS)
    b = a - s_hat
    return -10*torch.log10(((a*a).sum()) / ((b*b).sum()+EPS))

def minSNRsdsdr(s,s_hat):
    """Computes the minimum between SNR and Scale-Dependant SDR as proposed in [1]_, obtaining a loss sensitive to both
    upscalings and downscalings.

    References
    ----------
    .. [1] Le Roux, Jonathan, et al. "SDR–half-baked or well done?." ICASSP 2019-2019 IEEE International Conference on
    Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019.

    Parameters:
        s: list of targets of any shape
        s_hat: list of corresponding estimates of any shape
    """
    s = torch.stack(s).view(-1)
    EPS = torch.finfo(s.dtype).eps
    s_hat = torch.stack(s_hat).view(-1)
    snr = 10*torch.log10((s**2).sum() / ((s-s_hat)**2).sum() + EPS)
    sdsdr = snr + 20*torch.log10(torch.dot(s_hat,s)/(s**2).sum() + EPS)
    return -torch.min(snr, sdsdr)

def train(args, unmix, device, train_sampler, optimizer):
    losses = utils.AverageMeter()

    # If losses use L1 or L2, we initialize them:
    if args.loss in ['L2time', 'L2mask', 'L2freq']:
        criteria = [torch.nn.MSELoss() for t in args.targets]
    if args.loss in ['L1time', 'L1mask', 'L1freq']:
        criteria = [torch.nn.L1Loss() for t in args.targets]
    if args.loss == 'BinaryCrossEntropy':
        criteria = [torch.nn.BCEWithLogitsLoss() for t in args.targets]
    if args.loss == 'CrossEntropy':
        criteria = torch.nn.CrossEntropyLoss()
    # Set model mode as train.
    unmix.train()
    print('Chosen loss: ')
    print(args.loss)
    # Initialize progress bar
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    # Start training loop
    for x, y in pbar:
        pbar.set_description("Training batch")
        # Send inputs and targets to GPU on time domain
        x = x.to(device)
        y = [i.to(device) for i in y]
        # Clear gradients
        optimizer.zero_grad()
        # Obtain tha estimated magnitude mask and reshape it
        Y_hats = unmix(x)    # outputs list of masks: frames, batch, channels, bins; len=sources
        X = unmix.stft(x).permute(3,0,1,2,4)
        # Compute input or mixture magnitude from mixture spectrogram X
        mag = (torchaudio.functional.complex_norm(X))
        loss = 0
        # IF FREQUENCY MASKING LOSS:
        if args.loss in ['L1mask', 'L2mask', 'CrossEntropy', 'BinaryCrossEntropy']:
            # Targets are built from the STFT(y)
            Ys = [torchaudio.functional.complex_norm(unmix.stft(target).permute(3, 0, 1, 2, 4)) for target in y]
            # Energy normalization for convergence, obtaining IRM Y:
            energy = torch.sum(torch.stack(Ys), dim=0)
            Y = [Y / (energy + 1e-18) for Y in Ys]
            # For the BCE_IBM case, compute IBM setting argmax(pixels) among all sources to 1 and the rest to 0
            if args.loss in ['BinaryCrossEntropy', 'CrossEntropy']:
                Y = torch.stack(Y)
                _, Y = Y.max(0)
                if args.loss == 'BinaryCrossEntropy':   # We one-hot code the targets it for aggregating all BCEs
                    Y = torch.nn.functional.one_hot(Y,4).float().unbind(4)
            # Compute Cross-Entropy mask loss:
            if args.loss == 'CrossEntropy':
                loss = criteria(Y_hats, Y)
            # Or Compute the aggregate losses (L1, L2 or BinaryCrossEntropy)
            else:
                for Y_hat, target, criterion in zip(Y_hats, Y, criteria):
                    loss = loss + criterion(Y_hat, target)
        # IF MAPPING LOSS:
        else:
            # Apply the masks
            Y_hats = [Y_hat * mag for Y_hat in Y_hats]    # obtaining magnitude estimates
            # IF TIME DOMAIN LOSS
            if args.loss in ['L2time', 'L1time', 'SISDRtime', 'MinSNRsdsdr']:
                phase = torchaudio.functional.angle(X) # phase from mixture STFT X
                # Incorporate mixture phase into our estimates
                specs = [torch.stack([magnitude * torch.cos(phase),
                                      magnitude * torch.sin(phase)],
                                     dim=len(magnitude.shape)) for magnitude in Y_hats]
                # GPU-capable ISTFT
                y_hats = [unmix.istft(spec, x.shape[-1]) for spec in specs]
                # Compute the time-domain loss:
                if args.loss == 'SISDRtime':
                    loss = SISDR(y, y_hats)
                elif args.loss == 'MinSNRsdsdr':
                    loss = minSNRsdsdr(y, y_hats)
                else:
                    for Y_hat, target, criterion in zip(y_hats, y, criteria):
                        loss = loss + criterion(Y_hat, target)
            # IF FREQUENCY MAPPING:
            else:
                # Targets are abs(stft(y))
                Y = [torchaudio.functional.complex_norm(unmix.stft(target).permute(3, 0, 1, 2, 4)) for target in y]
                # Compute the loss
                if args.loss == 'SISDRfreq':
                    loss = SISDR(Y, Y_hats)
                else:
                    for Y_hat, target, criterion in zip(Y_hats, Y, criteria):
                        loss = loss + criterion(Y_hat, target)

        loss.backward()
        optimizer.step()
        losses.update(loss.item())#, Y_hat.size(1))
    return losses.avg


def valid(args, unmix, device, valid_sampler):
    #Sames as train() --above--, but with unmix.eval(), no backward and no_grad() mode
    losses = utils.AverageMeter()
    if args.loss in ['L2time', 'L2mask', 'L2freq']:
        criteria = [torch.nn.MSELoss() for t in args.targets]
    if args.loss in ['L1time', 'L1mask', 'L1freq']:
        criteria = [torch.nn.L1Loss() for t in args.targets]
    if args.loss == 'BinaryCrossEntropy':
        criteria = [torch.nn.BCEWithLogitsLoss() for t in args.targets]
    if args.loss == 'CrossEntropy':
        criteria = torch.nn.CrossEntropyLoss()
    unmix.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x = x.to(device)
            y = [i.to(device) for i in y]
            Y_hats = unmix(x)  # outputs list of masks: frames, batch, channels, bins; len=sources
            X = unmix.stft(x).permute(3, 0, 1, 2, 4)
            mag = torchaudio.functional.complex_norm(X)
            loss = 0
            #IF FREQUENCY MASKING:
            if args.loss in ['L1mask', 'L2mask', 'BCE_IRM', 'BCE_IBM', 'CrossEntropy']:
                Ys = [torchaudio.functional.complex_norm(unmix.stft(target).permute(3, 0, 1, 2, 4)) for target in y]
                energy = torch.sum(torch.stack(Ys), dim=0)
                Y = [Y / (energy + 1e-18) for Y in Ys]
                # For the Cross-Entropy cases, compute IBM setting argmax(pixels) among all sources to 1 in a zeros
                if args.loss in ['BinaryCrossEntropy', 'CrossEntropy']:
                    Y = torch.stack(Y)
                    _, Y = Y.max(0)
                    if args.loss == 'BinaryCrossEntropy':    # We one-hot code the targets it for aggregating all BCEs
                        Y = torch.nn.functional.one_hot(Y, 4).float().unbind(4)
                if args.loss == 'CrossEntropy':
                    loss = criteria(Y_hats, Y)
                # Compute the L1, L2 or Binary Cross-Entropy mask loss:
                else:
                    for Y_hat, target, criterion in zip(Y_hats, Y, criteria):
                        loss = loss + criterion(Y_hat, target)
            #IF NOT MASKING
            else: #Apply the masks
                Y_hats = [Y_hat * mag for Y_hat in Y_hats]
                #if time domain:
                if args.loss in ['L2time', 'L1time', 'SISDRtime', 'MinSNRsdsdr']:
                    phase = torchaudio.functional.angle(X)
                    specs = [torch.stack([magnitude * torch.cos(phase),
                                          magnitude * torch.sin(phase)],
                                         dim=len(magnitude.shape)) for magnitude in Y_hats]
                    y_hats = [unmix.istft(spec, x.shape[-1]) for spec in specs]
                    if args.loss == 'SISDRtime':
                        loss = SISDR(y, y_hats)
                    elif args.loss == 'MinSNRsdsdr':
                        loss = minSNRsdsdr(y, y_hats)
                    else:
                        for Y_hat, target, criterion in zip(y_hats, y, criteria):
                            loss = loss + criterion(Y_hat, target)
                #if frequency domain mapping
                else:
                    Y = [torchaudio.functional.complex_norm(unmix.stft(target).permute(3, 0, 1, 2, 4)) for target in y]
                    if args.loss == 'SISDRfreq':
                        loss = SISDR(Y, Y_hats)
                    else:
                        for Y_hat, target, criterion in zip(Y_hats, Y, criteria):
                            loss = loss + criterion(Y_hat, target)
            losses.update(loss.item())#, Y_hat.size(1))

        return losses.avg


def get_statistics(args, dataloader):
    # If using a different dataset than MUSDB18HQ, uncomment this and import sklearn.preprocessing
    '''
    # What follows computes the dataset statistics with sklearn
    scaler = sklearn.preprocessing.StandardScaler()

    spec = torch.nn.Sequential(
        model.STFT(n_fft=args.nfft, n_hop=args.nhop),
        model.Spectrogram(mono=True)
    )

    pbar = tqdm.tqdm(dataloader, disable=args.quiet)
    for x, y in pbar:
        pbar.set_description("Compute dataset statistics")
        X = spec(x)
        scaler.partial_fit(np.squeeze(X))

    std = np.maximum(
        scaler.scale_,
        1e-4*np.max(scaler.scale_)
    )
    np.save('scalermean.npy',scaler.mean_)
    np.save('std.npy', std)


    return scaler.mean_, std
    '''

    # Otherwise, we directly load the MUSDB18-HQ statistics from file
    return np.load('scalermean.npy'), np.load('std.npy')

def main():
    parser = argparse.ArgumentParser(description='Open Unmix Trainer')
    # Loss parameters
    parser.add_argument('--loss', type=str, default="L2freq",
                        choices=[
                            'L2freq', 'L1freq', 'L2time', 'L1time',
                            'L2mask', 'L1mask', 'SISDRtime', 'SISDRfreq',
                            'MinSNRsdsdr', 'CrossEntropy', 'BinaryCrossEntropy',
                        ],
                        help='kind of loss used during training')

    # Dataset paramaters
    parser.add_argument('--dataset', type=str, default="musdb",
                        choices=[
                            'musdb', 'aligned', 'sourcefolder',
                            'trackfolder_var', 'trackfolder_fix'
                        ],
                        help='Name of the dataset.')

    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--output', type=str, default="open-unmix",
                        help='provide output path base folder name')
    parser.add_argument('--model', type=str, help='Path to checkpoint folder')

    # Trainig Parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument(
        '--reduce-samples',
        type=int,
        default=1,
        help="reduce training samples by factor n"
    )

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    parser.add_argument('--patience', type=int, default=140,
                        help='maximum number of epochs to train (default: 140)')
    parser.add_argument('--lr-decay-patience', type=int, default=80,
                        help='lr decay patience for plateau scheduler')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.3,
                        help='gamma of learning rate scheduler decay')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    # Model Parameters
    parser.add_argument('--seq-dur', type=float, default=6.0,
                        help='Sequence duration in seconds'
                        'value of <=0.0 will use full/variable length')
    parser.add_argument('--unidirectional', action='store_true', default=False,
                        help='Use unidirectional LSTM instead of bidirectional')
    parser.add_argument('--nfft', type=int, default=4096,
                        help='STFT fft size and window size')
    parser.add_argument('--nhop', type=int, default=1024,
                        help='STFT hop size')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='hidden size parameter of dense bottleneck layers')
    parser.add_argument('--bandwidth', type=int, default=16000,
                        help='maximum model bandwidth in herz')
    parser.add_argument('--nb-channels', type=int, default=2,
                        help='set number of channels for model (1, 2)')
    parser.add_argument('--nb-workers', type=int, default=0,
                        help='Number of workers for dataloader.')

    # Misc Parameters
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='less verbose during training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args, _ = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    dataloader_kwargs = {'num_workers': args.nb_workers, 'pin_memory': True} if use_cuda else {}

    repo_dir = os.path.abspath(os.path.dirname(__file__))
    repo = Repo(repo_dir)
    commit = repo.head.commit.hexsha[:7]

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    train_dataset, valid_dataset, args = data.load_datasets(parser, args)

    num_train = len(train_dataset)
    indices = list(range(num_train))

    # shuffle train indices once and for all
    np.random.seed(args.seed)
    np.random.shuffle(indices)

    if args.reduce_samples > 1:
        split = int(np.floor(num_train / args.reduce_samples))
        train_idx = indices[:split]
    else:
        train_idx = indices
    sampler = SubsetRandomSampler(train_idx)
    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, **dataloader_kwargs
    )

    stats_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=1,
        sampler=sampler, **dataloader_kwargs
    )

    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1,
        **dataloader_kwargs
    )

    if args.model:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, stats_sampler)

    max_bin = utils.bandwidth_to_max_bin(
        train_dataset.sample_rate, args.nfft, args.bandwidth
    )
    unmix = model.OpenUnmixSingle(
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        hidden_size=args.hidden_size,
        nb_channels=args.nb_channels,
        sample_rate=train_dataset.sample_rate,
        nb_layers=3,
        input_mean=scaler_mean,
        input_scale=scaler_std,
        max_bin=max_bin,
        unidirectional=args.unidirectional,
        power=1,
    ).to(device)
    print('learning rate:')
    print(args.lr)
    optimizer = torch.optim.Adam(
        unmix.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10
    )

    es = utils.EarlyStopping(patience=args.patience)

    # if a model is specified: resume training
    if args.model:
        print('LOADING MODEL')
        model_path = Path(args.model).expanduser()
        with open(Path(model_path, str(len(args.targets)) + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, "model.chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # train for another epochs_trained
        t = tqdm.trange(
            results['epochs_trained'],
            results['epochs_trained'] + args.epochs + 1,
            disable=args.quiet
        )
        train_losses = results['train_loss_history']
        valid_losses = results['valid_loss_history']
        train_times = results['train_time_history']
        best_epoch = results['best_epoch']
        es.best = results['best_loss']
        es.num_bad_epochs = results['num_bad_epochs']
        print('Model loaded')
    # else start from 0
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = train(args, unmix, device, train_sampler, optimizer)
        valid_loss = valid(args, unmix, device, valid_sampler)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(
            train_loss=train_loss, val_loss=valid_loss
        )

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': unmix.state_dict(),
                'best_loss': es.best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            is_best=valid_loss == es.best,
            path=target_path,
        )

        # save params
        params = {
            'epochs_trained': epoch,
            'args': vars(args),
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs,
            'commit': commit
        }

        with open(Path(target_path,  str(len(args.targets)) + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break


if __name__ == "__main__":
    main()
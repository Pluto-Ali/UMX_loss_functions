import argparse
import json
import warnings
from pathlib import Path
import faulthandler

import norbert
import numpy as np
import resampy
import soundfile as sf
import torch
import torchaudio

import model
import utils


def load_model(targets, model_name='umxhq', device='cpu'):
    """
    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)
    """
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        raise NotImplementedError
    else:
        # load model from disk
        with open(Path(model_path, str(len(targets)) + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(model_path) / "model.pth"
        state = torch.load(
            target_model_path,
            map_location=device
        )

        max_bin = utils.bandwidth_to_max_bin(
            44100,
            results['args']['nfft'],
            results['args']['bandwidth']
        )

        unmix = model.OpenUnmixSingle(
            n_fft=results['args']['nfft'],
            n_hop=results['args']['nhop'],
            nb_channels=results['args']['nb_channels'],
            hidden_size=results['args']['hidden_size'],
            max_bin=max_bin
        )

        unmix.load_state_dict(state)
        unmix.stft.center = True
        unmix.eval()
        unmix.to(device)
        print('loadmodel function done')
        return unmix

def separate(
    audio,
    targets,
    model_name='umxhq',
    niter=1, softmask=False, alpha=1.0,
    residual_model=False, device='cpu'
):
    """
    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

    model_name: str
        name of torchhub model or path to model folder, defaults to `umxhq`

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary of all restimates as performed by the separation model.

    """
    # convert numpy audio to torch
    print('loading audio')
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)
    print('audio loaded')
    source_names = targets
    unmix = load_model(
        targets=targets,
        model_name=model_name,
        device=device
    )
    print('model loaded')
    # Obtain the mask from the model
    V = unmix(audio_torch)
    print('separation obtained')
    X = unmix.stft(audio_torch).permute(3, 0, 1, 2, 4)
    # Apply the mask
    mag = torchaudio.functional.complex_norm(X)
    V = [Y_hat * mag for Y_hat in V] #TODO: check if we need to do (Y_hat ** 2) for SNRPSA. Re-run get_statistics
    # From torch to numpy complex, for norbert EM algorithm
    V = np.array([m.cpu().detach().numpy() for m in V])[:, :, 0, :, :]
    V = V.transpose(1,3,2,0)
    X = X.detach().cpu().numpy()[:,0,:,:]
    X = X[..., 0] + X[..., 1] * 1j
    X = X.transpose(0,2,1)
    print('pre-norbert OK')
    # Apply norbert Wiener Filter
    Y_EM = norbert.wiener(V, X.astype(np.complex128), niter,
                       use_softmask=softmask)
    print('norbert OK')
    # back to torch complex for torchaudio ISTFT:
    Y_hats = torch.stack([torch.from_numpy(np.real(Y_EM)), torch.from_numpy(np.imag(Y_EM))]).permute(1,4,3,2,0)
    Y_hats = Y_hats.float().unsqueeze(2).unbind(1)
    y_hats = [unmix.istft(spec, audio_torch.shape[-1]) for spec in Y_hats]
    # back to numpy for BSSeval
    y_hats = [y_hat.cpu().detach().numpy() for y_hat in y_hats]
    print('numpy OK')
    estimates = {}
    for j, name in enumerate(source_names):
        estimates[name] = y_hats[j][0].T #final estimate should be [length,2] and float64
    return estimates

def inference_args(parser, remaining_args):
    # noinspection PyTypeChecker
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    inf_parser.add_argument(
        '--softmask',
        dest='softmask',
        action='store_true',
        help=('if enabled, will initialize separation with softmask.'
              'otherwise, will use mixture phase with spectrogram')
    )

    inf_parser.add_argument(
        '--niter',
        type=int,
        default=1,
        help='number of iterations for refining results.'
    )

    inf_parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='exponent in case of softmask separation'
    )

    inf_parser.add_argument(
        '--samplerate',
        type=int,
        default=44100,
        help='model samplerate'
    )

    inf_parser.add_argument(
        '--residual-model',
        action='store_true',
        help='create a model for the residual'
    )
    return inf_parser.parse_args()


def test_main(
    input_files=None, samplerate=44100, niter=1, alpha=1.0,
    softmask=False, residual_model=False, model='umxhq',
    targets=('vocals', 'drums', 'bass', 'other'),
    outdir=None, start=0.0, duration=-1.0, no_cuda=False
):

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    for input_file in input_files:
        # handling an input audio path
        info = sf.info(input_file)
        start = int(start * info.samplerate)
        # check if dur is none
        if duration > 0:
            # stop in soundfile is calc in samples, not seconds
            stop = start + int(duration * info.samplerate)
        else:
            # set to None for reading complete file
            stop = None

        audio, rate = sf.read(
            input_file,
            always_2d=True,
            start=start,
            stop=stop
        )

        if audio.shape[1] > 2:
            warnings.warn(
                'Channel count > 2! '
                'Only the first two channels will be processed!')
            audio = audio[:, :2]

        if rate != samplerate:
            # resample to model samplerate if needed
            audio = resampy.resample(audio, rate, samplerate, axis=0)

        if audio.shape[1] == 1:
            # if we have mono, let's duplicate it
            # as the input of OpenUnmix is always stereo
            audio = np.repeat(audio, 2, axis=1)

        estimates = separate(
            audio,
            targets=targets,
            model_name=model,
            niter=niter,
            alpha=alpha,
            softmask=softmask,
            residual_model=residual_model,
            device=device
        )
        if not outdir:
            model_path = Path(model)
            if not model_path.exists():
                outdir = Path(Path(input_file).stem + '_' + model)
            else:
                outdir = Path(Path(input_file).stem + '_' + model_path.stem)
        else:
            outdir = Path(outdir)

        outdir.mkdir(exist_ok=True, parents=True)

        for target, estimate in estimates.items():
            print(Path(Path(input_file).stem + '_' + target))
            sf.write(
                str(outdir / Path(target).with_suffix('.wav')),
                estimate,
                samplerate
            )


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='OSU Inference',
        add_help=False
    )

    parser.add_argument(
        'input',
        type=str,
        nargs='+',
        help='List of paths to wav/flac files.'
    )

    parser.add_argument(
        '--targets',
        nargs='+',
        default=['vocals', 'drums', 'bass', 'other'],
        type=str,
        help='provide targets to be processed. \
              If none, all available targets will be computed'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        help='Results path where audio evaluation results are stored'
    )

    parser.add_argument(
        '--start',
        type=float,
        default=0.0,
        help='Audio chunk start in seconds'
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=-1.0,
        help='Audio chunk duration in seconds, negative values load full track'
    )

    parser.add_argument(
        '--model',
        default='umxhq',
        type=str,
        help='path to mode base directory of pretrained models'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)
    faulthandler.enable()

    test_main(
        input_files=args.input, samplerate=args.samplerate,
        alpha=args.alpha, softmask=args.softmask, niter=args.niter,
        residual_model=args.residual_model, model=args.model,
        targets=args.targets, outdir=args.outdir, start=args.start,
        duration=args.duration, no_cuda=args.no_cuda
    )
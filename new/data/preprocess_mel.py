from sqlite3 import paramstyle
import numpy as np
import torch
import torchaudio as T\
import librosa as lr
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm


def transform(filename):
#   if T.__version__ > '0.7.0':
    audio, sr = T.load(filename)
    audio = torch.clamp(audio[0], -1.0, 1.0)
#   else:
#     audio, sr = T.load_wav(filename)
#     audio = torch.clamp(audio[0] / 32767.5, -1.0, 1.0)

    # if params.sample_rate != sr:
    #     raise ValueError(f'Invalid sample rate {sr}.')
    mel_args = {
        'sample_rate': sr,
        'win_length': 1024,
        'hop_length': 256,
        'n_fft': 1024,
        'f_min': 20.0,
        'f_max': sr / 2.0,
        'n_mels': 80,
      'power': 1.0,
      'normalized': True,
    }
    mel_spec_transform = TT.MelSpectrogram(**mel_args)

    with torch.no_grad():
        spectrogram = mel_spec_transform(audio)
        spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
        spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
        np.save('/'+f'{filename}_spec.npy', spectrogram.cpu().numpy())

def main(args):
    filenames = glob(f'{args.dir}/*.wav', recursive=True)
    print(filenames)
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a dataset to train DiffWave')
    parser.add_argument('dir',
        help='directory containing .wav files for training')
    main(parser.parse_args())



def transform(filepath):
#   if T.__version__ > '0.7.0':
    filename = os.path.basename(filepath)
    audio, sr = lr.load(filepath)
    audio = lr.resample(audio, orig_sr=sr, target_sr=22050)
    mel_spect = lr.feature.melspectrogram(audio, sr=22050, n_fft=2048, hop_length=512, power=2)

    cqt_magnitude = np.abs(mel_spect)
    # cqt_log_mag = np.log(cqt_magnitude)

    print(cqt_representation.shape)

    print(os.path.join(OUT_DIR, f'{filename}_cqt.npy'))
    np.save(os.path.join(OUT_DIR, f'{filename}_cqt.npy'), cqt_magnitude)

def main(args):
    filenames = glob(f'{args.dir}/*.wav', recursive=True)
    print(filenames)
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a dataset to train DiffWave')
    parser.add_argument('dir',
        help='directory containing .wav files for training')
    main(parser.parse_args())
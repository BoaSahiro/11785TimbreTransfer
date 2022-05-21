import numpy as np
import librosa as lr
import os

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

OUT_DIR = "/root/timbre/data_syn/cropped"


def transform(filepath):
    filename = os.path.basename(filepath)
    audio, sr = lr.load(filepath, sr=None)
    assert(sr==16000)
    cqt_representation = lr.cqt(audio, sr=sr, hop_length=256)

    cqt_magnitude = np.abs(cqt_representation)
    # cqt_log_mag = np.log(cqt_magnitude)

    print(cqt_representation.shape)

    print(os.path.join(OUT_DIR, f'{filename}_cqt.npy'))
    np.save(os.path.join(OUT_DIR, f'{filename}_cqt.npy'), cqt_magnitude)


def main(args):
    filepaths = glob(f'{args.dir}/*.wav', recursive=True)
    print(filepaths)
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(transform, filepaths),
             desc='Preprocessing', total=len(filepaths)))


if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a dataset to train DiffWave')
    parser.add_argument('--dir', default="/root/timbre/data_syn/cropped",
                        help='directory containing .wav files for training')
    main(parser.parse_args())

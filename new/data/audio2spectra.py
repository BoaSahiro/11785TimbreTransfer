import matplotlib

matplotlib.use("Agg")  # librosa.display includes matplotlib
import librosa.display
import librosa
import numpy as np
from utils import mkdir, read_via_scipy, get_spectrogram, get_cqt, plot_figure
import glob
import os

config = {
    # basic parameters
    "sr": 22050,
    "n_fft": 2048,
    "hop_length": 256,
    "input_type": "exp",  # power, dB with ref_dB, p_log, exp with exp_b. it's input of training data
    "is_mel": True,
    # for spectra
    "n_mels": 256,
    "exp_b": 0.3,
    "ref_dB": 1e-5,
    # for cepstrum
    "dct_type": 2,
    "norm": "ortho",
    # for slicing and overlapping
    "audio_samples_frame_size": 77175,  # 3.5sec * sr
    "audio_samples_hop_length": 77175,
    "output_hei": 256,
    "output_wid": 302,  # num_output_frames = 1 + (77175/hop_length256)
    "use_cqt": True,
    "num_digit": 4,
}


def audio2npys(input_file, config, out_dir):
    # read an audio file and then write a lot of numpy files
    song_name = input_file.split("/")[-1][:-4]
    print("!song_name = {}!".format(song_name))

    y, sr = read_via_scipy(input_file)
    print("dtype={}, sampling rate={}, len_samples={}".format(y.dtype, sr, len(y)))

    Len = y.shape[0]
    cnt = 0
    st_idx = 0
    ed_idx = st_idx + config["audio_samples_frame_size"]
    nxt_idx = st_idx + config["audio_samples_hop_length"]

    while st_idx < Len:
        if ed_idx > Len:
            ed_idx = Len
        data = np.zeros(config["audio_samples_frame_size"], dtype="float32")
        data[: ed_idx - st_idx] = y[st_idx:ed_idx]

        out_var = np.zeros(
            (config["output_hei"], config["output_wid"]), dtype="float32"
        )

        list_spec = []
        list_cqt = []

        w_len = config["n_fft"]
        # list_spec.append(get_spectrogram(data, config, w_len))
        # out_var = list_spec[-1]
        mel_spec = librosa.feature.melspectrogram(
            data,
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
        )

        npy_name = os.path.join(
            out_dir,
            "npy",
            song_name + "_" + str(cnt).zfill(config["num_digit"]) + ".npy",
        )

        np.save(npy_name, mel_spec)
        img_name = os.path.join(
            out_dir,
            "img",
            song_name + "_" + str(cnt).zfill(config["num_digit"]) + ".png",
        )

        # plots: 1. spec 2. ceps (all in single file)
        plot_figure(img_name, mel_spec, config)

        cnt += 1
        st_idx = nxt_idx
        ed_idx = st_idx + config["audio_samples_frame_size"]
        nxt_idx = st_idx + config["audio_samples_hop_length"]


if __name__ == "__main__":
    input_dir = "/home/sahiro/cmu/11785/project/data_yt/guitar/wav"
    ls = sorted(glob.glob(input_dir + "/*.wav"))
    for file in ls:
        print("file = ", file)
        audio2npys(file, config, "data_yt/guitar")

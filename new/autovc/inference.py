import os
# import pickle
import argparse
import torch
import numpy as np
from math import ceil
from model_vc import Generator


device = 'cuda:0'

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

def inference(config):
    G = Generator(config.dim_neck, config.dim_emb, config.dim_pre, config.freq).eval().to(device)
    cp_path = os.path.join(config.save_dir, "weights_log_cqt_down32")
    if os.path.exists(cp_path):
        save_info = torch.load(cp_path)
        G.load_state_dict(save_info["model"])

    emb_org = np.load(config.emb_org)
    emb_org = emb_org[10]
    # emb_org = np.reshape(emb_org, (-1))
    emb_org = torch.from_numpy(emb_org[np.newaxis, :]).to(device)
    emb_trg = np.load(config.emb_trg)
    emb_trg = emb_trg[10]
    # emb_trg = np.reshape(emb_trg, (-1))
    emb_trg = torch.from_numpy(emb_trg[np.newaxis, :]).to(device)

    x_org = np.log(np.load(config.spectrogram_path).T)
    # x_org = np.load(config.spectrogram_path).T
    x_org, len_pad = pad_seq(x_org)
    x_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)

    with torch.no_grad():
        _, x_identic_psnt, _ = G(x_org, emb_org, emb_org)
        if len_pad == 0:
            x_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            x_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

    np.save("result_recon.npy", x_trg.T)
    print("result saved.")

    with torch.no_grad():
        _, x_identic_psnt, _ = G(x_org, emb_org, emb_trg)
        if len_pad == 0:
            x_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            x_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

    np.save("result_trans.npy", x_trg.T)
    print("result saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=0, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    
    parser.add_argument('--dim_emb', type=int, default=16)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    # Training configuration.
    parser.add_argument('--spectrogram_path', type=str, default='/root/timbre/piano_test_cqt.npy')
    parser.add_argument('--emb_org', type=str, default='/root/timbre/data_syn/entries/piano_entry_embeddings.npy')
    parser.add_argument('--emb_trg', type=str, default='/root/timbre/data_syn/entries/guitar_entry_embeddings.npy')
    # parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    # parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    
    # Miscellaneous.
    parser.add_argument('--save_dir', type=str, default="/root/timbre/autovc_cp")

    config = parser.parse_args()
    print(config)
    inference(config)
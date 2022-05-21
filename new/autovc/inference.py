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
    cp_path = os.path.join(config.save_dir, "/root/timbre/autovc_cp/weights_log_cqt_down32_neck32_onehot4_withcross")
    if os.path.exists(cp_path):
        save_info = torch.load(cp_path)
        G.load_state_dict(save_info["model"])


    # one-hot
    ins_list = ['harp', 'trumpet', 'epiano', 'viola', 'piano', 'guitar', 'organ', 'flute']
    ins_org = config.emb_org
    ins_trg = config.emb_trg
    emb_org = ins_list.index(ins_org)
    emb_trg = ins_list.index(ins_trg)
    # emb_org = [i == ins_org for i in ins_list]
    # emb_trg = [i == ins_trg for i in ins_list]
    emb_org = torch.unsqueeze(torch.tensor(emb_org), dim=0).to(device)
    emb_trg = torch.unsqueeze(torch.tensor(emb_trg), dim=0).to(device)

    x_org = np.log(np.load(config.feature_path).T)
    # x_org = np.load(config.spectrogram_path).T
    x_org, len_pad = pad_seq(x_org)
    x_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)

    with torch.no_grad():
        _, x_identic_psnt, _ = G(x_org, emb_org, emb_org)
        if len_pad == 0:
            x_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            x_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

    emb_name = os.path.basename(config.emb_org)
    ins_name_org = emb_name.split('_')[0]

    np.save(os.path.basename(config.feature_path)[:-4] + ins_name_org + "_" + ins_name_org + ".npy", x_trg.T)
    print("result saved.")

    with torch.no_grad():
        _, x_identic_psnt, _ = G(x_org, emb_org, emb_trg)
        if len_pad == 0:
            x_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            x_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

    emb_name = os.path.basename(config.emb_trg)
    ins_name_trg = emb_name.split('_')[0]

    np.save(os.path.basename(config.feature_path)[:-4] + ins_name_org + "_" + ins_name_trg + ".npy", x_trg.T)
    print("result saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=0, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    
    parser.add_argument('--dim_emb', type=int, default=4)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    # Training configuration.
    parser.add_argument('--feature_path', type=str, default='/root/timbre/piano_test.wav_cqt.npy')
    parser.add_argument('--emb_org', type=str, default='piano')
    parser.add_argument('--emb_trg', type=str, default='piano')
    # parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    # parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    
    # Miscellaneous.
    parser.add_argument('--save_dir', type=str, default="/root/timbre/autovc_cp")

    config = parser.parse_args()
    print(config)
    inference(config)
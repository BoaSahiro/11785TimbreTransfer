from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from matplotlib import pyplot as plt

def cor(C_X, C_X_X):
    pccs = []
    for i in range(84):
        pccs.append(np.corrcoef(C_X[i], C_X_X[i])[0, 1])
    pccs = np.array(pccs)
    return pccs



class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.lambda_cd_cross = config.lambda_cd_cross
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step
        self.save_step = config.save_step
        self.save_dir = config.save_dir
        self.summary_writer = None

        # Build the model and tensorboard.
        self.build_model()

            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 1e-4)
        
        self.G.to(self.device)

        # load previous model
        cp_path = os.path.join(self.save_dir, "weights_log_cqt_down32_neck32_onehot4_withcross")
        if os.path.exists(cp_path):
            save_info = torch.load(cp_path)
            self.G.load_state_dict(save_info["model"])
            self.g_optimizer.load_state_dict(save_info["optimizer"])
            self.g_optimizer.state_dict()['param_groups'][0]['lr'] /= 2
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd', 'G/loss_cd_cross']
            
        # Start training.
        print('Start training...')
        start_time = time.time()

        # scaler = torch.cuda.amp.GradScaler()  # FP 16

        pccs = np.zeros((8, 84))
        cnts = np.zeros(8)

        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org, emb_trg = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org, emb_trg = next(data_iter)
            
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 
            emb_trg = emb_trg.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
            x_identic = torch.squeeze(x_identic, dim=1)
            x_identic_psnt = torch.squeeze(x_identic_psnt, dim=1)
            g_loss_id = F.l1_loss(x_real, x_identic)
            # g_loss_norm = F.l1_loss(torch.sum(x_real, 2), torch.sum(x_identic, 2))
            # g_loss_f0

            g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt)
            # g_loss_id = F.l1_loss(x_real, x_identic)   
            # g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt)
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)

            # cross-domain code semantic loss
            x_trans, x_trans_psnt, code_trans_real = self.G(x_real, emb_org, emb_trg)
            code_trans_reconst = self.G(x_trans_psnt, emb_trg, None)
            g_loss_cd_cross = F.l1_loss(code_trans_real, code_trans_reconst)


            for b in range(emb_org.shape[0]):
                t = emb_org[b]
                cnts[t] += 1
                
                pccs[t] += cor(x_real[b].detach().cpu().numpy().T, x_identic_psnt[b].detach().cpu().numpy().T)


            if i % 100 == 0:
                np.save("pccs.npy", pccs)
                np.save("cnts.npy", cnts)
                # plt.figure()
                # plt.xlabel("Note Steps")
                # plt.ylabel("Pearson Correlations")
                
                # plt.plot(pccs[0] / cnts[0], label='harp')
                # plt.plot(pccs[1] / cnts[1], label='trumpet')
                # plt.plot(pccs[2] / cnts[2], label='e-piano')
                # plt.plot(pccs[3] / cnts[3], label='viola')
                # plt.plot(pccs[4] / cnts[4], label='piano')
                # plt.plot(pccs[5] / cnts[5], label='guitar')
                # plt.plot(pccs[6] / cnts[6], label='organ')
                # plt.plot(pccs[7] / cnts[7], label='flute')
                # plt.legend(loc='upper right') # 绘制曲线图例，信息来自类型label
                # plt.savefig("/root/timbre/pccs.png", dpi=300)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd + self.lambda_cd_cross * g_loss_cd_cross
            self.reset_grad()

            # This is a replacement for loss.backward()
            # scaler.scale(g_loss).backward()
            # This is a replacement for optimizer.step()
            # scaler.step(self.g_optimizer)
            # scaler.update()  # This is something added just for FP16
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()
            loss['G/loss_cd_cross'] = g_loss_cd_cross.item()


            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
                self._write_summary(i, g_loss, loss['G/loss_id'], loss['G/loss_id_psnt'], loss['G/loss_cd'], loss['G/loss_cd_cross'])

            # save model checkpoint
            if i % self.save_step == 1 and i > 1000:
                save_info = {
                    "iteration": i,
                    "model": self.G.state_dict(),
                    "optimizer": self.g_optimizer.state_dict()
                }
                save_name = "weights_log_cqt_down32_neck32_onehot4_withcross"
                save_path = os.path.join(self.save_dir, save_name)
                torch.save(save_info, save_path)

    def _write_summary(self, i, loss, loss_id, loss_id_psnt, loss_cd, loss_cd_cross):
        writer = self.summary_writer or SummaryWriter(self.save_dir, purge_step=i)
        writer.add_scalar('train/loss_all', loss, i)
        writer.add_scalar('train/loss_id', loss_id, i)
        writer.add_scalar('train/loss_id_psnt', loss_id_psnt, i)
        writer.add_scalar('train/loss_cd', loss_cd, i)
        writer.add_scalar('train/loss_cd_cross', loss_cd_cross, i)
        writer.flush()

        self.summary_writer = writer

    
    

    
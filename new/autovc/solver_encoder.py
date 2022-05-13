from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import os
import datetime
from torch.utils.tensorboard import SummaryWriter


class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
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
        cp_path = os.path.join(self.save_dir, "weights_log_cqt_down8_neck16")
        if os.path.exists(cp_path):
            save_info = torch.load(cp_path)
            self.G.load_state_dict(save_info["model"])
            # self.g_optimizer.load_state_dict(save_info["optimizer"])
            # self.g_optimizer.state_dict()['param_groups'][0]['lr'] /= 2
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
            
        # Start training.
        print('Start training...')
        start_time = time.time()

        # scaler = torch.cuda.amp.GradScaler()  # FP 16

        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)
            
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
            x_identic = torch.squeeze(x_identic, dim=1)
            x_identic_psnt = torch.squeeze(x_identic_psnt, dim=1)
            g_loss_id = F.l1_loss(x_real, x_identic)
            g_loss_norm = F.l1_loss(torch.sum(x_real, 2), torch.sum(x_identic, 2))
            # g_loss_f0


            g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt)
            # g_loss_id = F.l1_loss(x_real, x_identic)   
            # g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt)
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
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
                self._write_summary(i, g_loss, loss['G/loss_id'], loss['G/loss_id_psnt'], loss['G/loss_cd'])

            # save model checkpoint
            if i % self.save_step == 1:
                save_info = {
                    "iteration": i,
                    "model": self.G.state_dict(),
                    "optimizer": self.g_optimizer.state_dict()
                }
                save_name = "weights_log_cqt_down8_neck16"
                save_path = os.path.join(self.save_dir, save_name)
                torch.save(save_info, save_path)

    def _write_summary(self, i, loss, loss_id, loss_id_psnt, loss_cd):
        writer = self.summary_writer or SummaryWriter(self.save_dir, purge_step=i)
        writer.add_scalar('train/loss_all', loss, i)
        writer.add_scalar('train/loss_id', loss_id, i)
        writer.add_scalar('train/loss_id_psnt', loss_id_psnt, i)
        writer.add_scalar('train/loss_cd', loss_cd, i)
        writer.flush()

        self.summary_writer = writer

    
    

    
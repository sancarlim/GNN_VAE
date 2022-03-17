def huber_loss(self, pred, gt, mask, delta):
        pred = pred * mask
        gt = gt * mask 
        error = torch.sum(torch.where(torch.abs(gt-pred) < delta , (0.5*(gt-pred)**2), torch.abs(gt - pred)*delta - 0.5*(delta**2)), dim=-1)
        overall_sum_time = error.sum(dim=-2) 
        overall_num = mask.sum(dim=-1).type(torch.int) 
        return overall_sum_time, overall_num, error

def frange_cycle_linear(self, start=0.0, stop=1.0, ratio=0.7, period=500):
    step = (stop-start)/(period*ratio) 
    
    if self.beta >= stop:
        if self.global_step % period == 0:
            self.beta = start
    else:
        self.beta += step

def vae_loss(self, preds, mode_probs, gt, mask, KL_terms, z_sample):
    if mode_probs.shape[0] == NUM_MODES:
        preds = preds.view(NUM_MODES, gt.shape[0], -1, 2)
        losses = torch.Tensor().requires_grad_(True).to(preds.device)
        for pred in preds:
            overall_sum_time, overall_num, error = self.huber_loss(pred, gt[:,:,:2], mask, hparams.delta)  
            losses = torch.cat((losses, torch.sum(error,dim=-1).unsqueeze(0)), 0) 

        with torch.no_grad():
            best_mode = torch.argmin(losses, dim=0)

        reconstruction_loss = torch.sum( losses[best_mode, torch.arange(gt.shape[0])]) / torch.sum(overall_num.sum(dim=-1)) 
    else:
        preds = preds.view(gt.shape[0], -1, 2)
        overall_sum_time, overall_num,_ = self.huber_loss(preds[:,:,:2], gt[:,:,:2], mask, hparams.delta)  
        reconstruction_loss = torch.sum(overall_sum_time/overall_num.sum(dim=-2)) 

    std = torch.exp(KL_terms[1] / 2)
    kld_loss = kl_divergence(
    Normal(KL_terms[0], std), Normal(torch.zeros_like(KL_terms[0]), torch.ones_like(std))
    ).sum(-1)
    if self.global_step > 10:
        self.frange_cycle_linear(stop=hparams.beta_max_value, ratio=hparams.beta_ratio_annealing, period=self.beta_period)

    if self.training:
            ### KL PRIOR Loss ###
            std_prior = torch.exp(KL_terms[3] / 2)
            kld_prior = kl_divergence(
                Normal(KL_terms[0], std), Normal(KL_terms[2], std_prior)
            ).sum(-1)
            loss = reconstruction_loss + self.beta * kld_loss.mean() + self.beta * kld_prior.mean() 
            return loss, {'loss': loss, 'Reconstruction_Loss':reconstruction_loss, 'Classification_Loss':classification_loss,  'KL': kld_loss.mean(), 'KL_prior':  kld_prior.mean(),  'beta': self.beta}  
    else:
        loss = reconstruction_loss + self.beta * kld_loss.mean() 
        return loss, {'loss': loss, 'Reconstruction_Loss':reconstruction_loss, 'KL': kld_loss.mean(),  'beta': self.beta}#, 'z0': z0_loss}

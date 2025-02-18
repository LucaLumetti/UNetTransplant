import torch
#qui non servono i gradienti solo il modello in eval e si calcola la loss con  e senza pertubazione
def perturb_model(model, sigma):
    for nn, m in model.named_parameters():
        if m.dim() > 1:
            perturbation = []
            for j in range(m.shape[0]):
                pert = torch.randn(m.shape[1:]).cuda()
                # normalize perturbation
                # compute weight norm
                pert *= (torch.norm(m[j]) / (torch.norm(pert) + 1e-10)) * sigma
                perturbation.append(pert)
            perturbation = torch.stack(perturbation, dim=0)
            m += perturbation

#qui servono i gradienti del modello
def compute_fisher(self, dataset):
    fish = torch.zeros_like(self.net.get_params())

    for ex, lab in zip(inputs, labels):
        self.opt.zero_grad()
        output = self.net(ex.unsqueeze(0))
        loss = #loss di segmentazione

        ###############questo potrebbe non essere necessario
        exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
        loss = torch.mean(loss)
        ################

        loss.backward()
        fish += exp_cond_prob * self.net.get_grads() ** 2

    fish /= (len(dataset.train_loader) * self.args.batch_size) 

    if self.fish is None:
        self.fish = fish
    else:
        self.fish *= self.args.gamma #questo pure potrebbe non sevire
        self.fish += fish




import torch 
import torch.nn as nn

# High magnitude alligned weights
alligned_a, alligned_pa, total = 0, 0, 0
gap_a_b = 0
losses = []
loss = nn.MSELoss()
loss_dict = {}

#model_a, model_b 
for key in model_a.visual.state_dict().keys():
    # MODEL A
    magnitudes = torch.abs(model_a.visual.state_dict()[key])
    threshold = torch.quantile(magnitudes, 0.9)
    high_magnitude_mask_a = magnitudes >= threshold

    # MODEL B
    magnitudes = torch.abs(model_b.visual.state_dict()[key])
    threshold = torch.quantile(magnitudes, 0.9)
    high_magnitude_mask_b = magnitudes >= threshold

    # SIGNS
    sign_mask_b = model_b.visual.state_dict()[key]>0
    sign_mask_a = model_a.visual.state_dict()[key]>0

    mask_a_b = sign_mask_a == sign_mask_b

    new_mask = torch.sum(high_magnitude_mask_a & high_magnitude_mask_b & mask_a_b).item()
    total += torch.sum(high_magnitude_mask_b).item()
    alligned_a += new_mask
    
    #MSE
    gap_a_b = loss(model_a.visual.state_dict()[key], model_b.visual.state_dict()[key])
    loss_dict[key] = gap_a_b

    losses.append(gap_a_b)

print(f'Percentage of alligned weights A: {alligned_a/total}')
print(f'Percentage of alligned weights PA: {alligned_pa/total}')
import math
import torch
import numpy as np

import matplotlib.pyplot as plt


def get_all_gradients(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)


def optimize(optics, optimizer, sampling, num_iter, nshow=20, regularization=None):
    viridis = plt.get_cmap('viridis')

    fig, ax = plt.subplots()

    # torch.autograd.detect_anomaly(True)

    parameters_record = {
        n: []
        for n, _ in optics.named_parameters()
    }

    loss_record = torch.zeros(num_iter)

    show_every = math.ceil(num_iter / nshow)
   
    for i in range(num_iter):

        optimizer.zero_grad()
    
        output = optics(default_input, sampling)

        # Get loss from the accumulator in the output
        loss = output.loss

        if regularization is not None:
            loss = loss + regularization(optics)


        loss_record[i] = loss.detach()
        loss.backward()

        # Record parameter values
        for n, param in optics.named_parameters():
            parameters_record[n].append(param.detach().clone())

        grad = get_all_gradients(optics)
        if torch.isnan(grad).any():
            print("ERROR: nan in grad", grad)
            raise RuntimeError("nan in gradient, check your torch.where() =)")

        optimizer.step()
        
        if i % show_every == 0:
            iter_str = f"[{i:>3}/{num_iter}]"
            L_str = f"L= {loss.item():>6.3f} | grad norm= {torch.linalg.norm(grad)}"
            print(f"{iter_str} {L_str}")
            
            #for mod in optics.modules():   
                #if isinstance(mod, OpticalSurface) and mod.surface is not None:
                #    draw_surface_module(ax, mod.surface, color=viridis(i / num_iter))

    plt.gca().set_aspect("equal")
    plt.show()

    # Plot parameters and loss
    fig, (ax1, ax2) = plt.subplots(2, 1)
    epoch_range = np.arange(0, num_iter)
    ax2.plot(epoch_range, loss_record.detach(), label="loss")
    for n, param in optics.named_parameters():
        if parameters_record[n][0].dim() == 0:
            data = torch.stack(parameters_record[n]).detach().numpy()
            ax1.plot(epoch_range, data, label=n)
    ax1.set_title("parameter")
    ax1.legend()
    ax2.set_title("loss")
    ax2.legend()
    plt.show()


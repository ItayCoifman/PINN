from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import numpy as np


def plot_data(X, bc, y_bc, ic, y_ic, dy_ic_dx=None):
    """
    Plot the input data.

    """
    f = plt.figure(figsize=(18, 9))
    plt.subplot(2, 2, 1)
    plt.title('$X$')
    plt.scatter(X[:, 0].cpu().detach().numpy(), X[:, 1].cpu().detach().numpy(), c='r', label='X')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.subplot(2, 2, 2)
    plt.title('$bc$')
    plt.scatter(bc[:, 0].cpu().detach().numpy(), bc[:, 1].cpu().detach().numpy(), c=y_bc, label='bc')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.subplot(2, 2, 3)
    plt.title('$ic$')
    plt.scatter(ic[:, 0].cpu().detach().numpy(), y_ic.cpu().detach().numpy(), c=ic[:, 1].detach().numpy(), label='ic')
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    if dy_ic_dx is not None:
        plt.subplot(2, 2, 4)
        plt.title('$ {∂u_{ic}} /\ {∂t} $')
        plt.scatter(ic[:, 0].cpu().detach().numpy(), dy_ic_dx.cpu().detach().numpy(), c=ic[:, 1].detach().numpy(),
                    label='ic')
        plt.xlabel('$x$')
        plt.ylabel('$du/dx$')

    plt.show()
    return f


def plot_learning_curve(history):
    """
    Plot the learning curve.
    """
    f = plt.figure(figsize=(12, 5))
    plt.plot(history["loss"], label="Total loss")
    plt.plot(history["loss_bc"], label="BC loss")
    plt.plot(history["loss_ic"], label="IC loss")
    plt.plot(history["loss_pde"], label="PDE loss")
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("$log({Loss})$")
    plt.legend()
    plt.show()
    return f


def plot_solution(model, t_start, t_end, x_start, x_end, num_test_samples=1001):
    t_flat = np.linspace(t_start, t_end, num_test_samples)
    x_flat = np.linspace(x_start, x_end, num_test_samples)
    x, t = np.meshgrid(x_flat, t_flat)
    xt = np.stack([x.flatten(), t.flatten()], axis=-1)

    u = model(xt).cpu().detach().numpy()
    u = u.reshape(t.shape)

    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    plt.pcolormesh(t, x, np.flipud(u), cmap='jet')
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('h(t,x)')

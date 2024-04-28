import torch
from torch import nn


class Scale(nn.Module):
    """
    Scales input data to the range [-1, 1].
    """

    def __init__(self, lb, ub):
        super(Scale, self).__init__()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if not isinstance(lb, torch.Tensor):
            lb = torch.tensor(lb)
        if not isinstance(ub, torch.Tensor):
            ub = torch.tensor(ub)
        self.lb = lb.to(device)
        self.ub = ub.to(device)

    def forward(self, x):
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0


# todo This can be optimized by implementing PINN class that "talks" with the trainer and asks using a dict for the relevant gradients
class HigherOrderGradients(nn.Module):
    def __init__(self, k):
        """
        Initialize the HigherOrderGradients module with the order of derivatives 'k' to compute.

        Parameters:
            k: Order of derivatives to compute. For example, if k = 1, it will return du_dX, du_dt.
               If k = 2, it will return du_dx, du_dxx, du_dt, du_dtt, and so on.
        """
        super(HigherOrderGradients, self).__init__()
        self.k = k

    def forward(self, X, u):
        """
        Compute the spatial and temporal gradients of the solution 'u' with respect to the input 'X'.

        Parameters:
            X: Input tensor.
            u: Output tensor.

        Returns:
            List of computed gradients and their derivatives up to order k.

        Example:
            >>> X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            >>> u = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
            >>> ho_gradients = HigherOrderGradients(k=2)
            >>> gradients = ho_gradients(X, u)
            >>> # The gradients list will contain:
            >>> # [du_dX[:, 0], du_dX[:, 1], du_dx, du_dxx, du_dt, du_dtt]
        """
        # Compute the first gradient of 'u' with respect to 'X'
        du_dX = torch.autograd.grad(
            inputs=X,
            outputs=u,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        gradients = []
        for j in range(du_dX.shape[-1]):
            new_gradients = [du_dX[:, j: j + 1]]
            # Compute higher-order derivatives up to order k
            for _ in range(self.k - 1):
                new_grad = torch.autograd.grad(
                    inputs=X,
                    outputs=new_gradients[-1],
                    grad_outputs=torch.ones_like(new_gradients[-1]),
                    retain_graph=True,
                    create_graph=True
                )[0][:, j:j + 1]
                new_gradients.append(new_grad)
            gradients.extend(new_gradients)
        return gradients


class PINN(nn.Module):
    """
    PyTorch model for solving a wave equation.

    Args:
        layers (list): List containing the number of neurons in each hidden layer.
        lb (float): Lower bound of the input data.
        ub (float): Upper bound of the input data.
        c (float): Coefficient in the wave equation.

    Attributes:
        lb (float): Lower bound of the input data.
        ub (float): Upper bound of the input data.
        c (float): Coefficient in the wave equation.
        scaling_layer (torch.nn.Sequential): Scaling layer to normalize the input data.
        hidden_layers (torch.nn.ModuleList): List of hidden layers in the neural network.
        output_layer (torch.nn.Linear): Output layer of the neural network.

    Methods:
        forward(x): Forward pass through the neural network.
        _func_r(h_tt, h_xx): Compute the residual of the wave equation.
        get_r(X): Calculate the residual of the wave equation for a given input.
        compute_loss(X0, h0, ht0, Xb, X_train, w_h0, w_ht0, w_hb, X0_val, h0_val, ht0_val, Xb_val):
            Compute the loss function.
        get_grad(X0, h0, ht0, Xb, X_train, w_h0, w_ht0, w_hb, X0_val, h0_val, ht0_val, Xb_val):
            Compute the gradients of the model parameters.
    """

    def __init__(self, layers: list, lb: torch.Tensor = None, ub: torch.Tensor = None, input_shape=2):
        super(PINN, self).__init__()
        if lb is None or ub is None:
            self.scaling_layer = nn.Identity()
        else:
            self.scaling_layer = nn.Sequential(Scale(lb, ub))
        self.hidden_layers = nn.ModuleList()
        for j, n in enumerate(layers):
            if j == 0:
                self.hidden_layers.append(nn.Linear(input_shape, n))
            else:
                self.hidden_layers.append(nn.Linear(layers[j - 1], n))
            self.hidden_layers.append(nn.Tanh())

        self.output_layer = nn.Linear(layers[-1], 1)

        # initialize weights
        for m in self.hidden_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t=None, scaling=True):
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if t is not None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.float32)
            x = torch.cat([x, t], dim=1)
        x = x.to(self.hidden_layers[0].weight.device)
        if scaling:
            x = self.scaling_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


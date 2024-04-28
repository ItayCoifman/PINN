import torch
import inspect
from .pinn import HigherOrderGradients, PINN
import numpy as np


class Trainer:
    def __init__(self,
                 model: PINN,
                 X,
                 X_bc,
                 X_ic,
                 y_bc,
                 y_ic,
                 pde,
                 dy_ic_dx=None
                 ):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = model.to(device)
        self.init_data(X, X_bc, X_ic, y_bc, y_ic, dy_ic_dx)
        self.pde = pde
        # number of inputs to the pde
        self.grad_layer = HigherOrderGradients(
            k=int((len(inspect.signature(pde).parameters) - 1) / 2))  # -1 because of the input u

        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        self.history = {"loss": [], "loss_bc": [], "loss_ic": [], "loss_pde": []}

    def test_dims(self):
        assert self.X_bc.dim() == 2
        assert self.X_ic.dim() == 2
        assert self.y_bc.dim() == 2
        assert self.y_ic.dim() == 2
        assert self.X.dim() == 2
        if self.dy_ic_dx is not None:
            assert self.dy_ic_dx.dim() == 2

    def init_data(self, X, X_bc, X_ic, y_bc, y_ic, dy_ic_dx=None):
        # check that all parameters have dim =2
        # and the first dim is the batch size
        # move all the data to torch tensors if they are not
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if dy_ic_dx is None:
            data = [X, X_bc, X_ic, y_bc, y_ic]
            self.dy_ic_dx = None
        else:
            data = [X, X_bc, X_ic, y_bc, y_ic, dy_ic_dx]

        for i, d in enumerate(data):
            if not isinstance(d, torch.Tensor):
                data[i] = torch.tensor(d)
        for i, d in enumerate(data):
            data[i] = d.to(device)
        if dy_ic_dx is None:
            self.X, self.X_bc, self.X_ic, self.y_bc, self.y_ic = data
        else:
            self.X, self.X_bc, self.X_ic, self.y_bc, self.y_ic, self.dy_ic_dx = data

        self.X.requires_grad = True
        if self.dy_ic_dx is not None:
            #self.dy_ic_dx.requires_grad = True
            self.X_ic.requires_grad = True

        self.test_dims()

    def loss_func(self):
        self.adam.zero_grad()

        y_bc_pred = self.model(self.X_bc)
        loss_bc = self.criterion(y_bc_pred, self.y_bc)

        y_ic_pred = self.model(self.X_ic)
        loss_ic = self.criterion(y_ic_pred, self.y_ic)
        # todo This needs to be generalized
        if self.dy_ic_dx is not None:
            du_dx, du_dxx, du_dt, du_dtt = self.grad_layer(self.X_ic, y_ic_pred)
            loss_ic = loss_ic + self.criterion(du_dt, self.dy_ic_dx)

        # todo add gradient loss to the boundary condition

        # this loss is for the PDE initial condition and boundary condition
        loss_data = loss_bc + loss_ic

        u = self.model(self.X)
        grads = self.grad_layer(self.X, u)

        # loss pde for the wave equation
        left, right = self.pde(u, *grads)

        loss_pde = self.criterion(left, right)

        loss = loss_pde + loss_data
        self.history["loss"].append(loss.item())
        self.history["loss_bc"].append(loss_bc.item())
        self.history["loss_ic"].append(loss_ic.item())
        self.history["loss_pde"].append(loss_pde.item())

        loss.backward()
        if self.iter % 100 == 0:
            print(
                f"Iteration: {self.iter}, Loss: {loss} , Loss BC: {loss_bc}, Loss IC: {loss_ic}, Loss PDE: {loss_pde}")
        self.iter = self.iter + 1
        return loss

    def train(self, adam_epochs=1000, lbfgs_epochs=5000):
        self.model.train()
        self.adam = torch.optim.Adam(self.model.parameters())
        for i in range(adam_epochs):
            self.adam.step(self.loss_func)

        if lbfgs_epochs > 0:
            self.lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                lr=1.0,
                max_iter=lbfgs_epochs,
                max_eval=lbfgs_epochs,
                # history_size=50,
                tolerance_grad=1e-7,
                tolerance_change=1.0 * np.finfo(float).eps,
                line_search_fn="strong_wolfe",  # better numerical stability
            )

            self.lbfgs.step(self.loss_func)

        return self.history

    def eval_(self):
        self.model.eval()

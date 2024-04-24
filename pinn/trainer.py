import torch
import inspect
from .pinn import HigherOrderGradients, PINN
import numpy as np


class Trainer:
    def __init__(self,
                 model:PINN,
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

        self.X_bc = X_bc.to(device)
        self.X_ic = X_ic.to(device)
        self.y_bc = y_bc.unsqueeze(1).to(device)
        self.y_ic = y_ic.unsqueeze(1).to(device)

        if dy_ic_dx is not None:
            self.X_ic.requires_grad = True
            self.dy_ic_dx = dy_ic_dx.to(device)
            # self.X_ic.requires_grad = True
        else:
            self.dy_ic_dx = None

        self.X = X.to(device)
        self.X.requires_grad = True

        self.pde = pde
        # number of inputs to the pde
        self.grad_layer = HigherOrderGradients(
            k=int((len(inspect.signature(pde).parameters) - 1) / 2))  # -1 because of the input u

        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        self.history = {"loss": [], "loss_bc": [], "loss_ic": [], "loss_pde": []}

    def loss_func(self):
        # this is more like a not so elegant hack to zero grad both optimizers
        self.adam.zero_grad()

        y_bc_pred = self.model(self.X_bc)
        loss_bc = self.criterion(y_bc_pred, self.y_bc)

        y_ic_pred = self.model(self.X_ic)
        loss_ic = self.criterion(y_ic_pred, self.y_ic)

        if self.dy_ic_dx is not None:
            du_dx, du_dxx, du_dt, du_dtt = self.grad_layer(self.X_ic, y_ic_pred)
            #du_dX, du_dt, du_dx, du_dxx, du_dtt = self.model.gradient(self.X_ic, y_ic_pred)
            loss_ic = loss_ic + self.criterion(du_dt, self.dy_ic_dx)

        # this loss is for the PDE initial condition and boundary condition
        loss_data = loss_bc + loss_ic

        u = self.model(self.X)
        du_dx, du_dxx, du_dt, du_dtt = self.grad_layer(self.X, u)
        #du_dX, du_dt, du_dx, du_dxx, du_dtt = self.model.gradient(self.X, u)

        # todo remove this
        """
        du_dX = torch.stack([du_dx, du_dt], dim=-1)
        du_dxx = torch.autograd.grad(
            inputs=self.X,
            outputs=du_dX,
            grad_outputs=torch.ones_like(du_dX),
            retain_graph=True,
            create_graph=True
        )[0][:, 0]
        """

        # loss pde for the wave equation
        left, right = self.pde(u=u.squeeze(), du_dx=du_dx, du_dt=du_dt, du_dxx=du_dxx, du_dtt=du_dtt)


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
        # optimizeres
        self.adam = torch.optim.Adam(self.model.parameters())

        for i in range(adam_epochs):
            self.adam.step(self.loss_func)

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

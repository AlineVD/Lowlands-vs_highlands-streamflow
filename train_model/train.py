import torch
import numpy as np
import pandas as pd
from train_model import RE_utils as ut
import torch.nn as nn
import matplotlib.pyplot as plt
from apollo import metrics as me
from apollo import streamflow as strf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, train_test_split


### Set global model parameters
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Define Neural Network structure and initialisation procedure
class AntecedentNET(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout_rate):
        super(AntecedentNET, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout_rate
        self.linear_layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, int(hidden_dim/4)),
            #nn.BatchNorm1d(int(hidden_dim/4)),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(hidden_dim/4), 1),
        )


    def forward(self, z):
        z = self.linear_layers(z)
        return z

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def load_network(len_x, len_y, hidden_dim, dropout_rate):

    ### Network initialisation
    net = AntecedentNET(len_x, len_y, hidden_dim, dropout_rate)
    net = nn.DataParallel(net)
    return net.apply(init_weights)


class RELoss(torch.nn.Module):
    def __init__(self):
        super(RELoss, self).__init__()

    def forward(self, prediction, target, psi):
        '''
        Calculate the reflective error loss as per the RELossFunc.

        Parameters:
        - prediction (Tensor): Predicted values from the model.
        - target (Tensor): Ground truth values.
        - psi (Tensor): Element-wise weights for the loss calculation.

        Returns:
        - Tensor: The computed weighted loss.
        '''
        # Compute the element-wise reflective error loss

        if not isinstance(psi, torch.Tensor):
            psi = torch.tensor(psi, dtype=prediction.dtype, device=prediction.device)

        if not prediction.requires_grad:
            prediction.requires_grad_(True)
        if not target.requires_grad:
            target.requires_grad_(True)
        if not psi.requires_grad:
            psi.requires_grad_(True)

        # Compute the element-wise reflective error loss without detaching
        return torch.mean(me.RELossFunc(prediction, target, psi))


def fit(net, x, y, loss_func, loss_func_type, optimizer, verbose, psi):
    loss_list = []
    for i in range(5000):
        y_pred = net(x.float())
        if loss_func_type is None:
            loss = loss_func(y_pred, y.float())
        else:
            loss = loss_func(y_pred, y.float(), psi)
        net.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data)
        if (i % 500 == 0) and verbose:
            print('epoch {}, loss {}'.format(i, loss.data))
    return net


class NeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 input_size=10,
                 hidden_size=64,
                 output_size=1,
                 dropout_rate=0,
                 learning_rate=0.001,
                 weight_decay=0.2,
                 num_epochs=9000,
                 criterion=nn.MSELoss(),
                 patience=10,
                 early_stopping=True,
                 verbose=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.model = load_network(input_size, output_size, hidden_dim=hidden_size,
                                  dropout_rate=dropout_rate).to(device)
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.patience = patience
        self.verbose = verbose
        self.early_stopping = early_stopping

    def fit(self, X_extended, y):

        loss_list = []
        val_loss_list = []
        self.model.train()

        if self.early_stopping is True:
            X_train, X_val, y_train, y_val = train_test_split(X_extended, y, test_size=0.1, random_state=42)

            X_val_data = X_val[:, :-1]
            psi_val = X_val[:, -1:]

            X_val_data = torch.from_numpy(X_val_data).float().to(device)
            y_val = torch.from_numpy(y_val).float().to(device)

            best_val_loss = float('inf')
            patience_counter = 0
        else:
            X_train = X_extended
            y_train = y

        X_train_data = X_train[:, :-1]
        psi_train = X_train[:, -1:]

        X_train_data = torch.from_numpy(X_train_data).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train_data)
            if isinstance(self.criterion, nn.MSELoss):
                loss = self.criterion(outputs, y_train)
            else:
                loss = self.criterion(outputs, y_train, psi_train)
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())

            if epoch % 500 == 0 and self.verbose:
                print(f'epoch {epoch}, loss {loss.item()}')

            if self.early_stopping is True:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_data)
                    if isinstance(self.criterion, nn.MSELoss):
                        val_loss = self.criterion(val_outputs, y_val)
                    else:
                        val_loss = self.criterion(val_outputs, y_val, psi_val)
                    val_loss_list.append(val_loss.item())

                # Check for early stopping
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f'Early stopping at epoch {epoch}, best validation loss: {best_val_loss}')
                    break

        if self.verbose:
            self.plot_loss(loss_list, val_loss_list)

    def predict(self, X_extended):
        self.model.eval()

        # The grid search includes the psi param, a separate evaluation doesn't
        if X_extended.shape[1] == self.input_size:
            X = X_extended
        else:
            X = X_extended[:, :-1]

        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X.detach().clone().float()

        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        return predictions

    def plot_loss(self, losses, val_losses=[]):
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        if len(val_losses) > 1:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()


def train(x, y, verbose=True, loss_func_type=None, psi=None, grid_search=True, early_stopping=True, network_params={}):

    if loss_func_type == 'Reflective':
        loss_func = RELoss()
    else:
        loss_func = torch.nn.MSELoss()

    model = NeuralNetworkRegressor(input_size=x.shape[1], output_size=y.shape[1], criterion=loss_func, verbose=verbose,
                                   **network_params)

    if grid_search is True:
        print('using grid search')

        """
        param_grid = {
            'hidden_size': [64, 128],
            'dropout_rate': [0.1, 0.3, 0.5],
            'learning_rate': [0.001, 0.002, 0.0005],
            'weight_decay': [0.001, 0.01]
        }
        """

        param_grid = {
            'hidden_size': [96, 128],
            'dropout_rate': [0.05, 0.1, 0.2],
            'learning_rate': [0.004, 0.005, 0.006],
            'weight_decay': [0.005, 0.01, 0.02]
        }

        x_and_psi = np.hstack((x, psi))
        grid_search = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error', verbose=2)
        grid_search.fit(x_and_psi, y)

        return grid_search.best_estimator_

    else:
        x_and_psi = np.hstack((x, psi))
        model.fit(x_and_psi, y)
        return model


def evaluate(net, y):

    ### Evaluate Network
    net = net.eval()
    return net(y.float()).data.cpu().numpy()


def calculate_performance_metrics(outdf, years_to_consider, plot=True):

    maxflow = int(0.8 * max(np.array(outdf['Flow'])))

    df = outdf[outdf['Date'].dt.year.isin(years_to_consider)]

    if 'Groundtruth' in outdf.columns:
        groundtruth = 'Groundtruth'
    else:
        groundtruth = 'Flow'

    psi_RE_df = pd.DataFrame(ut.psi_distribution(outdf[groundtruth], 'lognorm'), columns=['psi'])
    psi_RE_df['Date'] = outdf['Date']
    psi_RE = psi_RE_df[psi_RE_df['Date'].dt.year.isin(years_to_consider)]['psi'].squeeze()

    if plot is True:
        strf.scatter_plot(maxflow, df, 'Predicted', 'Flow')
    RMSE = me.RMSE(df['Flow'], df['Predicted'])
    NSE = me.R2(df['Flow'], df['Predicted'])
    RE = me.RE(df['Flow'], df['Predicted'], psi_RE)
    return {'RMSE': RMSE, 'NSE': NSE, 'RE': RE}
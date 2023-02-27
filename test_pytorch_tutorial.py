import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#PIPELINE
# 1) Design model: num_in, num_out, forward pass
# 2) Choose loss and optimizer
# 3) Training loop:
    # - forward pass
    # - backward pass
    # - update

# y = 3 * x
# y = w * x
def LinearReg():
    X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
    Y = torch.tensor([[3],[6],[9],[12]], dtype=torch.float32)

    X_test = torch.tensor([5], dtype=torch.float32)

    n_samples, n_features = X.shape
    print(n_samples, n_features)

    input_size = n_features
    output_size = n_features

    class LinearRegression(nn.Module):

        def __init__(self, input_dim, output_dim):
            super(LinearRegression, self).__init__()
            self.lin = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.lin(x)

    model = LinearRegression(input_size, output_size)

    # training
    lr = 0.1
    n_eps = 100
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print(f"model prediction before: {model(X_test).item():.3f}")

    for epoch in range(n_eps):
        # prediction
        y_pred = model(X)

        #loss
        l = loss(Y, y_pred)

        # backward pass (all relevant gradient computations)
        l.backward()

        # update the weights using optimizer
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1) % 5 == 0:
            [w, b] = model.parameters()
            print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}, model prediction: {model(X_test).item():.3f}")
    return model

def LogReg():
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    n_samples, n_features = X.shape
    print(n_samples, n_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    sc = StandardScaler() # zero mean, unit variance.
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    y_train = y_train.view(y_train.shape[0], 1)
    y_test = y_test.view(y_test.shape[0], 1)

    class LogisticRegression(nn.Module):
        def __init__(self, input_features):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(n_features, 1)

        def forward(self, x):
            y_pred = torch.sigmoid(self.linear(x))
            return y_pred

    model = LogisticRegression(n_features)

    # loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 100
    #training loop
    for epoch in range(num_epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch+1) % 10 == 0:
            print(f"epoch {epoch+1}: loss={loss:.4f}")
    # testing
    with torch.no_grad():
        y_predicted_test = model(X_test)
        y_pred_cls = y_predicted_test.round()
        acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
        print(f"acc: {acc:.4f}")

    return model

def py_torch_dataset():
    data = np.loadtxt("wine.csv")

if __name__ == "__main__":
    LogReg()

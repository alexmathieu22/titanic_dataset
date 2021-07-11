import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F

import os


def train_data_titanic():
    df = pd.read_csv("train.csv")

    #dropping ID and Names because they aren't a relevant information
    df.drop(columns=["PassengerId", "Name"], inplace=True)

    #dropping cabin because there is too much missing information
    df.drop(columns=["Cabin"], inplace=True)

    # ------------MISSING_VALUES------------------ #
    #You can either use a dataframe method (pandas), or scikit's Simple Imputer.
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # df["Age"] = imp.fit_transform(df[["Age"]])


    #Either you imput the most_frequent, or you remove the data. In this case, only 2 rows have missing values, so we'll remove, but here is the code incase you want to see how it's done.
    df.dropna(inplace=True)
    # imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    # df["Embarked"] = imp.fit_transform(df[["Embarked"]])

    #Sex
    dic = {'male': 1, 'female': 0}
    df['Sex'] = df['Sex'].map(dic)
    #You can also use the following: df['Sex'] = pd.get_dummies(df['Sex'],drop_first=True)

    embarked_encoded = pd.get_dummies(df['Embarked'])
    df = pd.concat([df,embarked_encoded],axis=1)

    #you can also use the following code:
    # ohe = OneHotEncoder()
    # emb_ohe = ohe.fit_transform(df[["Embarked"]]).toarray()
    # df_emb = pd.DataFrame(emb_ohe, columns=["C","Q","S"])
    # df = pd.concat([df, df_emb], axis=1)

    df.drop(columns=["Embarked"], inplace=True)
    df.drop(columns=["Ticket"], inplace=True)

    df = df[df["Age"] < 60]

    X, y = df.to_numpy()[:, 1:], df.to_numpy()[:, 0]

    scaler_x = MinMaxScaler()
    X = scaler_x.fit_transform(X)

    return X, y


def submission_data_titanic():
    df = pd.read_csv("test.csv")

    #dropping ID and Names because they aren't a relevant information
    df.drop(columns=["Name"], inplace=True)

    #dropping cabin because there is too much missing information
    df.drop(columns=["Cabin"], inplace=True)

    # ------------MISSING_VALUES------------------ #
    #You can either use a dataframe method (pandas), or scikit's Simple Imputer.
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # df["Age"] = imp.fit_transform(df[["Age"]])

    df['Fare'].fillna(df['Fare'].mean(), inplace=True)

    #Sex
    dic = {'male': 1, 'female': 0}
    df['Sex'] = df['Sex'].map(dic)
    #You can also use the following: df['Sex'] = pd.get_dummies(df['Sex'],drop_first=True)

    embarked_encoded = pd.get_dummies(df['Embarked'])
    df = pd.concat([df,embarked_encoded],axis=1)

    #you can also use the following code:
    # ohe = OneHotEncoder()
    # emb_ohe = ohe.fit_transform(df[["Embarked"]]).toarray()
    # df_emb = pd.DataFrame(emb_ohe, columns=["C","Q","S"])
    # df = pd.concat([df, df_emb], axis=1)

    df.drop(columns=["Embarked"], inplace=True)
    df.drop(columns=["Ticket"], inplace=True)

    X, pass_id = df.to_numpy()[:, 1:], df.to_numpy()[:, 0]

    scaler_x = MinMaxScaler()
    X = scaler_x.fit_transform(X)

    return X, pass_id


def xavier_init(m):
    """ Xavier initialization """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #Here 9 is the number of features given as an input
        self.model = nn.Sequential(
            nn.Linear(9, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        self.model.apply(xavier_init)

    def forward(self, input):
        return self.model(input)

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mb_size = 128

    x_train, y_train = train_data_titanic()

    # Loading the training data
    train_dataset = torch.utils.data.TensorDataset(torch.as_tensor(x_train).float(),
                                            torch.as_tensor(y_train).unsqueeze(-1).float())

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=mb_size)


    for lr in [0.008]:
        net = Net().to(device)

        optimizer = opt.Adam(net.parameters(), lr=lr)

        for epoch in range(400):
            net_loss_run = 0

            for i, (X,y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()

                y_pred = net(X)
                loss = F.binary_cross_entropy(y_pred, y)

                loss.backward()
                optimizer.step()

                net_loss_run += loss.item()

            print(f'Epoch: {epoch}, Loss: {net_loss_run/(i+1)}')
        #     final_loss_test = net_loss_run/(i+1)
        # print(f'Final loss: {final_loss_test}')


# --------------------- SUBMISSION ----------------------- #
    x_submission, pass_id = submission_data_titanic()

    pass_id = pd.DataFrame(pass_id, columns=["PassengerId"])

    x_submission = torch.Tensor(x_submission).to(device)

    with torch.no_grad():
        y_pred = net(x_submission)
        y_pred = torch.where(y_pred<=0.5, 0, 1)

    pred = pd.DataFrame(y_pred.cpu().int().numpy(), columns=["Survived"])
    submit = pd.concat([pass_id.astype(int), pred], axis=1)

    os.remove('submission.csv')

    submit.to_csv('submission.csv', index=False)


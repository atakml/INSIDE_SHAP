from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from codebase.GINUtil import load_gin_dataset
from torch.nn.functional import softmax, log_softmax
from sklearn.neural_network import MLPClassifier
from torch.optim import SGD
import torch
import torch.nn as nn
dataset = "BBBP"

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.fc3 = nn.Linear(hidden_size, output_size)  # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


from traingin import kl_loss

input_size =71
model = MLP(input_size, 20, 2)
epoches = 200
optimizer = SGD(model.parameters(), 0.01)

model.cuda()

training_loader, validation_loader, test_data_loader, rule_to_delete = load_gin_dataset(dataset)
X_train = []
labels = []


for data in training_loader:
    edge_index, features, label = data
    features = features[0]
    features = features.sum(axis=0)#.cpu().numpy()
    label = softmax(label[0][0], -1)
    labels.append(label)
    X_train.append(features)

model.train()


val_labels = []
X_valid = []
for data in validation_loader:
    edge_index, features, label = data
    features = features[0]
    features = features.sum(axis=0)#.cpu().numpy()
    label = softmax(label[0][0], -1)
    val_labels.append(label)
    X_valid.append(features)


test_labels = []
X_test = []
for data in test_data_loader:
    edge_index, features, label = data
    features = features[0]
    features = features.sum(axis=0)#.cpu().numpy()
    label = softmax(label[0][0], -1)
    test_labels.append(label)
    X_test.append(features)

best_acc = 0

early_stop = 0
for epoche in range(epoches):
    model.train()
    for i, data in enumerate(X_train):
        optimizer.zero_grad()
        loss = kl_loss(torch.log_softmax(model(X_train[i]), dim=-1), labels[i])
        loss.backward()
        optimizer.step()
    if epoche % 5 ==0:
        model.eval()
        acc_val = 0
        with torch.no_grad():
            train_loss, valid_loss, test_loss = 0, 0, 0
            for i, data in enumerate(X_valid):
                valid_loss += kl_loss(torch.log(softmax(model(X_valid[i]), dim=-1)), val_labels[i]) 
                if torch.argmax(model(X_valid[i])) == torch.argmax(val_labels[i]):
                    acc_val += 1
            acc_val /= len(X_valid)
            valid_loss /= len(X_valid)
            if acc_val > best_acc:
                early_stop = 0
                acc_test = 0 
                for i, data in enumerate(X_test):
                    test_loss += kl_loss(torch.log(softmax(model(X_test[i]), dim=-1)), test_labels[i])
                    if torch.argmax(model(X_test[i])) == torch.argmax(test_labels[i]):
                        acc_test += 1
                acc_test /= len(X_test)
                test_loss /= len(X_test)
                acc_train = 0
                for i, data in enumerate(X_train):
                    train_loss += kl_loss(torch.log(softmax(model(X_train[i]), dim=-1)), labels[i])
                    if torch.argmax(model(X_train[i])) == torch.argmax(labels[i]):
                        acc_train += 1
                train_loss /= len(X_train)
                acc_train /= len(X_train)
                best_acc = acc_val
                print(f"{epoche=}{acc_train=}, {acc_val=}, {acc_test=}")
                print(f"{epoche=}{train_loss=}, {valid_loss=}, {test_loss=}")
            else:
                early_stop += 1
            if early_stop == 2:
                break
            

"""
#linear_model = LinearRegression()
#linear_model = DecisionTreeRegressor()
#linear_model.fit(X_train, labels)
train_labels = linear_model.predict(X_train)
#clf = MLPClassifier((20, 20, 20, 20), random_state=1,  verbose=False)
#clf = clf.fit(X_train, labels)
train_acc = sum([(labels[0] < 0.5) == (train_labels[i] < 0.5) for i in range(len(labels))])/len(labels)
print(labels[0])
print(f"{train_acc=}")
print("!"*100)

X_test = []
labels = []
#test_labels = linear_model.predict(X_test)

test_acc = clf.score(X_test, labels)#sum([(labels[0] < 0.5) == (test_labels[i] < 0.5) for i in range(len(labels))])/len(labels)
print(f"{test_acc=}")


X_valid = []
labels = []
#valid_labels = linear_model.predict(X_valid)

valid_acc = clf.score(X_valid, labels)#sum([(labels[0] < 0.5) == (valid_labels[i] < 0.5) for i in range(len(labels))])/len(labels)
print(f"{valid_acc=}")"""

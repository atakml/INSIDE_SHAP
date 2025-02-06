import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from codebase.GINUtil import load_gin_dataset
from traingin import kl_loss
from torch.nn.functional import softmax, log_softmax



# MLP Model Definition
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Apply Kaiming initialization
        self.init_weights()

    def init_weights(self):
        # Kaiming Initialization for ReLU activation
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        # Optional: Zero-initialize the biases for stability
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Prepare datasets
def prepare_data(loader):
    X = []
    y = []
    for data in loader:
        _, features, label = data
        features = features[0].sum(axis=0)
        label = softmax(label[0][0], -1)
        X.append(features)
        y.append(label)
    X = torch.stack(X).cuda()
    y = torch.stack(y).cuda()
    return X, y



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    args = parser.parse_args()
    # Define the dataset
    dataset = "mutag"
    input_size = 71
    batch_size = 32  # Set batch size
    num_epochs = 200
    learning_rate = 0.0001
    # Initialize model, optimizer
    model = MLP(input_size, 64, 2).cuda()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    # Load dataset
    training_loader, validation_loader, test_data_loader, rule_to_delete = load_gin_dataset(dataset)


    # Create TensorDatasets
    X_train, y_train = prepare_data(training_loader)
    X_valid, y_valid = prepare_data(validation_loader)
    X_test, y_test = prepare_data(test_data_loader)

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Loss function
    loss_fn = kl_loss

    # Training Loop with Batching
    best_acc = 0
    early_stop = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(log_softmax(outputs, dim=-1), batch_y).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Accuracy
        model.eval()
        val_loss = 0.0
        val_acc = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                outputs = model(batch_x)
                loss = loss_fn(log_softmax(outputs, dim=-1), batch_y).mean()
                val_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                labels = torch.argmax(batch_y, dim=1)
                val_acc += (predictions == labels).sum().item()

            val_acc /= len(valid_dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(valid_loader):.4f}, Val Accuracy: {val_acc:.4f}")

        # Check for early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            early_stop = 0

            # Test Accuracy
            test_acc = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = model(batch_x)
                    loss = loss_fn(log_softmax(outputs, dim=-1), batch_y).mean()
                    test_loss += loss.item()
                    predictions = torch.argmax(outputs, dim=1)
                    labels = torch.argmax(batch_y, dim=1)
                    test_acc += (predictions == labels).sum().item()
                test_acc /= len(test_dataset)
                print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_acc:.4f}")

        else:
            early_stop += 1
            if early_stop == 20:
                print("Early stopping triggered.")
                break


    """

    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from codebase.GINUtil import load_gin_dataset
    from torch.nn.functional import softmax, log_softmax
    from sklearn.neural_network import MLPClassifier
    from torch.optim import SGD
    import torch
    import torch.nn as nn
    dataset = "mutag"

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
        print(epoche)
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

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from surrogatemod.surrogate_utils import load_gin_dataset
from surrogatemod.traingin import kl_loss
from torch.nn.functional import softmax, log_softmax
import math
from datetime import datetime
import glob


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
        features = features[0].mean(axis=0)
        label = softmax(label[0][0], -1)
        X.append(features)
        y.append(label)
    X = torch.stack(X).cuda()
    y = torch.stack(y).cuda()
    return X, y

def load_mlp(dataset, input_size, device="cuda:0"):
    model_files = glob.glob(f"model_{dataset}_epoch*_val_loss*.pth")
    best_model_file = None
    best_loss = float('inf')
    latest_date = None

    for model_file in model_files:
        parts = model_file.split('_')
        date_str = parts[-1].split('.')[0]
        date = datetime.strptime(date_str, '%Y%m%d').date()
        loss = float(parts[-2][8:])

        if latest_date is None or date > latest_date or (date == latest_date and loss < best_loss):
            latest_date = date
            best_loss = loss
            best_model_file = model_file

    if best_model_file is not None:
        model = MLP(input_size, 64, 2).to(device)
        model.load_state_dict(torch.load(best_model_file, map_location=device))
        model.to(device)
        return model
    else:
        raise FileNotFoundError("No suitable model file found.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    args = parser.parse_args()
    # Define the dataset
    dataset = args.dataset_name

    

    # Load dataset
    #training_loader, validation_loader, test_data_loader, rule_to_delete = load_gin_dataset(dataset)
    
    training_loader, validation_loader, test_loader, rule_to_delete = load_gin_dataset(dataset, device="cpu")
    
    
    input_size = next(iter(training_loader))[1].shape[-1]
    print(f"{input_size=}")
    batch_size = 32  # Set batch size
    num_epochs = 400
    learning_rate = 0.0001
    # Initialize model, optimizer
    model = MLP(input_size, 64, 2).cuda()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    # Create TensorDatasets
    X_train, y_train = prepare_data(training_loader)
    X_valid, y_valid = prepare_data(validation_loader)
    X_test, y_test = prepare_data(test_loader)

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Loss function
    loss_fn = kl_loss
    best_val, best_train, best_test = 0, 0, 0
    # Training Loop with Batching
    best_loss = math.inf
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
                loss = loss_fn(log_softmax(outputs, dim=-1), batch_y).sum()
                val_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                labels = torch.argmax(batch_y, dim=1)
                val_acc += (predictions == labels).sum().item()

            val_acc /= len(valid_dataset)
            val_loss /= len(valid_dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Check for early stopping
        if val_loss< best_loss:
            best_loss = val_loss
            early_stop = 0
          
            train_acc =0
            train_loss =0
            for batch_x, batch_y in train_loader:
                outputs = model(batch_x)
                predictions = torch.argmax(outputs, dim=1)
                labels = torch.argmax(batch_y, dim=1)
                train_acc += (predictions == labels).sum().item()
                loss = loss_fn(log_softmax(outputs, dim=-1), batch_y).sum()
                train_loss += loss.item()
            train_acc /= len(train_dataset)
            train_loss /= len(train_dataset)
            
            # Test Accuracy
            test_acc = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = model(batch_x)
                    loss = loss_fn(log_softmax(outputs, dim=-1), batch_y).sum()
                    test_loss += loss.item()
                    predictions = torch.argmax(outputs, dim=1)
                    labels = torch.argmax(batch_y, dim=1)
                    test_acc += (predictions == labels).sum().item()
                test_acc /= len(test_dataset)
                test_loss /= len(test_dataset)
                print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
            model_filename = f"model_{dataset}_epoch{epoch+1}_val_loss{val_loss:.4f}_{datetime.now().strftime('%Y%m%d')}.pth"
            torch.save(model.state_dict(), model_filename)
            best_train_acc, best_val_acc, best_test_acc = train_acc, val_acc, test_acc
            best_train_loss, best_val_loss, best_test_loss = train_loss, val_loss, test_loss

        else:
            early_stop += 1
            if early_stop == 20:
                print("Early stopping triggered.")
                break

    with open('surrogatemod/simplemodel/results.txt', 'a') as f:
        f.write(f"Best Train Accuracy: {best_train_acc:.6f}, Best Val Accuracy: {best_val_acc:.6f}, Best Test Accuracy: {best_test_acc:.6f}\n")
        f.write(f"Best Train Loss: {best_train_loss:.6f}, Best Val Loss: {best_val_loss:.6f}, Best Test Loss: {best_test_loss:.6f}\n")

    print(f"Best Train Accuracy: {best_train_acc:.6f}, Best Val Accuracy: {best_val_acc:.6f}, Best Test Accuracy: {best_test_acc:.6f}")
    print(f"Best Train Loss: {best_train_loss:.6f}, Best Val Loss: {best_val_loss:.6f}, Best Test Loss: {best_test_loss:.6f}")

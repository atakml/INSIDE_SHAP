from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from codebase.GINUtil import load_gin_dataset
from torch.nn.functional import softmax
dataset = ("AlkaneCarbonyl")
training_loader, validation_loader, test_data_loader, rule_to_delete = load_gin_dataset(dataset)
X_train = []
labels = []
for data in training_loader:
    edge_index, features, label = data
    features = features[0]
    features = features.sum(axis=0).cpu().numpy()
    label = softmax(label[0][0], -1)[0].item()
    labels.append(label)
    X_train.append(features)
linear_model = LinearRegression()
#linear_model = DecisionTreeRegressor()
linear_model.fit(X_train, labels)
train_labels = linear_model.predict(X_train)

train_acc = sum([(labels[0] < 0.5) == (train_labels[i] < 0.5) for i in range(len(labels))])/len(labels)
print(f"{train_acc=}")

X_valid = []
labels = []
for data in validation_loader:
    edge_index, features, label = data
    features = features[0]
    features = features.sum(axis=0).cpu().numpy()
    label = softmax(label[0][0], -1)[0].item()
    labels.append(label)
    X_valid.append(features)

valid_labels = linear_model.predict(X_valid)

valid_acc = sum([(labels[0] < 0.5) == (valid_labels[i] < 0.5) for i in range(len(labels))])/len(labels)
print(f"{valid_acc=}")
X_test = []
labels = []
for data in test_data_loader:
    edge_index, features, label = data
    features = features[0]
    features = features.sum(axis=0).cpu().numpy()
    label = softmax(label[0][0], -1)[0].item()
    labels.append(label)
    X_test.append(features)

test_labels = linear_model.predict(X_test)

test_acc = sum([(labels[0] < 0.5) == (test_labels[i] < 0.5) for i in range(len(labels))])/len(labels)
print(f"{test_acc=}")




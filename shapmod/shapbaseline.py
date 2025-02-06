import torch
import numpy as np
from functools import partial
from torch.optim import SGD
from tqdm import tqdm


def generate_subsets_with_size(univers, set_sizes):
    sets = []
    for i in range(set_sizes.shape[0]):
        sets.append(univers[torch.randperm(univers.shape[0])][:set_sizes[i]])
    return sets


def generate_initial_sets(feature_index, num_features, max_set_size, num_of_noisy):
    u = np.arange(num_features)
    u = u[u != feature_index]
    set_sizes = torch.randint(1, max_set_size, (num_of_noisy,))
    return generate_subsets_with_size(u, set_sizes)


def replace_values_with_base_line(feature_matrix, baseline, indices_to_keep):
    u = np.arange(feature_matrix.shape[-1])
    indices_to_replace = torch.from_numpy(np.setdiff1d(u, indices_to_keep))
    mask_matrix = torch.ones_like(feature_matrix)
    mask_matrix[:, indices_to_replace] = 0
    feature_matrix_copy = feature_matrix * mask_matrix + (1 - mask_matrix) * (
                torch.ones_like(feature_matrix) * baseline)
    return feature_matrix_copy


def refine_set(initial_set, univers, eps, evaluator_func):
    best_set, best_value = np.copy(initial_set), evaluator_func(initial_set)
    while True:
        tmp_value = best_value
        np.random.shuffle(univers)
        for j in univers:
            new_set = np.concatenate((best_set, [j])) if j not in best_set else np.setdiff1d(best_set, [j])
            new_value = torch.abs(evaluator_func(new_set))
            if new_value > best_value:
                best_set, best_value = new_set, new_value
        if torch.abs(best_value - tmp_value) < eps:
            break
    return best_set


def evaluator(set_to_evaluate, model, replace_func, edge_index, feature):
    test = model(replace_func(np.concatenate((set_to_evaluate, [feature]))), edge_index) - model(
        replace_func(set_to_evaluate), edge_index)
    return test


def get_sampled_sets(feature_matrix, edge_index, model, feature, max_set_size, num_of_noisy, replace_func, eps=1e-5):
    num_features = feature_matrix.shape[-1]
    init_sets = generate_initial_sets(feature, num_features, max_set_size, num_of_noisy)
    init_values = torch.stack(list(map(lambda x: model(replace_func(x), edge_index), init_sets)))
    best_set = init_sets[torch.argmax(init_values)]
    m_sets = generate_subsets_with_size(np.setdiff1d(np.arange(num_features),
                                                     np.concatenate((best_set, [feature]))),
                                        torch.arange(start=1, end=max_set_size + 1))
    m_sets = list(map(lambda x: np.union1d(best_set, x), m_sets))
    evaluator_to_pass = partial(evaluator, model=model, replace_func=replace_func, edge_index=edge_index,
                                feature=feature)
    m_sets = list(map(lambda x: refine_set(x, np.setdiff1d(np.arange(num_features), [feature]), eps,
                                           evaluator_to_pass), m_sets))
    return m_sets


def loss_function(feature_matrix, edge_index, model, base_line, max_set_size, num_of_noisy):
    num_features = feature_matrix.shape[-1]
    loss = torch.tensor(0.0, requires_grad=True)
    replace_func = partial(replace_values_with_base_line, feature_matrix, base_line)
    for feature in range(num_features):
        sampled_sets = get_sampled_sets(feature_matrix, edge_index, model, feature, max_set_size, num_of_noisy,
                                        replace_func)

        for subset in sampled_sets:
            loss = loss.add(evaluator(subset, model, replace_func, edge_index, feature))
    loss = loss.div(num_features * max_set_size)
    return loss


def find_base_line(data_loader, model, max_set_size, num_of_noisy, feature_size, mask):
    model.eval()
    model.requires_grad = False
    baseline = torch.zeros(feature_size, requires_grad=True)/2
    baseline = torch.nn.Parameter(baseline.float())
    optimizer = SGD([baseline], lr=0.1)
    for _ in range(8):
        for i, data in tqdm(enumerate(data_loader)):
            if not mask[i]:
                continue
            edge_index, features, labels = data
            edge_index, features = edge_index[0], features[0].float()
            target_class = torch.argmax(model(features, edge_index))
            optimizer.zero_grad()
            loss = loss_function(features, edge_index,
                                 lambda x, y: torch.nn.functional.softmax(model(x, y), -1)[target_class],
                                 baseline,
                                 max_set_size, num_of_noisy)
            loss.retain_grad()
            loss.backward(retain_graph=True)
            #print(baseline.grad)
            optimizer.step()
            #print(baseline)
    return baseline

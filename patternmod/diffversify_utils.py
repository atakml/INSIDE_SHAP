import pickle


def load_diffversify_patterns(dataset_name, method="diffversify", lamb=None, lambc=None, lr=None, t1=None, t2=None):
    #if t1 is None:
    #    with open(f"patternmod/res/pattern_hyperparam_{method}_{dataset_name}.pkl", "rb") as file:
    #        t1, t2, _ = pickle.load(file)
    #if lamb is None:
    #    with open(f"patternmod/res/train_hyperparam_{method}_{dataset_name}.pkl", "rb") as file:
    #        lamb, lr, lambc, best_loss = pickle.load(file)
    #with open(f"patternmod/res/{method}_{dataset_name}_args.t1={t1}_{t2}_patterns_{lr}_{lamb}_{lambc}.pkl", "rb") as file:
    with open(f"patternmod/res/{method}_{dataset_name}_patterns_default.pkl", "rb") as file:
        patterns = pickle.load(file)
    #print(patterns)
    patterns = patterns[1]
    final_patterns = {key: [] for key in patterns[0].keys()}
    #final_patterns = patterns[0]
    print(sum(list(map(len, final_patterns.values()))))
    #for value in patterns[1].values():
    #    print(sum(list(map(len, value.values()))))
        #for key, v, in value.items():
    for key, v in patterns[1][2].items():
        final_patterns[key].extend(v)
    for key in final_patterns.keys():
        final_patterns[key] = list(set(set(map(tuple,final_patterns[key]))))
    return final_patterns


def convert_patterns_to_dict(pattern_dict):
    new_format_pattern = []
    for key in pattern_dict.keys():
        for pattern in pattern_dict[key]:
            new_format_pattern.append({"Target": key, 'components': list(map(lambda x: x, pattern)), 'layer': -1, 'type': "diff"})
    return new_format_pattern




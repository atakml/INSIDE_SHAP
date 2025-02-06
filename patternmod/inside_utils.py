import pandas as pd
import re 

def parse_layer_components(l_values):
    layer_component_pairs = re.findall(r'l_(\d+)c_(\w+)', l_values)
    layers = set()
    components_dict = {}
    
    for layer, component in layer_component_pairs:
        layers.add(layer)
        if layer not in components_dict:
            components_dict[layer] = []
        components_dict[layer].append(component)

    return int(list(layers)[0]), list(map(int, list(components_dict.values())[0]))


def read_patterns(dataset_name, file_path=None):
    if file_path is None:
        file_path = f"codebase/ExplanationEvaluation/PatternMining/{dataset_name}_activation_encode_motifs.csv"  # Replace with your CSV file path
    df = pd.read_csv(file_path, header=None)
    df.columns = ["ID", "Details"]
    df['Target'] = df['Details'].str.extract(r'target:(\d+)').astype(int)
    df['c+'] = df['Details'].str.extract(r'c\+:(\d+)').astype(int)
    df['c-'] = df['Details'].str.extract(r'c\-:(\d+)').astype(int)
    df['Score'] = df['Details'].str.extract(r'score:(\d+\.\d+)').astype(float)
    df['Score+'] = df['Details'].str.extract(r'score\+:(\d+\.\d+)').astype(float)
    df['Score-'] = df['Details'].str.extract(r'score\-:(\d+\.\d+)').astype(float)
    df['nb'] = df['Details'].str.extract(r'nb:(\d+)').astype(int)
    df['l_values'] = df['Details'].str.extract(r'=(.*)')
    df['layer'], df['components'] = zip(*df['l_values'].apply(parse_layer_components))
    s = df.to_dict(orient='records')
    if dataset_name in ["Alkanecarbonyl", "Benzen"]:
        if s[0]['ID'] != s[1]["ID"]:
            s = [s[0]] + s 
    return s

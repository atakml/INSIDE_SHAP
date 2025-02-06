from surrogatemod.surrogate_utils import get_the_latest_date, input_feature_size
from os import walk
from GModel import GModel
from modelmod.gnn_utils import evaluate_metric_on_loader
from surrogatemod.traingin import kl_loss
from pathlib import Path
parent_directory = str(Path.cwd().parent)
def get_model_names(dataset, path):
    model_names = [filenames for filenames in next(walk(path))[2] if dataset in filenames and "model" in filenames ]#and "feature" in filenames]
    latest_date = get_the_latest_date(model_names)
    model_names = list(filter(lambda x: list(map(int, x.split("_")[2:4])) == latest_date, model_names)),
                          key=lambda x: float(x.split("_")[-1]))
    time_list = list(map(lambda x: x.split("_")[3], model_names))
    candidate_model_names = []
    for model_time in time_list:
        last_model = max(list(filter(lambda x: x.split("_")[3] == model_time, model_names)), key = lambda x: int(x.split("_")[4]))
        candidate_model_names.append(last_model)
    return candidate_model_names

def evaluate_models(dataset, random=False):
    prefix = "random features/" if random else ""
    path = f"{parent_directory}/shap_inside/models/{prefix}{dataset}"
    training_loader, validation_loader, test_data_loader, _ = load_gin_dataset(dataset_name, "inside", random=random, device=device)
    model_names = get_model_names(dataset, path)
    model_dicts = {}
    for model_name in model_names:
        model = GModel(input_feature_size[dataset], 20, 2)
        model.load_state_dict(torch.load(f"{path}/{best_model_name}"))
        model.eval()
        train_acc, valid_acc, test_acc = 

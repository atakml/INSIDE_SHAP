from datasetmod.datasetloader import load_dataset_gnn
from modelmod.gnn import load_gnn
from patternmod.diffnaps import run_dyfnapps
from patternmod.run_inside import run_inside




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Specify the device to use (default: cuda:0)")
    parser.add_argument("--method", type=str, default="inside", help="Method to mine the patterns (default: inside)")
    parser.add_argument("--t1", type=float, default=0.15)
    parser.add_argument("--t2", type=float, default=0.1)
    parser.add_argument("--lamb", type=float, default=0.2)
    parser.add_argument("--lambc", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=300)
    parser.add_argument("--pmining",  action="store_true", help="Mine the patterns (default: False)")
    args = parser.parse_args()
    if args.method in ["diffnaps", "diffversify"]:
        run_dyfnapps(args.dataset_name, args)
    elif args.method == "inside":
        run_inside(args.dataset_name)
    else:
        raise NotImplemented

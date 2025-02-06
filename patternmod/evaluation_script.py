import os
datasets = ["mutag"]#["BBBP", "ba2", "aids", "AlkaneCarbonyl", "Benzen"]# ["ba2", "aids", "AlkaneCarbonyl", "Benzen", "mutag", "BBBP"]
methods = ["diffversify"]
for dataset in datasets:
    for method in methods:
        command = f"python patternmod/pattern_evaluation.py {dataset} --method {method} --file patterns/{dataset}_patterns.txt --evolution"
        print(command)
        stat = os.system(command)
        assert stat == 0

        command = f"python patternmod/plot.py {dataset} --method {method}"
        print(command)
        stat = os.system(command)
        assert stat == 0


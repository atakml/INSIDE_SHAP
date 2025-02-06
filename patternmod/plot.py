import matplotlib.pyplot as plt
import re
import pandas as pd
import os 
import os
import matplotlib.pyplot as plt
import pandas as pd
import re

# Function to parse the file and extract data
def parse_file(file_path):
    data = []
    
    # Reading the file content all at once
    with open(file_path, 'r') as file:
        content = file.read()  # Read the entire file content
        
        # Split content by 'dataset_name=' to get individual data entries
        entries = content.split('dataset_name=')[1:]  # Skip the first empty string before the first entry
        
        for entry in entries:
            # Adding the prefix 'dataset_name=' back to the entry
            entry = 'dataset_name=' + entry
            
            # Using regex to capture dataset_name, method, number_of_patterns, and metric values
            match = re.search(r"dataset_name='(.*?)' method='(.*?)' number_of_patterns=(\d+) "
                              r"purity_metric=([\d.]+) cover_metric=([\d.]+) weighted_f1_metric=([\d.]+)", entry)
            if match:
                dataset_name = match.group(1)
                method = match.group(2)
                number_of_patterns = int(match.group(3))
                purity_metric = float(match.group(4))
                cover_metric = float(match.group(5))
                weighted_f1_metric = float(match.group(6))
                
                # Storing the extracted information in a list
                data.append({
                    "dataset_name": dataset_name,
                    "method": method,
                    "number_of_patterns": number_of_patterns,
                    "purity_metric": purity_metric,
                    "cover_metric": cover_metric,
                    "weighted_f1_metric": weighted_f1_metric
                })
    
    # Converting list to DataFrame for easier plotting
    df = pd.DataFrame(data)
    return df

# Function to plot the data and save the plot to a file
def plot_data(df):
    # Create a figure for the plot
    plt.figure(figsize=(10, 6))
    
    # Plotting each point for each metric with different colors and markers
    plt.plot(df['number_of_patterns'], df['purity_metric'], 
                label='Purity Metric', color='blue', marker='o')
    print(df['number_of_patterns'].tolist(), df['purity_metric'].tolist())
    plt.plot(df['number_of_patterns'], df['cover_metric'], 
                label='Cover Metric', color='green', marker='s')
    plt.plot(df['number_of_patterns'], df['weighted_f1_metric'], 
                label='Weighted F1 Metric', color='red', marker='^')

    # Adding labels, legend, and grid
    plt.title(f"Pattern Evolution of {df['dataset_name'].iloc[0]} with {df['method'].iloc[0]}")
    plt.xlabel('Number of Patterns')
    plt.ylabel('Metric Values')
    plt.legend()
    plt.grid(True)
    # Save the plot to a file
    save_dir = 'patternmod/plots'  # Specify the directory to save the plots
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    filename = f"{save_dir}/{df['dataset_name'].iloc[0]}_{df['method'].iloc[0]}_pattern_evolution_purity.png"
    
    # Save the figure as a PNG image
    plt.savefig(filename, dpi=300)  # Adjust dpi for better quality
    plt.close()   

# Main function to execute the program
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be used")
    parser.add_argument("--method", type=str, default="inside", help="Method to mine the patterns (default: inside)")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    method = args.method
    file_path = f"{dataset_name}_{method}_pattern_evaluation_res.txt"
    df = parse_file(file_path)
    plot_data(df)



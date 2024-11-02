import os
import pandas as pd

def combine_csv_files(folder_path, output_filename):
    # List to hold individual DataFrames
    dataframes = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_filename, index=False)

if __name__ == "__main__":
    folder_path = "/Users/howardhsu/Desktop/2024Fall_HW/mslab/NegativePrompt-Replication"  # Change this to your folder path
    output_filename = "mistral_istruction_induction_no_shots.csv"  # Change this to your desired output filename
    combine_csv_files(folder_path, output_filename)


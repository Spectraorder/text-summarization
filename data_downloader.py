from datasets import load_dataset, load_from_disk

def download_dataset(dataset_name, save_path, version=None):
    dataset = load_dataset(dataset_name, version)

    dataset.save_to_disk(save_path)
    print(f"Dataset '{dataset_name}' downloaded and saved to '{save_path}'")



def read_local_dataset(local_data_path, split_name='train', num_examples=5):
    dataset = load_from_disk(local_data_path)

    if split_name not in dataset:
        raise ValueError(f"Split '{split_name}' not found in the dataset.")

    test_split = dataset[split_name].to_pandas()  # Shuffle the dataset (optional)
    # Display the first 'num_examples' examples
    # print(test_split[:5])
    test_split[:500].to_csv("CNNML_tiny.csv", index=False)

if __name__ == "__main__":
    # version = "3.0.0"
    # dataset_name = "cnn_dailymail"
    # save_path = "data/cnn_dailymail"
    # download_dataset(dataset_name, save_path, version)
    
    data_path = "data/cnn_dailymail"
    split_name = "train"  # Change this to "test" or "validation" if needed
    num_examples = 5  # Change this to the number of examples you want to display
    read_local_dataset(data_path, split_name, num_examples)

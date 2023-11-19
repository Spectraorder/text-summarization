from datasets import load_dataset

def download_dataset(dataset_name, save_path, version=None):
    dataset = load_dataset(dataset_name, version)

    dataset.save_to_disk(save_path)
    print(f"Dataset '{dataset_name}' downloaded and saved to '{save_path}'")




if __name__ == "__main__":
    version = "3.0.0"
    dataset_name = "cnn_dailymail"
    save_path = "data/"
    download_dataset(dataset_name, save_path, version)

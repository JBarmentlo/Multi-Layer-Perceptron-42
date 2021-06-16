from modules import load_model
from utils import create_dataset_from_path, calculate_and_display_metrics
import os
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A typical logistic regression')
    parser.add_argument("data_path", nargs = '?', default="data/data.csv", help="path to input data")
    parser.add_argument("model_name", help="model name")
    args = parser.parse_args()
    m = load_model(args.model_name)
    dataset_path = os.path.join(os.environ["BASE_DIR"], args.data_path)
    d = create_dataset_from_path(dataset_path)
    calculate_and_display_metrics(m, d.x, d.y)

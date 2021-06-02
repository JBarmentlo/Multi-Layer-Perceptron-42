from modules import load_model
from utils import create_dataset_from_path, calculate_and_display_metrics


if __name__ == "__main__":
    m = load_model("mymodel")
    d = create_dataset_from_path()
    calculate_and_display_metrics(m, d.x, d.y)

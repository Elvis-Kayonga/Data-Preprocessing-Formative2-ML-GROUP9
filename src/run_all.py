from data_pipeline import main as run_pipeline
from train_product_model import train_and_evaluate


if __name__ == "__main__":
    run_pipeline()
    train_and_evaluate()

from DataIngestion import data_ingestion
from DataPreprocessor import building_preprocessor
from DataTraining import train_model
from DataEvaluation import eval_model


def run_pipeline():
    print("Starting pipeline...")

    print("Data is being ingested...")
    data_ingestion()
    
    print("Data preprocessing...")
    building_preprocessor()
    
    print("Model is being trained...")
    train_model()

    print("Evaluating model...")
    eval_model()

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
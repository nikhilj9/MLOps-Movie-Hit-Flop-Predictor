from prefect_flow import training_flow

if __name__ == "__main__":
    training_flow.serve(name="ml_training")

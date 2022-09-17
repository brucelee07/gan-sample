from pathlib import Path

Base = Path(__file__).resolve().parent

Data_Fold = Base / "new_data"


class Config:
    lr = 3e-2
    dropout = 0.2
    hidden1 = 54
    hidden2 = 54
    epochs = 10
    batch_size = 8
    node_size = 54
    input_dim = 54
    hidden_dim = 54
    layer_dim = 2
    output_dim = 1
    window_length = 1

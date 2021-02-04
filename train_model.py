import torch
import yaml

from cnn_vae.dataset.dataset import get_dataloaders
from cnn_vae.model.model import Model
from cnn_vae.model.train_utils import Runner

with open("configs/model_config.yaml", 'r') as f:
    model_config = yaml.load(f)

model = Model(**model_config)
dataloaders = get_dataloaders("data", batch_size=16)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

runner = Runner()

runner.train(
    model=model,
    optimizer=optimizer,
    loaders=dataloaders,
    logdir="./logs",
    num_epochs=10,
    verbose=True,
    load_best_on_end=True,
    main_metric="ELBO",
)


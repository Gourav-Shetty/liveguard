"""pytorchexample: A Flower / PyTorch app."""

import os
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, set_seed, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # FedProx knob: 0.0 = plain FedAvg local training, > 0 = FedProx.
    # Override per-run with: flwr run . --run-config "proximal-mu=0.1"
    proximal_mu: float = context.run_config.get("proximal-mu", 0.0)

    # Seed knob for multi-seed statistical runs.
    # Override per-run with: flwr run . --run-config "seed=1"
    seed: int = context.run_config.get("seed", 42)
    set_seed(seed)

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy (aggregation itself is unchanged for
    # FedProx -- the proximal term is applied client-side in task.train();
    # see task.py. This keeps the aggregation rule identical between the
    # FedAvg and FedProx conditions, so any performance difference is
    # attributable specifically to the proximal term, not to a different
    # aggregator.)
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr, "proximal_mu": proximal_mu}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        data_dir = os.environ.get(
            "LIVEGUARD_DATA_DIR", "/mnt/c/Users/mirza/Downloads/livegaurd"
        )
        # Filename encodes the experiment condition so sweep runs
        # (different mu / seed) don't clobber each other.
        mu_tag = f"mu{proximal_mu}".replace(".", "p")
        model_out = os.path.join(data_dir, f"stage1_cnn_{mu_tag}_seed{seed}.pth")

        print(f"\nSaving final model to {model_out}...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, model_out)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load centralized validation set (NOT the held-out test set)
    val_dataloader = load_centralized_dataset()

    # Evaluate the global model
    val_loss, val_acc = test(model, val_dataloader, device)

    return MetricRecord({"accuracy": val_acc, "loss": val_loss})
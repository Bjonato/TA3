# Trading DQN Example

This repository provides a minimal setup to train a Deep Q-Network (DQN) agent on OHLCV trading data.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Place your CSV file with columns `open`, `high`, `low`, `close`, and `volume` in a known location. Then run:

```bash
python train_dqn.py --data PATH_TO_YOUR_DATA.csv --episodes 100
```

The training script will print episode rewards and a simple evaluation score at the end.

Hyperparameters such as learning rate, buffer size, and epsilon schedule can be modified via command-line arguments.

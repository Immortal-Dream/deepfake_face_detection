from configs.xception_bsl import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
args = parser.parse_args()

num_epochs = args.epochs

print(f"Training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    train.train(train_loader)
    train.val(val_loader)

print("complete.")

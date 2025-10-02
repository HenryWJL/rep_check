import matplotlib.pyplot as plt
from typing import Optional, List


def plot_curves(
    num_epochs: int,
    loss: List,
    accuracy: List,
    train: Optional[bool] = True
) -> None:
    assert len(loss) == len(accuracy)
    epochs = range(1, num_epochs + 1, num_epochs // len(loss))
    split = "Training" if train else "Validation"
    plt.figure(figsize=(12, 5))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-')
    plt.title(f'{split} Loss')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle="--", alpha=0.6)
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'r-')
    plt.title(f'{split} Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
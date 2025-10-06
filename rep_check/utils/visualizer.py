import matplotlib.pyplot as plt
from typing import Optional, List


def plot_curves(
    num_epochs: int,
    loss: List,
    accuracy: List,
    filename: Optional[str] = None
) -> None:
    assert len(loss) == len(accuracy)
    epochs = range(1, num_epochs + 1, num_epochs // len(loss))
    plt.figure(figsize=(12, 5))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle="--", alpha=0.6)
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'r-')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0, dpi=300)
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("models/ml_results.json") as f:
    ml_results = json.load(f)["accuracies"]

with open("models/rnn_results.json") as f:
    rnn_results = json.load(f)

with open("models/rnn_history.json") as f:
    rnn_history = json.load(f)

all_results = dict(ml_results)
all_results["SimpleRNN (DL)"] = rnn_results["accuracy"]

# ---- Chart 1: Bar chart comparing all models ----
names = list(all_results.keys())
accs = [all_results[n] * 100 for n in names]
colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#DD8452"]  # ML blue, DL orange

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(names, accs, color=colors[: len(names)])
ax.axhline(80, color="red", linestyle="--", linewidth=1, label="80% Target")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison — ML vs SimpleRNN (Deep Learning)")
ax.set_ylim(0, 100)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 1.5, f"{acc:.2f}%",
            ha="center", va="bottom", fontweight="bold")
plt.xticks(rotation=15, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("assets/accuracy_comparison.png", dpi=150)
plt.close()

# ---- Chart 2: RNN training curves ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
epochs = range(1, len(rnn_history["accuracy"]) + 1)

axes[0].plot(epochs, [a * 100 for a in rnn_history["accuracy"]], marker="o", label="Train")
axes[0].plot(epochs, [a * 100 for a in rnn_history["val_accuracy"]], marker="o", label="Validation")
axes[0].axhline(80, color="red", linestyle="--", linewidth=1)
axes[0].set_title("SimpleRNN Accuracy per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()

axes[1].plot(epochs, rnn_history["loss"], marker="o", label="Train")
axes[1].plot(epochs, rnn_history["val_loss"], marker="o", label="Validation")
axes[1].set_title("SimpleRNN Loss per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("assets/rnn_training_curves.png", dpi=150)
plt.close()

print("Saved charts to assets/accuracy_comparison.png and assets/rnn_training_curves.png")
print(json.dumps(all_results, indent=2))

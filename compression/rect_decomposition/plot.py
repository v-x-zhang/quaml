import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

orig  = pd.read_csv("original.csv")
recon = pd.read_csv("reconstructed.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

def draw_rectangles(ax):
    with open("rectangles.csv") as f:
        lines = f.read().strip().split("\n")[1:]  # skip header
    seg_x, seg_y = [], []
    for line in lines:
        if line.strip() == "":
            if seg_x:
                ax.plot(seg_x, seg_y, color="orange", linewidth=1.2, alpha=0.8)
            seg_x, seg_y = [], []
        else:
            x, y = map(float, line.split(","))
            seg_x.append(x)
            seg_y.append(y)
    if seg_x:
        ax.plot(seg_x, seg_y, color="orange", linewidth=1.2, alpha=0.8)

# Left: original points + detected rectangle outlines
axes[0].scatter(orig["x"], orig["y"], s=6, color="steelblue", alpha=0.6, label="original")
draw_rectangles(axes[0])
axes[0].set_title("Original points + detected rectangles")
axes[0].legend(handles=[
    mpatches.Patch(color="steelblue", label="original points"),
    mpatches.Patch(color="orange",    label="detected rectangles"),
])

# Right: reconstructed points
axes[1].scatter(recon["x"], recon["y"], s=6, color="tomato", alpha=0.6, label="reconstructed")
draw_rectangles(axes[1])
axes[1].set_title("Reconstructed points")
axes[1].legend(handles=[
    mpatches.Patch(color="tomato",  label="reconstructed points"),
    mpatches.Patch(color="orange",  label="rectangle outlines"),
])

plt.tight_layout()
plt.savefig("plot.png", dpi=150)
plt.show()
print("Saved plot.png")

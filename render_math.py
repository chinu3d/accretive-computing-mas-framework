import matplotlib.pyplot as plt

formulas = [
    r"\Psi_A(\mathbf{r}, t) = A \cdot \exp(-|\mathbf{r} - \mathbf{p}|) \cdot \exp(i(\omega t + \theta))",
    r"\mathrm{Overlap}(A, B) = A_A A_B \cdot \exp\left(-\frac{1}{2} ||\mathbf{x}_A - \mathbf{x}_B||^2\right) \cdot \cos(\theta_A - \theta_B)",
    r"F(\Theta) = \sum_{i} \sum_{j > i} \left( 1 - \mathrm{Overlap}(A_i, A_j) - \delta \right)",
    r"\frac{\partial \theta_i}{\partial t} = -\alpha \frac{\partial F}{\partial \theta_i}"
]

for i, f in enumerate(formulas):
    fig = plt.figure(figsize=(8, 1))
    # use mathtext
    fig.text(0.5, 0.5, f"${f}$", size=16, ha='center', va='center', usetex=False)
    plt.axis('off')
    plt.savefig(f"math_{i}.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    print(f"Generated math_{i}.png")

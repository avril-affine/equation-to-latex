from IPython.lib.latextools import latex_to_png
import matplotlib.pyplot as plt


def create_latex(filename, eq, fontsize=35):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.5, 0.5, '$%s$' % eq, fontsize=fontsize,
            ha='center', va='center')
    ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)

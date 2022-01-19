import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import librosa
from datetime import datetime
import numpy as np
import seaborn as sn
import pandas as pd


def plot_spectrogram(spec, title=None, ylabel="Freq Bin", aspect="auto", xmax=None):
    """Plot a spectrogram

    Args:
        spec (2D Array): log mel spectrogram
    """
    fig, axs = plt.subplots(1, 1)

    # set axes labels
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("Frame")

    # show spectogram
    # im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    im = axs.imshow(spec, origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    """Plot raw waveform data"""
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape

    # set axes labels
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1)
    axes.set_ylabel("Amplitude")
    axes.set_xlabel("Time")

    # show waveform
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)

    figure.suptitle(title)
    plt.show(block=False)


def plot_losses(train_loss_history, val_loss_history, title=None, save_dir=None):
    """Plot validation and training loss in one graph"""
    # set axes labels
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Losses")
    axs.set_ylabel("Loss")
    axs.set_xlabel("Epoch")

    # show losses
    plt.plot(train_loss_history, label="training loss")
    plt.plot(val_loss_history, label="validation loss")
    plt.legend()
    if save_dir is not None:
        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        save_path = f"{save_dir}/train_val_loss{date}.png"
        fig.savefig(save_path, dpi=fig.dpi)
    plt.show()


def plot_classwise_prob(probs, classes, exp_name=None):
    """plots and saves the classwise probabilites.
    Args:
        probs (torch.Tensor): probs calculated with the Solver.get_classwise_prob() method. It has a shape of (num_classes, num_classes).
        classes (list of str): classes names
        exp_name (str, optional): The plot will be saved using this name. Defaults to None.
    """
    probs = probs.cpu().numpy()

    fig, axs = plt.subplots(nrows=len(classes) // 2, ncols=2, figsize=(8, 12))
    plt.yticks(np.arange(0, 1))
    for i, class_name in enumerate(classes):
        barlist = axs[i // 2, i % 2].bar(np.arange(0, len(classes)), probs[i, :])
        barlist[i].set_color("r")
        axs[i // 2, i % 2].set_title(f"class {class_name}")
        axs[i // 2, i % 2].set_ylim([0, 100])
        axs[i // 2, i % 2].set_xlabel("Labels")
        axs[i // 2, i % 2].set_ylabel("Probability")

        formatter = mticker.ScalarFormatter()
        axs[i // 2, i % 2].xaxis.set_major_formatter(formatter)
        axs[i // 2, i % 2].xaxis.set_major_locator(
            mticker.FixedLocator(np.arange(0, len(classes) + 1, 1))
        )
        axs[i // 2, i % 2].yaxis.set_major_formatter(formatter)
        axs[i // 2, i % 2].yaxis.set_major_locator(
            mticker.FixedLocator(np.arange(0, 100 + 1, 20))
        )

    if exp_name is not None:
        fig.suptitle(exp_name, fontsize=20)
        save_path = f"checkpoints/{exp_name}/classwise_prob_{exp_name}.png"
    else:
        save_path = f"checkpoints/{exp_name}/classwise_prob.png"

    fig.tight_layout(pad=3.0)
    fig.savefig(save_path, dpi=fig.dpi)

    plt.show(block=False)


def plot_conf_matrix(conf_matrix, classes, set, exp_name=None):

    df_cm = pd.DataFrame(conf_matrix, classes, classes)
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes()
    sn.set(font_scale=1.5)  # for label size
    sn.heatmap(
        df_cm, annot=True, annot_kws={"size": 18}, ax=ax, cmap="YlGnBu", fmt="g"
    )  # font size
    if exp_name is not None:
        ax.set_title(f"{exp_name}_{set}")
        save_path = f"checkpoints/{exp_name}/conf_matrix_{exp_name}{set}.png"
    else:
        save_path = f"checkpoints/{exp_name}/conf_matrix{set}.png"

    ax.set_xlabel("predicted labels", fontsize=18)
    ax.set_ylabel("ground truth labels", fontsize=18)
    ax.tick_params(axis="x", labelsize="medium")
    ax.tick_params(axis="y", labelsize="medium")
    fig.tight_layout()
    fig.savefig(save_path, dpi=fig.dpi)
    plt.show(block=False)

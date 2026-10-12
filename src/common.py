#!/usr/bin/env python3
"""
Common functions and parameters for machine-learning scripts.
"""
from collections.abc import Callable
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as met

import utils

# Code should be run with versions defined in requirements.txt
warnings.simplefilter(action="ignore", category=FutureWarning)

DF_DFC = "DFC"
DF_VDFC = "VDFC"
DF_RIDFC = "RIDFC"
DF_RIVDFC = "RIVDFC"
NUM_COEFFS = {
  DF_DFC: 53,
  DF_VDFC: 53,
  DF_RIDFC: 17,
  DF_RIVDFC: 17,
}
DF_ALL_COEFFS = NUM_COEFFS.keys()

plt.rcParams["pdf.fonttype"] = 42 # embedded
plt.rcParams["figure.dpi"] = 300


def generate_coeffs_columns(
  num_coeffs: int,
  prefix: str = "",
  suffix: str = "",
) -> list[str]:
  """
  Generate a list ["c_00", "c_01", ..., "c_<num_coeffs>"].
  Each name is eventually prefixed or suffixed:
  ["<prefix> c_00 <suffix>", ...]
  """
  return [
    " ".join(filter(None, [prefix, f"c_{c:02}", suffix]))
    for c in range(num_coeffs)
  ]

def f1_normalize(conf_matrix: np.ndarray) -> np.ndarray:
  """
  F1 normalize a matrix. Each cell is the harmonic average of
  row-wise and column-wise normalized matrix. Mask out zeros.
  """
  # Also avoids divide by zero warning.
  conf_matrix = np.ma.masked_equal(conf_matrix, 0)
  precision_norm = conf_matrix / conf_matrix.sum(axis=0)
  recall_norm = conf_matrix / conf_matrix.sum(axis=1)
  f1_norm = 2 * precision_norm * recall_norm
  f1_norm /= (precision_norm + recall_norm)
  return f1_norm


def plot_confusion_matrix(
  y_test: np.ndarray,
  y_pred: np.ndarray,
  title: str,
  out_path: str,
  labels: list | None = None,
  display_labels: list | None = None,
  fig_size: tuple | None = None,
  xticks_rotation: float = 0,
  inner_font_size: float = 12,
  before_save_hook: Callable | None = None,
) -> None:
  """
  Compute, display and export confusion matrices plots.
  """
  utils.log("Computing confusion matrices")

  if labels is None:
    labels = np.unique(y_test)

  if display_labels is None:
    display_labels = labels

  conf_matrix = met.confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
  )

  print("Confusion matrix")
  print(conf_matrix)

  utils.log("Plotting...")

  conf_norm = f1_normalize(conf_matrix)

  conf_display = met.ConfusionMatrixDisplay(
    confusion_matrix=conf_norm,
    display_labels=display_labels,
  )

  conf_display.plot(
    cmap=plt.cm.Blues,
    im_kw={"vmin": 0, "vmax": 1},
  )

  size = len(conf_matrix)
  for i in range(0, size):
    for j in range(0, size):
      text = conf_display.text_[i, j]
      text.set_text(str(conf_matrix[i, j]))
      text.set_fontsize(inner_font_size)

  plt.title(title, fontsize=10)
  plt.xticks(rotation=xticks_rotation)

  if fig_size:
    conf_display.figure_.set_size_inches(fig_size)

  if before_save_hook is not None:
    before_save_hook()

  plt.savefig(out_path, bbox_inches="tight")

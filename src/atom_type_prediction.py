#!/usr/bin/env python3
"""
Element and atom type prediction module.
Usage:
  ./atom_type_prediction.py
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  import cuml.accel
  cuml.accel.install()
  print("Using cuML acceleration.", file=sys.stderr)
except ModuleNotFoundError:
  print("Not using GPU.", file=sys.stderr)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model as lm
from sklearn import metrics as met
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import common
import utils

# DF_PATH = "../data/atomtypes_1-6.hdf"
# OUT_DIR = "../out_atomtypes_1-6"
DF_PATH = "../data/atomtypes_1-7.hdf"
OUT_DIR = "../out_atomtypes_1-7"

COLOR_MAP = {
  "C": "tab:gray",
  "N": "tab:blue",
  "O": "tab:red",
  "F": "tab:green",
}
MMFF94_JSON_PATH = "../data/mmff94_atomtypes.json"

KEY_ELEMENT = "Element"
KEY_ATOM_TYPE = "Atom Type MMFF94"
KEY_DFC = "DF Coeff. (6-311G**-rifit)"
KEY_RIDFC = "Rot. Inv. DF Coeff. (6-311G**-rifit)"
KEY_VDFC = "DF Difference Coeff. (6-311G**-rifit)"
KEY_RIVDFC = "Rot. Inv. DF Difference Coeff. (6-311G**-rifit)"
KEY_COEFFS = {
  "DFC": KEY_DFC,
  "RIDFC": KEY_RIDFC,
  "VDFC": KEY_VDFC,
  "RIVDFC": KEY_RIVDFC,
}
ELEMENTS_ORDER = [ "C", "N", "O", "F" ]

RANDOM_STATE = 1
VALIDATION_PROP = 0.2
N_SPLITS = 5
SHOW_PLOTS = False

MODELS = {
  "LR": lm.LogisticRegression(max_iter=10000),
  "Ri": lm.RidgeClassifier(class_weight="balanced"),
  "RF": RandomForestClassifier(
    n_estimators=50,
    random_state=RANDOM_STATE,
  ),
  "XGB": XGBClassifier(
    enable_categorical=True,
    tree_method="hist",
    n_estimators=50,
    device="gpu",
    random_state=RANDOM_STATE,
  ),
}

_atom_types_count: list


def _generate_element_lookup(df: pd.DataFrame) -> dict[str, str]:
  """
  Generate a lookup table that associates to each atom type, the
  element.
  """
  types = df[KEY_ATOM_TYPE].unique()
  return {
    k: df[df[KEY_ATOM_TYPE] == k][KEY_ELEMENT].iloc[0]
    for k in types
  }


def _init_atom_types_labels() -> dict[str, str]:
  """
  Load JSON of names and return dictionary.
  Select the first symbol and add ellipsis if several symbols exist
  in order to provide a label usable in plotting.
  """
  with open(MMFF94_JSON_PATH, "r", encoding="utf8") as f:
    full_dict = json.load(f)

  atom_types_lookup = {}
  for number, obj in full_dict.items():
    symbol = obj["symbols"][0]
    if len(obj["symbols"]) == 2:
      symbol += ", " + obj["symbols"][1]
    elif len(obj["symbols"]) > 2:
      symbol += ", ..."

    atom_types_lookup[int(number)] = f"{number}: {symbol}"

  return atom_types_lookup


def load_and_curate(
  key_coeff: str,
  path: str = DF_PATH,
) -> pd.DataFrame:
  """
  Load the data, reshape, and generate element lookup.
  """
  utils.log("Loading...")
  df = pd.read_hdf(path)
  df = reshape(df, key_coeff)
  df = curate(df)
  return df


def reshape(df: pd.DataFrame, key_coeff: str) -> pd.DataFrame:
  """
  Discard unwanted columns and expand sub-arrays as proper columns
  while renaming long names.
  """
  df = df.rename(columns={v: k for k, v in KEY_COEFFS.items()})
  cols = [KEY_ELEMENT, KEY_ATOM_TYPE, key_coeff]
  df = df[cols]
  df[KEY_ELEMENT] = pd.Categorical(df[KEY_ELEMENT], ELEMENTS_ORDER)

  num_coeffs = len(df[key_coeff][0])

  coeffs_cols = common.generate_coeffs_columns(num_coeffs)
  coeffs = pd.DataFrame(
    np.column_stack(list(zip(*df[key_coeff]))),
    columns=coeffs_cols,
  )

  # Copy missing columns
  for key in [KEY_ELEMENT, KEY_ATOM_TYPE]:
    coeffs[key] = df[key]

  # Reorder columns
  return coeffs[[*cols[:-1], *coeffs_cols]]


def reshape_all_coeffs(df: pd.DataFrame) -> pd.DataFrame:
  """
  Discard unwanted columns and expand sub-arrays as proper columns,
  keeping all coefficients.
  """
  df = df.rename(columns={v: k for k, v in KEY_COEFFS.items()})
  cols = [KEY_ELEMENT, KEY_ATOM_TYPE, *KEY_COEFFS.keys()]
  df = df[cols]
  df[KEY_ELEMENT] = pd.Categorical(df[KEY_ELEMENT], ELEMENTS_ORDER)

  all_coeffs = pd.DataFrame()
  all_cols = []

  for key_coeff in KEY_COEFFS:
    num_coeffs = len(df[key_coeff][0])

    coeffs_cols = common.generate_coeffs_columns(
      num_coeffs, key_coeff
    )

    coeffs = pd.DataFrame(
      np.column_stack(list(zip(*df[key_coeff]))),
      columns=coeffs_cols,
    )

    all_coeffs = pd.concat((all_coeffs, coeffs), axis=1)
    all_cols += coeffs_cols

  # copy missing columns
  for key in [KEY_ELEMENT, KEY_ATOM_TYPE]:
    all_coeffs[key] = df[key]

  # reorder columns
  return all_coeffs[[*cols[:-len(KEY_COEFFS)], *all_cols]]


def curate(
  df: pd.DataFrame,
  threshold: int = N_SPLITS,
) -> pd.DataFrame:
  """
  Removes unique or few appearing occurrences (less than `threshold`
  samples) of atom types. This allows proper cross-validation.
  """
  utils.log("Curating data...")
  lonelies = df[KEY_ATOM_TYPE].value_counts()
  atom_types = lonelies.index[lonelies <= threshold].values
  indices = df[df[KEY_ATOM_TYPE].isin(atom_types)].index

  print("Removed atom types, rows, # out of #")
  print(atom_types, ", ", indices, len(indices), " out of ", len(df))

  return df.drop(index=indices).reset_index(drop=True)


def display_stats(df: pd.DataFrame) -> None:
  """
  Plot and print stats. 
  """
  utils.log("Computing stats...")
  print("Statistics:")
  elements = df[KEY_ELEMENT].value_counts()
  print(elements)

  types = df[[KEY_ELEMENT, KEY_ATOM_TYPE]].groupby(
    KEY_ELEMENT, observed=True
  ).value_counts()

  print(types)
  all_elements = types.index.map(lambda t: t[0])
  # Remove element from index
  types.index = types.reset_index()[KEY_ATOM_TYPE]

  color_map = COLOR_MAP.copy()
  color_map["C"] = "#ddd"

  fig, ax = plt.subplots()
  scale = 0.9
  fig.set_size_inches(6.4 * scale, 4.8 * scale)
  types.plot.bar(
    color=all_elements.map(color_map),
    xlabel="Atom type",
    ylabel="Number of samples",
    rot=0,
    edgecolor="black",
    linewidth=0.5,
  )

  inset = ax.inset_axes([0.1, 0.15, 0.8, 0.8])

  wedges, labels, autotexts = inset.pie(
    elements,
    colors=[color_map[e] for e in elements.index.to_list()],
    autopct="%1.1f%%",
    textprops={"weight": "bold"},
    labels=elements.index.to_list(),
    startangle=90,
    counterclock=False,
  )

  for wedge in wedges:
    wedge.set_edgecolor("black")
    wedge.set_linewidth(0.5)

  for label in labels:
    e = label.get_text()
    label.set_color(COLOR_MAP[e])

  for t in autotexts:
    num = t.get_text()[:-1]
    t.set_text(f"{num} %")
    t.set_color("white")

  # C
  labels[0].set_color("black")
  autotexts[0].set_color("black")

  # N
  n_pos = autotexts[2].get_position()
  autotexts[2].set_position((n_pos[0] - 0.04, n_pos[1] + 0.05))

  # F
  f_label_pos = labels[3].get_position()
  labels[3].set_position((f_label_pos[0] + 0.05, f_label_pos[1]))

  f_atx_pos = autotexts[3].get_position()
  autotexts[3].set_position((f_atx_pos[0] + 0.25, f_atx_pos[1] + 0.2))
  autotexts[3].set_color(COLOR_MAP["F"])

  plt.savefig(
    f"{OUT_DIR}/atom_types_proportions.png",
    bbox_inches="tight",
    dpi=400,
  )


def train_and_evaluate_models(
  df: pd.DataFrame,
  key_coeff: str,
  key_model: str,
  atom_types_lookup: dict[str, str],
  model,
) -> None:
  """
  Cross-validation analysis and final assessment of models with test
  set and confusion matrices.
  """
  global _atom_types_count

  utils.log(f"Model {key_model}")
  print(f"Model {key_model}")
  x_cols = df.columns[2:] # skip targets
  df_train, df_test = train_test_split(
    df,
    test_size=VALIDATION_PROP,
    random_state=RANDOM_STATE,
  )

  for key in [KEY_ELEMENT, KEY_ATOM_TYPE]:
    utils.log(f"For {key}:")
    print(f"For {key}:")
    y_train = df_train[key]

    le = LabelEncoder()
    y_codes = le.fit_transform(y_train)

    if key == KEY_ATOM_TYPE:
      _atom_types_count = df_train[key].value_counts()

    skf = StratifiedKFold(
      n_splits=N_SPLITS,
      random_state=RANDOM_STATE,
      shuffle=True,
    )

    cv_score = cross_val_score(
      model,
      df_train[x_cols],
      y_codes,
      cv=skf,
      scoring="accuracy", # F1 micro avg.
    )

    print(f"Accuracy for {N_SPLITS}-fold CV:")
    print(cv_score)
    print("Average and std dev: ", cv_score.mean(), cv_score.std())

    utils.log("Fitting...")
    print("Fitting")
    utils.tic(f"{key_model} {key} {key_coeff} fit and pred")
    fitted = model.fit(df_train[x_cols], y_codes)
    y_pred = fitted.predict(df_test[x_cols])
    y_pred = le.inverse_transform(y_pred)

    if hasattr(model, "n_iter_"):
      print(f"done in {model.n_iter_} iterations")
    utils.tac()

    print("Test metrics")
    print_metrics(y_pred, df_test[key])

    labels = np.unique(y_train)

    if key == KEY_ATOM_TYPE:
      plot_labels = [atom_types_lookup[at] for at in labels]

      plot_labels, labels = np.array(
        sorted(np.c_[plot_labels, labels], key=_atom_type_sort_key)
      ).T
      labels = labels.astype(int)
    else:
      labels = sorted(labels, key=_atom_type_sort_key)
      plot_labels = labels

    scale = 1 # main
    inner_font_size = 10 if key == KEY_ATOM_TYPE else 9 # main
    # scale = 0.6 # SI
    # inner_font_size = 6 if key == KEY_ATOM_TYPE else 9 # SI

    target_key = "el" if key == KEY_ELEMENT else "at"
    common.plot_confusion_matrix(
      df_test[key],
      y_pred,
      title=f"{key.replace('Type MMFF94', 'type')} " \
            f"({key_coeff}s)",
      out_path=f"{OUT_DIR}/{target_key}_{key_model}_{key_coeff}.png",
      labels=labels,
      display_labels=plot_labels,
      fig_size=(
        (14 * scale, 11 * scale) if key == KEY_ATOM_TYPE
        else (7 * scale, 6 * scale)
      ),
      xticks_rotation=90 if key == KEY_ATOM_TYPE else 0,
      inner_font_size=inner_font_size,
      before_save_hook=(
        _make_draw_squares_hook(df_test)
        if key == KEY_ATOM_TYPE
        else None
      ),
    )


def _atom_type_sort_key(arr: list) -> tuple[str, int]:
  """
  Sort atom types by atom element first, then by total count of said
  atom type in the dataset.
  """
  label = arr[0]

  if ":" in label: # atom type
    splits = label.split(":")
    atom = splits[1].strip()[0] # first letter of the atom type label
    mmff = int(splits[0].strip())
    order = -_atom_types_count[mmff]
  else: # element
    atom = arr[0][0]
    order = arr[0]

  return (ELEMENTS_ORDER.index(atom), order)


def _make_draw_squares_hook(df: pd.DataFrame):
  """
  Make a callable that draws colored blocks on a confusion matrix.
  """
  df = df[[KEY_ELEMENT, KEY_ATOM_TYPE]]
  counts = df.groupby(KEY_ELEMENT, observed=True).nunique()

  def _inner() -> None:
    ax = plt.gca()
    origin = 0

    for element in ELEMENTS_ORDER:
      offset = counts.loc[element][KEY_ATOM_TYPE]
      ax.add_patch(plt.Rectangle(
        (origin - 0.5, origin - 0.5),
        offset, offset,
        edgecolor=COLOR_MAP[element],
        linewidth=2,
        fill=False,
        clip_on=False,
      ))
      origin += offset

  return _inner


def print_metrics(true: list, pred: list) -> None:
  """
  Print different metrics.
  """
  report = met.classification_report(
    true, pred, zero_division=0,
  )
  print(report)


def main() -> None:
  """
  Cache the atom types lookup, load dataset and run LR and RF.
  """
  utils.tic("Main")
  utils.mkdirp(OUT_DIR)
  atom_types_lookup = _init_atom_types_labels()

  for key_coeff in KEY_COEFFS:
    utils.log(key_coeff)
    print(key_coeff)
    df = load_and_curate(key_coeff)
    display_stats(df)

    for key_model, model in MODELS.items():
      train_and_evaluate_models(
        df, key_coeff, key_model, atom_types_lookup, model,
      )

  utils.tac()

  if SHOW_PLOTS:
    plt.show()

  utils.log("Done.")

if __name__ == "__main__":
  main()

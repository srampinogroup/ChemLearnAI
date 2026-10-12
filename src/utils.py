#!/usr/bin/env python3
"""
Minimalistic logging and timing utility functions. Can add breakpoint
on unhandled exception with `AUTO_DEBUG` set to true.
"""
import sys
from pathlib import Path
import pdb
import time
import traceback
import warnings
import pandas as pd
from colorama import init as _colorama_init
from colorama import Back, Fore, Style

AUTO_DEBUG = False
DEFAULT_OUT = sys.stderr

_tictac: list = []


def log(s: str) -> None:
  """
  Print a message with a color in std err.
  """
  print(
    f"{Fore.BLUE}{Style.BRIGHT}{s}{Style.RESET_ALL}",
    file=DEFAULT_OUT,
  )


def err(s: str) -> None:
  """
  Error message in std err.
  """
  print(
    f"{Fore.RED}{Style.BRIGHT}{s}{Style.RESET_ALL}",
    file=DEFAULT_OUT,
  )


def warn(s: str) -> None:
  """
  Issue a warning with orange color.
  """
  warnings.warn(
    f"{Fore.YELLOW}{s}{Style.RESET_ALL}",
    file=DEFAULT_OUT,
  )


def debug(s: str) -> None:
  """
  Print a strongly highlighted debug message.
  """
  print(
    f"{Back.YELLOW}{Fore.BLACK}*** {s} ***{Style.RESET_ALL}",
    file=DEFAULT_OUT,
  )


def tic(s: str) -> None:
  """
  Push a new counting task on the stack of timers.
  """
  _tictac.append((s, time.process_time()))


def tac(should_print: bool = True) -> tuple:
  """
  Pop the timer with time elapsed.
  """
  last = _tictac.pop()
  elapsed = last[0], time.process_time() - last[1]

  if should_print:
    print_time(elapsed)

  return elapsed


def print_time(elapsed: tuple) -> None:
  """
  Print elapsed times in the timers stack.
  """
  print(f"{Fore.GREEN}{'> ' * len(_tictac)}"
      + f"{Style.BRIGHT}{elapsed[1]:.05f} s{Style.NORMAL} spent in "
      + f"{Style.BRIGHT}{elapsed[0]}"
      + f"{Style.RESET_ALL}",
    file=DEFAULT_OUT,
  )


def mkdirp(path: str) -> None:
  """
  Create if not already there all directories in provided `path` as
  would do the `mkdir -p <path>` bash command.
  """
  Path(path).mkdir(parents=True, exist_ok=True)


def _debughook(etype, value, tb):
  """
  Standard traceback print with debugger break. Used when
  `AUTO_DEBUG` is true to automatically pause and start pdb on
  exception.
  """
  traceback.print_exception(etype, value, tb)
  print()
  pdb.pm()


def _init() -> None:
  """
  Initialize colorama and set pandas prefs. Setup the debugger hook.
  """
  _colorama_init(autoreset=True)
  pd.set_option("display.max_rows", 500)
  pd.set_option("display.max_columns", 500)
  pd.set_option("display.width", 150)

  if AUTO_DEBUG:
    sys.excepthook = _debughook


_init()

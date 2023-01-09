"""Deep learning loggers.

Example:
    ```python
    logger = LoggerCollection(
        ConsoleLogger(),
        JsonLogger(),
        TensorBoardLogger(),
    )

    summary = {"train/loss": 0, "test/loss": 0.2}
    logger.log_summary(summary, step=1024, epoch=1)

    lg_state = logger.state_dict()  # Get logger state.
    logger.load_state_dict(lg_state)  # Restore logger.
    ```
"""
from __future__ import annotations
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path
import json
import yaml

from .pytypes import Summary, LoggerState


class Logger(ABC):
    """An abstract base class of loggers."""

    @abstractmethod
    def log_summary(self, summary: Summary, step: int, epoch: int) -> None:
        pass

    def log_hparams(self, hparams: dict) -> None:
        pass

    @abstractmethod
    def state_dict(self) -> LoggerState:
        pass

    @abstractmethod
    def load_state_dict(self, state: LoggerState) -> None:
        pass


class LoggerCollection(Logger):
    """Support to use multiple loggers at the same time.

    Args:
        loggers: Loggers.
    """

    def __init__(self, *loggers) -> None:
        self._loggers: tp.Sequence[Logger] = loggers

    def log_summary(self, summary: Summary, step: int, epoch: int) -> None:
        for lg in self._loggers:
            lg.log_summary(summary, step, epoch)

    def log_hparams(self, hparams: dict) -> None:
        for lg in self._loggers:
            lg.log_hparams(hparams)

    def state_dict(self) -> LoggerState:
        return [lg.state_dict() for lg in self._loggers]

    def load_state_dict(self, state: LoggerState) -> None:
        for lg, lg_state in zip(self._loggers, state):
            lg.load_state_dict(lg_state)


class ConsoleLogger(Logger):
    """Print values on console.

    Args:
        print_fun: Function to print summary or hparams.
    """

    def __init__(self, print_fun: tp.Callable = print) -> None:
        self._print_fun = print_fun

    def log_summary(self, summary: Summary, step: int, epoch: int) -> None:
        summary = dict(step=step, epoch=epoch, **summary)
        self._print_fun(summary)

    def log_hparams(self, hparams: dict) -> None:
        self._print_fun(yaml.dump(hparams, allow_unicode=True))

    def state_dict(self) -> LoggerState:
        return dict()

    def load_state_dict(self, state: LoggerState) -> None:
        del state  # unused.
        return None


class DiskLogger(Logger):
    """A logger that dumps log in local disk.

    Args:
        log_dir: str
    """

    log_file_name = "log.json"
    hparams_file_name = "hparams.yaml"

    def __init__(
        self, save_dir: str | Path, log_file_name="log.json", hparams_file_name="hparams.yaml"
    ) -> None:
        super().__init__()
        self._log_file = Path(save_dir, log_file_name)
        self._hparams_file = Path(save_dir, hparams_file_name)

        self._log = []
        self._hparams = dict()

    def log_summary(self, summary: Summary, step: int, epoch: int) -> None:
        summary = dict(step=step, epoch=epoch, **summary)
        self._log.append(summary)
        self._tmp_log_file.write_text(json.dumps(self._log, indent=2))
        self._tmp_log_file.rename(self._log_file)

    def log_hparams(self, hparams: dict) -> None:
        self._hparams = dict(self._hparams, **hparams)
        self._tmp_hparams_file.write_text(yaml.dump(self._hparams, allow_unicode=True))
        self._tmp_hparams_file.rename(self._hparams_file)

    def state_dict(self) -> LoggerState:
        return {"_log": self._log, "_hparams": self._hparams}

    def load_state_dict(self, state: LoggerState) -> None:
        self._log = state["_log"]
        self._hparams = state["_hparams"]

    @property
    def _tmp_log_file(self) -> Path:
        return self._log_file.with_suffix("tmp")

    @property
    def _tmp_hparams_file(self) -> Path:
        return self._log_file.with_suffix("tmp")

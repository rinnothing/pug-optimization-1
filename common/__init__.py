from typing import Protocol

from scipy.optimize import OptimizeResult
import numpy as np


class StateResult(OptimizeResult):
    """
    Represents result and internal state of optimization
    Should be exactly like `OptimizeResult`, but also store information on previous guesses

    Attributes
    ----------
    guesses
    """

    guesses : np.ndarray = np.array([])

    def add_guess(self, guess):
        self.guesses = np.append(self.guesses, guess)

    # to be continued


class StepCalculator(Protocol):
    """
    Represents abstract algorithm to determine new step size
    """

    def __call__(self, state: OptimizeResult) -> int:
        pass


class StopDeterminer(Protocol):
    """
    Represents abstract algorithm to determine need to stop
    """

    def __call__(self, state: OptimizeResult) -> bool:
        pass


class Optimizer(Protocol):
    """
    Represents abstract algorithm to optimize the result
    """

    def __call__(self, fun, x, step: StepCalculator, stop: StopDeterminer) -> StateResult:
        pass


class Visualizer(Protocol):
    """
    Represents abstract algorithm to visualize the result
    """

    def __call__(self, state: StateResult):
        pass

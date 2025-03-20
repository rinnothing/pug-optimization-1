from abc import abstractmethod

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

    guesses = np.array([])

    def add_guess(self, guess):
        self.guesses = np.append(self.guesses, guess)

    # to be continued

class StepCalculator:
    """
    Represents abstract algorithm to determine new step size
    """

    @abstractmethod
    def get_step(self, state: OptimizeResult) -> int:
        pass

class StopDeterminer:
    """
    Represents abstract algorithm to determine need to stop
    """

    @abstractmethod
    def check_stop(self, state: OptimizeResult) -> bool:
        pass

class Optimizer:
    """
    Represents abstract algorithm to optimize the result
    """

    @abstractmethod
    def optimize(self, fun, x, step: StepCalculator, stop: StopDeterminer) -> StateResult:
        pass

class Visualizer:
    """
    Represents abstract algorithm to visualize the result
    """

    @abstractmethod
    def visualize(self, state: StateResult):
        pass
from typing import Protocol

from scipy.optimize import OptimizeResult


def zero(x):
    return 0


class StateResult(OptimizeResult):
    """
    Represents result and internal state of optimization
    Should be exactly like `OptimizeResult`, but also store information on previous guesses

    Attributes
    ----------
    guesses vals you guessed on every step
    history some information on bounds and other stuff you need after run
    function is a function on which it did optimizations
    """
    guesses: list = []
    history: list = []

    def add_guess(self, guess):
        self.guesses.append(guess)

    def add_history(self, val):
        self.history.append(val)

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

    def __call__(self, fun, x, step: StepCalculator, stop: StopDeterminer, *args, **kwargs) -> StateResult:
        pass


class Visualizer(Protocol):
    """
    Represents abstract algorithm to visualize the result.
    When path is specified, should put the result there.
    """

    def __call__(self, state: StateResult, limits, freq, path: str = None):
        pass

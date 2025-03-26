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
    success: bool = False
    guesses: list = []
    history: list = []

    def add_guess(self, guess):
        self.guesses.append(guess)

    def add_history(self, val):
        self.history.append(val)

    def get_res(self):
        if not self.success or len(self.guesses) == 0:
            return None

        return self.guesses[-1]

    # to be continued

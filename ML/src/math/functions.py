from __future__ import annotations

from enum import Enum
from math import exp, tanh
from typing import Callable, Mapping

OneDFunction = Callable[[float], float]
TwoDFunction = Callable[[float, float], float]


class FunctionName(str, Enum):
    CONSTANT = "Costante"
    F8 = "Funzione 8"
    F9 = "Funzione 9"
    F10 = "Funzione 10"
    F11 = "Funzione 11"
    F13 = "Funzione 13"
    F14 = "Funzione 14"
    F15 = "Funzione 15"

class UnknownFunctionError(ValueError):
    """Raised when an unsupported function name (or dimensionality) is requested."""

class ParametersNotSetError(RuntimeError):
    """Raised when a function is requested before update_params() has been called."""

class Functions:    
    # Step size used for the finite-difference approximation of derivatives.
    _FD_STEP = 1e-4

    _VALID_BORDERS = (1, 2, 3, 4)

    def __init__(self) -> None:
        self._params: Mapping[str, float] | None = None

        self._dispatch_1d: dict[FunctionName, Callable[[float, int], OneDFunction]] = {
            FunctionName.CONSTANT: self._get_constant_1d,
            FunctionName.F9: self._get_border_f9,
            FunctionName.F10: self._get_border_f10,
        }
        self._dispatch_2d: dict[FunctionName, Callable[[float], TwoDFunction]] = {
            FunctionName.CONSTANT: self._get_constant_2d,
            FunctionName.F8: self._get_panettone_r,
            FunctionName.F11: self._get_freccia_11,
            FunctionName.F13: self._get_panettone_t,
            FunctionName.F14: self._get_panettone_l,
            FunctionName.F15: self._get_freccia_00,
        }

    def update_params(self, params: Mapping[str, float]) -> None:
        self._params = dict(params)

    @property
    def params(self) -> Mapping[str, float]:
        if self._params is None:
            raise ParametersNotSetError("Call update_params() before requesting a function.")
        return self._params

    def get_by_name_1d(self, name: str, magnitude: float, border: int = 1) -> OneDFunction:
        self._validate_border(border)
        builder = self._resolve(name, self._dispatch_1d)
        return builder(magnitude, border)

    def get_by_name_2d(self, name: str, magnitude: float) -> TwoDFunction:
        builder = self._resolve(name, self._dispatch_2d)
        return builder(magnitude)

    @staticmethod
    def _resolve(name: str, dispatch: Mapping[FunctionName, Callable]) -> Callable:
        try:
            function_name = FunctionName(name)
        except ValueError as exc:
            raise UnknownFunctionError(f"Function name is not valid: {name!r}") from exc

        try:
            return dispatch[function_name]
        except KeyError as exc:
            raise UnknownFunctionError(
                f"{name!r} is not available for this dimensionality"
            ) from exc

    @classmethod
    def _validate_border(cls, border: int) -> None:
        if border not in cls._VALID_BORDERS:
            raise ValueError(f"border must be one of {cls._VALID_BORDERS}, got {border!r}")

    # ------------------------------------------------------------------ #
    # Constant
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_constant_1d(magnitude: float, _border: int = 1) -> OneDFunction:
        return lambda x: magnitude

    @staticmethod
    def _get_constant_2d(magnitude: float) -> TwoDFunction:
        return lambda x, y: magnitude

    # ------------------------------------------------------------------ #
    # "Panettone" manufactured solutions
    # ------------------------------------------------------------------ #
    def _get_panettone_r(self, magnitude: float) -> TwoDFunction:
        magnitude *= 10.0

        def f(x: float, y: float) -> float:
            p = self.params
            mu, b1, b2, sigma, alpha = p["mu"], p["b1"], p["b2"], p["sigma"], p["alpha"]

            left = exp(-x) * (
                (1.0 - y) * ((2.0 * mu + b1 * (1.0 - x)) * y + (sigma - mu) * x * y)
                + (2.0 * mu + b2 * (1.0 - 2.0 * y)) * x
            )
            right = exp(-1.0 + (x - 1.0) / alpha) * (
                (1.0 - y) * ((2.0 * mu / alpha - b1 * (1.0 + x / alpha)) * y
                             + (mu / alpha ** 2 - sigma) * x * y)
                - (2.0 * mu + b2 * (1.0 - 2.0 * y)) * x
            )
            return magnitude * (left + right)

        return f

    def _get_panettone_t(self, magnitude: float) -> TwoDFunction:
        magnitude *= 10.0

        def f(x: float, y: float) -> float:
            p = self.params
            mu, b1, b2, sigma = p["mu"], p["b1"], p["b2"], p["sigma"]

            term1 = exp(-y) * (
                y * (b1 + 2.0 * mu - 2.0 * b1 * x
                     + (x - 1.0) * x * (b2 + mu - sigma))
                - (x - 1.0) * x * (b2 + 2.0 * mu)
            )
            term2 = exp(-1.0 + (y - 1.0) / mu) * (
                -x ** 2 * ((b2 - 2.0) * mu + y * (b2 + mu * sigma - 1.0))
                + x * ((b2 - 2.0) * mu + y * (-2.0 * b1 * mu + b2 + mu * sigma - 1.0))
                + mu * sigma * (b1 + 2.0 * mu)
            ) / mu

            return magnitude * (term1 - term2)

        return f

    def _get_panettone_l(self, magnitude: float) -> TwoDFunction:
        def f(x: float, y: float) -> float:
            p = self.params
            mu, b1, b2, sigma = p["mu"], p["b1"], p["b2"], p["sigma"]

            exp_term = exp(x * (1.0 + 1.0 / mu))

            y0 = -mu * (x - 1.0) * (b2 + 2.0 * mu) * (exp_term - 1.0)

            y2_coeff = (
                -b1 * mu - b1 + mu * sigma - 2.0 * mu
                - mu * exp_term * (mu + sigma - x * (b1 - mu + sigma))
                + b1 * x - mu * sigma * x + x - 1.0
            )

            y1_coeff = (
                b1 * mu + b1 + 2.0 * b2 * mu - mu * sigma + 2.0 * mu
                + mu * exp_term * (-2.0 * b2 + mu + sigma - x * (b1 - 2.0 * b2 - mu + sigma))
                - b1 * x - 2.0 * b2 * mu * x + mu * sigma * x + 1.0
            )

            return magnitude * (
                10.0 / mu * exp(-1.0 - x / mu) * (y0 + y ** 2 * y2_coeff + y * y1_coeff)
            )

        return f

    # ------------------------------------------------------------------ #
    # "Freccia" manufactured solutions (mirror images of one another)
    # ------------------------------------------------------------------ #
    def _freccia_bump(self) -> TwoDFunction:
        m = self.params["alpha"]
        c = exp(-1.0 / m) / (1.0 - exp(-1.0 / m))

        def alpha(x: float, y: float) -> float:
            return exp(-((y - x) ** 2) / 0.01)

        def rho(z: float) -> float:
            return z - (exp((z - 1.0) / m) - c)

        def delta(z: float) -> float:
            return 1.0 - exp(-z / m) + exp(-1.0 / m) - exp(-(1.0 - z) / m)

        return lambda x, y: (alpha(x, y) + rho(x) * rho(y)) * delta(x) * delta(y)

    def _apply_convection_diffusion_operator(
        self, uh: TwoDFunction, magnitude: float
    ) -> TwoDFunction:
        h = self._FD_STEP

        def operator(x: float, y: float) -> float:
            p = self.params
            mu, b1, b2, sigma = p["mu"], p["b1"], p["b2"], p["sigma"]

            d2x = (uh(x + h, y) - 2.0 * uh(x, y) + uh(x - h, y)) / h ** 2
            d2y = (uh(x, y + h) - 2.0 * uh(x, y) + uh(x, y - h)) / h ** 2
            dx = (uh(x + h, y) - uh(x - h, y)) / (2.0 * h)
            dy = (uh(x, y + h) - uh(x, y - h)) / (2.0 * h)

            return magnitude * (-mu * (d2x + d2y) + b1 * dx + b2 * dy + sigma * uh(x, y))

        return operator

    def _get_freccia_11(self, magnitude: float) -> TwoDFunction:
        uh = self._freccia_bump()
        return self._apply_convection_diffusion_operator(uh, magnitude)

    def _get_freccia_00(self, magnitude: float) -> TwoDFunction:
        base_uh = self._freccia_bump()
        uh = lambda x, y: base_uh(1.0 - x, 1.0 - y)  # 180-degree rotated bump
        return self._apply_convection_diffusion_operator(uh, magnitude)

    # ------------------------------------------------------------------ #
    # Border functions
    #
    # `border` identifies which edge of the unit-square domain the trace is
    # evaluated on (1-4);
    # ------------------------------------------------------------------ #
    def _get_border_f9(self, magnitude: float, border: int = 1) -> OneDFunction:
        magnitude *= 0.5
        mu = self.params["mu"]

        if border in (1, 3):
            return lambda x: magnitude * (tanh((2.0 * x - 1.0) / mu) + 1.0)
        if border == 2:
            return lambda x: magnitude * (tanh(1.0 / mu) + 1.0)
        return lambda x: magnitude * (tanh(-1.0 / mu) + 1.0)

    def _get_border_f10(self, magnitude: float, border: int = 1) -> OneDFunction:
        mu = self.params["mu"]

        if border in (2, 4):
            return lambda x: magnitude * tanh(2.0 * (1.0 - x) / mu)
        if border == 1:
            return lambda x: magnitude * tanh(2.0 / mu)
        return lambda x: 0.0

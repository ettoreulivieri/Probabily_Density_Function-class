# Copyright (C) 2026 Ettore
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline


class PDF:
    """
    Reconstruct a probability density function from a set of data points.

    The probability density function is obtained by interpolating the input
    data using a spline of order `k`. The interpolated function is clipped
    to prevent negative values and then normalized over the interval
    [x[0], x[-1]].

    Parameters
    ----------
    x : (N,) array_like
        Coordinates of the data points. Values must be strictly increasing.

    y : (N,) array_like
        Function values at the corresponding coordinates. Values must be
        non-negative.

    k : int, default=3
        Degree of the spline polynomial. Must be between 1 and 5, and
        the number of data points must be greater than `k`.

    Notes
    -----
    The input arrays `x` and `y` must have the same length and must be
    one-dimensional.

    The normalization factor is the integral of the clipped interpolating
    spline over the interval [x[0], x[-1]].

    The resulting probability density function has an integral equal to
    one over this interval, provided that the normalization factor is
    positive and finite.
    """

    def __init__(self, x: ArrayLike, y: ArrayLike, k: int = 3):
        """
        Initialize the PDF object.

        Parameters
        ----------
        x : (N,) array_like
            Coordinates of the input data points. Values must be strictly
            increasing.

        y : (N,) array_like
            Function values corresponding to the coordinates in `x`.
            All values must be non-negative.

        k : int, default=3
            Degree of the interpolating spline.

        Raises
        ------
        ValueError
            If any value of `y` is negative.

        Notes
        -----
        The interpolating spline is clipped to zero wherever it becomes
        negative. Its integral over the input domain is then calculated
        and used as the normalization factor.
        """
        self._x = np.array(x, dtype=float)
        self._y = np.array(y, dtype=float)
        self._k = k

        if np.any(self._y < 0):  # checking positivity of y values
            raise ValueError("All y values must be non-negative.")

        self._spl1 = InterpolatedUnivariateSpline(self._x, self._y, k=self._k)

        spl1 = self._spl1
        self._spl2 = lambda val: np.maximum(0, spl1(val))

        area, _ = integrate.quad(self._spl2, self._x[0], self._x[-1])

        spl2 = self._spl2

        def spl(x_in):
            x_arr = np.asarray(x_in, dtype=float)
            y_val = spl2(x_arr) / area
            return np.where((x_arr >= self._x[0]) & (x_arr <= self._x[-1]), y_val, 0.0)

        self._spl = spl

        self._norm = area

    @property
    def normalization(self):
        """
        Return the normalization factor of the probability density function.

        Returns
        -------
        float
            Integral of the clipped interpolating spline over the domain
            [x[0], x[-1]], before normalization.
        """
        return float(self._norm)

    def valuation(self, x_test: ArrayLike) -> NDArray:
        """
        Evaluate the normalized probability density function.

        Parameters
        ----------
        x_test : array_like
            Point or points at which to evaluate the PDF. A scalar or an
            array of values can be provided.

        Returns
        -------
        numpy.ndarray
            Evaluated PDF values. The output is always at least one-dimensional.
            Values outside the domain [x[0], x[-1]] are intended to be zero.
        """
        x_eval = np.atleast_1d(np.asarray(x_test, dtype=float))
        if np.any((x_eval < self._x[0]) | (x_eval > self._x[-1])):
            print("ERROR: values in x_test not in range of x[0] - x[-1] are evaluated 0")
        return np.atleast_1d(self._spl(x_eval))

    def probability(self, a: float, b: float) -> float:
        """
        Calculate probability of being in the interval [a,b]

        Parameters
        -----------
        a: float
            start_point of interval

        b: float
            end_point of interval

        returns
        -----------
        float
            integral of pdf in the interval
        """
        integral_val, _ = integrate.quad(self._spl, a, b)
        return float(integral_val)


def main():
    """Execution block for manual testing."""
    x_data = np.array([0, 1, 2, 3, 4, 5, 6])
    y_data = np.array([0, 1, 2, 3, 2, 1, 0])
    print("hi")
    pdf = PDF(x_data, y_data, k=1)
    print(pdf.valuation(1.5), type(pdf.valuation(1)))
    print(pdf.valuation([-100, 100]))
    print(f"normalization factor = {pdf.normalization}")
    print(f"probability in {x_data[0]}-{x_data[-1]} = {pdf.probability(x_data[0], x_data[-1])}")


if __name__ == "__main__":
    main()

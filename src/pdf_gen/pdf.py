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

"""Module for continuous Probability Density Function (PDF) estimation and sampling."""

from typing import Union
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
    x : ArrayLike
        Coordinates of the data points. Values must be strictly increasing.

    y : ArrayLike
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

    def __init__(self, x: ArrayLike, y: ArrayLike, k: int = 3) -> None:
        """
        Initialize the PDF object.

        Parameters
        ----------
        x : ArrayLike
            Coordinates of the input data points. Values must be
            strictly increasing.

        y : ArrayLike
            Function values corresponding to the coordinates in `x`.
            All values must be non-negative.

        k : int, default=3
            Degree of the interpolating spline.

        Raises
        ------
        ValueError
            If `x` is not strictly increasing, if any value of `y` is negative,
            or if the integral over the domain is non-positive / non-finite.
        """
        self._x = np.array(x, dtype=float)
        self._y = np.array(y, dtype=float)
        self._k = k

        if np.any(np.diff(self._x) <= 0):  # check if x array is strictly increasing.
            raise ValueError("ERROR: x values must be increasing.")
        if np.any(self._y < 0):  # checking positivity of y values
            raise ValueError("All y values must be non-negative.")

        self._spl1 = InterpolatedUnivariateSpline(self._x, self._y, k=self._k)

        spl1 = self._spl1
        self._spl2 = lambda val: np.maximum(0, spl1(val))

        area, _ = integrate.quad(self._spl2, self._x[0], self._x[-1])
        if not np.isfinite(area) or area <= 0:
            raise ValueError("The integral of the PDF must be positive.")
        spl2 = self._spl2

        def spl(x_in):
            x_arr = np.asarray(x_in, dtype=float)
            y_val = spl2(x_arr) / area
            return np.where((x_arr >= self._x[0]) & (x_arr <= self._x[-1]), y_val, 0.0)

        self._spl = spl
        self._norm = area

        n_points = max(10000, int((self._x[-1] - self._x[0]) * 100))
        self._x_cdf = np.linspace(self._x[0], self._x[-1], n_points)  # creating a grid
        pdf_values = self._spl(self._x_cdf)  # evaluating spline on the grid
        self._y_cdf = integrate.cumulative_trapezoid(  # integrating numerically the PDF
            pdf_values,
            self._x_cdf,
            initial=0
        )
        # Ensure that the CDF ends at 1
        self._y_cdf /= self._y_cdf[-1]

        self._rng = np.random.default_rng()  # random numpy generator

    @property
    def normalization(self) -> float:
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
        Evaluate the normalized probability density function at specified points.

        Parameters
        ----------
        x_test : ArrayLike
            Point or array of points at which to evaluate the PDF.

        Returns
        -------
        NDArray
            Evaluated PDF values normalized over the domain [x[0], x[-1]].
            Values outside this domain evaluate to 0.0.
        """
        x_eval = np.atleast_1d(np.asarray(x_test, dtype=float))
        if np.any((x_eval < self._x[0]) | (x_eval > self._x[-1])):
            print("ERROR: values in x_test not in range of x[0] - x[-1] are evaluated 0")
        return np.atleast_1d(self._spl(x_eval))

    def probability(self, a: float, b: float) -> float:
        """
        Calculate the probability over an interval.

        Parameters
        ----------
        a : float
            Lower integration limit.

        b : float
            Upper integration limit.

        Returns
        -------
        float
            Integral of the normalized PDF over the interval [a, b].

        Notes
        -----
        The probability is calculated by numerical integration of the
        normalized PDF between `a` and `b`.

        If the interval extends outside the domain of the PDF, the density
        is zero outside that domain.

        The result represents a probability when a <= b. The method does
        not currently check explicitly whether `a` is greater than `b`.
        """
        integral_val, _ = integrate.quad(self._spl, a, b)
        return float(integral_val)

    def CDF(self, x_top: float) -> float:
        """
        Evaluate the cumulative distribution function (CDF).

        Parameters
        ----------
        x_top : float
            Upper integration limit.

        Returns
        -------
        float
            Cumulative probability from the lower boundary of the PDF domain
            up to `x_top`.

        Notes
        -----
        The CDF is defined as

            F(x) = integral from x_min to x of f(t) dt,

        where x_min is the smallest input coordinate and f is the normalized
        probability density function.

        For values below the domain, the result is zero. For values above
        the domain, the result is one, up to numerical integration errors.
        """
        return self.probability(self._x[0], x_top)

    def CDF_inv(self, p_val: ArrayLike) -> NDArray:
        """
        Evaluate the inverse cumulative distribution function.

        Parameters
        ----------
        p_val : ArrayLike
            Scalar or array-like object containing probabilities between
            zero and one, inclusive.

        Returns
        -------
        NDArray
            One-dimensional NumPy array containing the approximate inverse
            CDF values corresponding to the input probabilities.

        Raises
        ------
        ValueError
            If any probability is smaller than zero or greater than one.

        Notes
        -----
        The inverse CDF is approximated by linear interpolation using the
        numerically computed CDF grid.

        Repeated CDF values are removed before interpolation to handle
        intervals where the PDF is zero.

        The inverse CDF is used by the `sample` method to generate random
        numbers following the reconstructed probability distribution.
        """
        p = np.atleast_1d(np.asarray(p_val, dtype=float))

        if np.any((p < 0) | (p > 1)):
            raise ValueError("Probabilities must be between 0 and 1.")

        y_cdf_unique, indices = np.unique(self._y_cdf, return_index=True)
        x_unique = self._x_cdf[indices]  # masking

        result = np.interp(p, y_cdf_unique, x_unique)

        return result

    def sample(self, size: Union[int, tuple[int, ...]] = 1) -> NDArray:
        """
        Generate random samples distributed according to the PDF.

        Parameters
        ----------
        size : int, default=1
            Number of random samples to generate.

        Returns
        -------
        NDArray
            NumPy array containing the generated random samples.

        Notes
        -----
        Random numbers are generated using inverse transform sampling.

        First, uniform random numbers between zero and one are generated.
        Then, the inverse CDF is applied to transform them into samples
        following the reconstructed probability distribution.

        The random number generator is initialized once when the PDF
        object is created and is reused for subsequent calls.
        """
        u = self._rng.uniform(0.0, 1.0, size=size)
        return np.asarray(self.CDF_inv(u))


def main() -> None:
    """Execution block for manual testing."""
    x_data = np.array([0, 1, 2, 3, 4, 5, 6])
    y_data = np.array([0, 1, 2, 3, 2, 1, 0])
    print("hi")
    pdf = PDF(x_data, y_data, k=1)
    print(pdf.valuation(1.5), type(pdf.valuation(1)))
    print(pdf.valuation([-100, 100]))
    print(f"normalization factor = {pdf.normalization}")
    print(f"probability in {x_data[0]}-{x_data[-1]} = {pdf.probability(x_data[0], x_data[-1])}")
    print("CDF at 3:", pdf.CDF(3))
    print("Inverse CDF at 0.5:", pdf.CDF_inv(0.5))

    samples = pdf.sample(1000)
    print("First 10 samples:", samples[:10])
    print("Minimum sample:", np.min(samples))
    print("Maximum sample:", np.max(samples))


if __name__ == "__main__":
    main()
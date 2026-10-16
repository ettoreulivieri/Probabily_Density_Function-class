import numpy as np

import pdf_gen

x_test = np.array([0, 1, 2, 3, 4, 5, 6])
y_test = np.array([0, 1, 2, 3, 2, 1, 0])
pdf_test = pdf_gen.PDF(x_test, y_test)


def test_valuation():
    assert np.allclose(pdf_test.valuation(x_test), y_test / pdf_test.normalization)
    assert pdf_test.valuation(1.5).shape == (1,)


def test_total_probability():
    assert pdf_test.probability(x_test[0], x_test[-1]) == 1.0


def test_positive_pdf():
    x_check = np.linspace(x_test[0], x_test[-1], 10000)
    y_check = pdf_test.valuation(x_check)
    assert np.all(y_check >= 0)

def test_cdf():
    assert pdf_test.cdf(x_test[-1]) == 1




def test_normalization():
    """The normalization factor must be positive."""
    assert pdf_test.normalization > 0


def test_probability_interval():
    """The probability of a subinterval must be between 0 and 1."""
    prob = pdf_test.probability(1, 4)

    assert 0 <= prob <= 1


def test_probability_additivity():
    """Probabilities of adjacent intervals must add up."""
    p1 = pdf_test.probability(0, 3)
    p2 = pdf_test.probability(3, 6)
    p_total = pdf_test.probability(0, 6)

    assert np.isclose(p1 + p2, p_total)


def test_probability_outside_domain():
    """The PDF must be zero outside its domain."""
    assert pdf_test.probability(-10, 0) == 0.0
    assert pdf_test.probability(6, 10) == 0.0


def test_cdf_boundaries():
    """The cdf must be zero and one at the domain boundaries."""
    assert np.isclose(pdf_test.cdf(0), 0.0)
    assert np.isclose(pdf_test.cdf(6), 1.0)


def test_cdf_monotonicity():
    """The cdf must be non-decreasing."""
    x_check = np.linspace(0, 6, 100)
    cdf_values = np.array([pdf_test.cdf(x) for x in x_check])

    assert np.all(np.diff(cdf_values) >= 0)


def test_cdf_inv_boundaries():
    """The inverse cdf must return the domain boundaries at 0 and 1."""
    assert np.isclose(pdf_test.cdf_inv(0)[0], 0.0)
    assert np.isclose(pdf_test.cdf_inv(1)[0], 6.0)


def test_cdf_inv_array():
    """The inverse cdf must accept an array of probabilities."""
    probabilities = np.array([0.1, 0.5, 0.9])
    x_values = pdf_test.cdf_inv(probabilities)

    assert x_values.shape == probabilities.shape
    assert np.all(np.diff(x_values) >= 0)
    assert np.all((x_values >= 0) & (x_values <= 6))

####
def test_cdf_inv_invalid_probability():
    """The inverse cdf must reject probabilities outside [0, 1]."""
    import pytest

    with pytest.raises(ValueError):
        pdf_test.cdf_inv(-0.1)

    with pytest.raises(ValueError):
        pdf_test.cdf_inv(1.1)


def test_sample_shape():
    """Sampling must return the requested number of samples."""
    samples = pdf_test.sample(100)

    assert samples.shape == (100,)


def test_sample_range():
    """All samples must lie inside the PDF domain."""
    samples = pdf_test.sample(1000)

    assert np.all(samples >= 0)
    assert np.all(samples <= 6)


def test_sample_distribution():
    """The sample mean should be close to the theoretical mean."""
    samples = pdf_test.sample(20000)

    # The density is symmetric around x = 3.
    assert abs(np.mean(samples) - 3.0) < 0.1

####
def test_invalid_x():
    """The constructor must reject non-increasing x values."""
    import pytest

    x_invalid = np.array([0, 1, 2, 2, 4])
    y_invalid = np.array([0, 1, 2, 3, 4])

    with pytest.raises(ValueError):
        pdf_gen.PDF(x_invalid, y_invalid)

#####
def test_negative_y():
    """The constructor must reject negative y values."""
    import pytest

    x_invalid = np.array([0, 1, 2, 3])
    y_invalid = np.array([0, 1, -1, 2])

    with pytest.raises(ValueError):
        pdf_gen.PDF(x_invalid, y_invalid)

#####
def test_zero_density_rejected():
    """The constructor must reject a density with zero total integral."""
    import pytest

    x_invalid = np.array([0, 1, 2, 3])
    y_invalid = np.zeros(4)

    with pytest.raises(ValueError):
        pdf_gen.PDF(x_invalid, y_invalid)

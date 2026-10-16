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

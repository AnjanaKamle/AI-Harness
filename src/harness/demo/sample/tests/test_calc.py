from calc import add, divide, multiply


def test_add():
    assert add(2, 2) == 4


def test_multiply():
    assert multiply(2, 3) == 6


def test_divide():
    assert divide(5, 2) == 2.5

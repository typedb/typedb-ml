
import unittest


class MyTest(unittest.TestCase):
    def test_something(self):
        raise ValueError

    def test_something_else(self):
        raise ValueError


if __name__ == "__main__":
    unittest.main()

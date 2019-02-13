import unittest

from mymath import fib


class TestFibonacci(unittest.TestCase):  # 단위 테스트를 상속
    def test0(self):
        self.assertEqual(0, fib(0))  # 초깃값 fib(0) = 0을 검증

    def test1(self):
        self.assertEqual(1, fib(1))  # 초깃값 fib(1) =1을 검증

    def test2(self):
        self.assertEqual(fib(0) + fib(1), fib(2))  # 규칙대로 나오는지 검증

    def test3(self):
        self.assertEqual(fib(8) + fib(9), fib(10))


unittest.main()

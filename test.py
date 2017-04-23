import unittest

from markov import to_nparray, meanNd, totalNd, shift_arrayNd, sum_and_mean, X1Xn, XXNd, X0X1Nd, XX

class MarkovTest(unittest.TestCase):
    def setUp(self):
        super(MarkovTest, self).setUp()
        self.year_range = 8
        self.X = to_nparray([
            (102, 147),
            (87, 133),
            (120, 140),
            (112, 120),
            (96, 125),
            (100, 167),
            (108, 148),
            (82, 155)
        ])


    def test_mean_Nd(self):
        mean_Xi, mean_Xj = meanNd(self.X)

        self.assertEqual(mean_Xi, 100.8750)
        self.assertEqual(mean_Xj, 141.8750)

    def test_total_Nd(self):
        total_Xi, total_Xj = totalNd(self.X)

        self.assertEqual(total_Xi, 807)
        self.assertEqual(total_Xj, 1135)

    def test_X0_array(self):
        X0 = shift_arrayNd(self.X)
        self.assertListEqual([147, 133, 140, 120, 125, 167, 148], X0.tolist())

    def test_X0_sum_and_mean(self):
        X0 = shift_arrayNd(self.X)
        sum_X0, mean_X0 = sum_and_mean(X0)

        self.assertEqual(140.0, mean_X0)
        self.assertEqual(980.0, sum_X0)

    def test_X1Xn(self):
        product = X1Xn(self.X)
        self.assertListEqual([14994, 11571, 16800, 13440, 12000, 16700, 15984, 12710], product.tolist())

    def test_squares(self):
        squares = list(XXNd(self.X))

        self.assertListEqual([10404, 7569, 14400, 12544, 9216, 10000, 11664, 6724], squares[0].tolist())
        self.assertListEqual([21609, 17689, 19600, 14400, 15625, 27889, 21904, 24025], squares[1].tolist())

    def test_X0X1(self):
        product = X0X1Nd(self.X)

        self.assertListEqual([12789, 15960, 15680, 11520, 12500, 18036, 12136], product.tolist())

    def test_X0X0(self):
        X0 = shift_arrayNd(self.X)
        product = XX(X0)

        self.assertListEqual([21609, 17689, 19600, 14400, 15625, 27889, 21904], product.tolist())

if __name__ == '__main__':
    unittest.main()

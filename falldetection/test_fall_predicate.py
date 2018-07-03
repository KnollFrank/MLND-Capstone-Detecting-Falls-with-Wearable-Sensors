from unittest import TestCase

from falldetection.fall_predicate import isFall


class FallPredicateTestCase(TestCase):

    def test_is_fall1(self):
        fall = isFall('../data/FallDataSet/209/Testler Export/916/Test_1/340535.txt')
        self.assertEquals(fall, True)

    def test_is_fall2(self):
        fall = isFall('../data/FallDataSet/209/Testler Export/813/Test_6/340535.txt')
        self.assertEquals(fall, False)

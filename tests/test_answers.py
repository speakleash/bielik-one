"""The answer shapes of /v1/systemone. Confidence follows the reference System One formulas (Hugging Face Transformers'
decision pipeline); labels, our own type, report the probability that the selected set is exactly right."""

import unittest

from bielik_one.api.server import (
    answer_confidence,
    choice_confidence,
    score_confidence,
    systemone_answer,
)


class ReferenceFormulaTests(unittest.TestCase):
    def test_choice_examples(self):
        self.assertAlmostEqual(choice_confidence([1 / 3] * 3), 0.0)
        self.assertAlmostEqual(choice_confidence([0, 1, 0]), 1.0)
        # Hugging Face blog: p_max 0.9983 over 3 options -> 0.9974
        self.assertAlmostEqual(choice_confidence([0.9983, 0.001, 0.0007]), 0.99745, places=5)
        # OpenRouter example: payments 0.84 of 3 -> about 0.75 (0.76 from the rounded probabilities)
        self.assertAlmostEqual(choice_confidence([0.0, 0.16, 0.84]), 0.76)
        self.assertEqual(choice_confidence([1.0]), 1.0)

    def test_score_example(self):
        # Hugging Face blog: urgency 0.0149 / 0.2648 / 0.7203 -> confidence 0.558
        self.assertAlmostEqual(score_confidence([0.0149, 0.2648, 0.7203]), 0.558, places=3)
        self.assertAlmostEqual(score_confidence([0.0, 0.0, 1.0]), 1.0)
        self.assertAlmostEqual(score_confidence([1 / 3] * 3), 0.0)  # floored at 0 (it would be slightly negative)
        # A near miss costs less than a far one, at the same modal probability.
        self.assertGreater(score_confidence([0.0, 0.4, 0.6]), score_confidence([0.4, 0.0, 0.6]))

    def test_score_on_an_even_scale(self):
        # Four levels: the uniform distance from the middle (1.5) is 1.0.
        self.assertAlmostEqual(score_confidence([0.1, 0.7, 0.1, 0.1]), 1 - (0.1 + 0.1 + 0.2) / 1.0)


class AnswerTests(unittest.TestCase):
    def test_noul_is_a_choice_between_its_sides(self):
        for yes, expected in ((0.9, 0.8), (0.2, 0.6), (0.5, 0.0)):
            answer = systemone_answer("noul", ["no", "yes"], [1 - yes, yes])
            self.assertAlmostEqual(answer["noul"], yes)
            self.assertAlmostEqual(answer["probabilities"]["true"], yes)
            self.assertAlmostEqual(answer["confidence"], expected)
            self.assertAlmostEqual(answer_confidence(answer), expected)

    def test_option_order_follows_keys(self):
        answer = systemone_answer("noul", ["yes", "no"], [0.3, 0.7])
        self.assertAlmostEqual(answer["noul"], 0.3)
        self.assertAlmostEqual(answer["confidence"], 0.4)

    def test_choice_and_score_answers(self):
        choice = systemone_answer("choice", ["a", "b"], [0.25, 0.75])
        self.assertEqual(choice["choice"], "b")
        self.assertAlmostEqual(choice["confidence"], 0.5)
        self.assertAlmostEqual(choice["probabilities"]["b"], 0.75)  # the probability of the answer stays here
        score = systemone_answer("score", ["0", "1", "2"], [0.0149, 0.2648, 0.7203], ["low", "mid", "high"])
        self.assertAlmostEqual(score["score"], 0.2648 + 2 * 0.7203)
        self.assertAlmostEqual(score["confidence"], 0.558, places=3)


class LabelsConfidenceTests(unittest.TestCase):
    def test_confidence_is_the_probability_of_the_exact_set(self):
        labels = systemone_answer("labels", ["a", "b", "c"], [0.9, 0.3, 0.6])
        self.assertEqual(labels["selected"], ["a", "c"])
        self.assertAlmostEqual(labels["confidence"], 0.9 * 0.7 * 0.6)
        self.assertAlmostEqual(answer_confidence(labels), 0.9 * 0.7 * 0.6)

    def test_empty_set_and_threshold(self):
        empty = systemone_answer("labels", ["a", "b"], [0.1, 0.2])
        self.assertEqual(empty["selected"], [])
        self.assertAlmostEqual(empty["confidence"], 0.9 * 0.8)
        # Each option's decision follows the served threshold, not 0.5.
        shifted = systemone_answer("labels", ["a", "b"], [0.4, 0.2], threshold=0.3)
        self.assertEqual(shifted["selected"], ["a"])
        self.assertAlmostEqual(shifted["confidence"], 0.4 * 0.8)


if __name__ == "__main__":
    unittest.main()

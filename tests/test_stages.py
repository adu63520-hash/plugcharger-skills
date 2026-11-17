"""Failure classification: which skill an episode stopped at.

Recording failures is only useful if the stage they failed at is recoverable, so this contract
is tested without needing mplib or a simulator.
"""
import unittest

from charger.stages import boundaries, classify_failure


def labels(*triples):
    return [dict(zip('ABC', t)) for t in triples]


class FailureClassificationTests(unittest.TestCase):
    def test_success_reports_no_failed_stage(self):
        failed, partial = classify_failure(labels((1, 0, 0), (1, 1, 0), (1, 1, 1)))
        self.assertIsNone(failed)
        self.assertEqual(partial, dict(A=[0, 1], B=[1, 2], C=[2, 3]))
        self.assertEqual(boundaries(labels((1, 0, 0), (1, 1, 0), (1, 1, 1))), partial)

    def test_failure_at_a_keeps_no_boundaries(self):
        failed, partial = classify_failure(labels((0, 0, 0), (0, 0, 0)))
        self.assertEqual(failed, 'A')
        self.assertEqual(partial, {})

    def test_failure_at_b_keeps_a(self):
        failed, partial = classify_failure(labels((1, 0, 0), (1, 0, 0)))
        self.assertEqual(failed, 'B')
        self.assertEqual(partial, dict(A=[0, 1]))

    def test_failure_at_c_keeps_a_and_b(self):
        failed, partial = classify_failure(labels((1, 0, 0), (1, 1, 0)))
        self.assertEqual(failed, 'C')
        self.assertEqual(partial, dict(A=[0, 1], B=[1, 2]))

    def test_empty_episode_fails_at_a(self):
        failed, partial = classify_failure([])
        self.assertEqual(failed, 'A')
        self.assertEqual(partial, {})

    def test_partial_boundaries_match_what_boundaries_recovers(self):
        # A skill that already completed may drop later (e.g. releasing on insertion);
        # classification must look at first occurrence only, like `boundaries` does.
        seq = labels((1, 0, 0), (0, 0, 0), (0, 0, 0))
        failed, partial = classify_failure(seq)
        self.assertEqual(failed, 'B')
        self.assertEqual(partial, dict(A=[0, 1]))


if __name__ == '__main__':
    unittest.main()

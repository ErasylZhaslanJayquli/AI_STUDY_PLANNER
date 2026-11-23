"""Unit tests for AI Study Planner (CSP Week 3, MDP Week 5, Classical ML Week 8)."""

import unittest
import numpy as np

from src.csp_scheduler import StudyScheduleCSP, SubjectSpec
from src.data_generator import generate_student_dataset
from src.mdp_optimizer import StudyRestMDP
from src.ml_predictor import ExamScorePredictor


class TestCSPScheduler(unittest.TestCase):
    def setUp(self):
        self.subjects = [
            SubjectSpec(name="AI Fundamentals", exam_day=6, required_sessions=4, difficulty=4, priority=2.1),
            SubjectSpec(name="Calculus II", exam_day=5, required_sessions=4, difficulty=5, priority=2.8),
            SubjectSpec(name="Data Structures", exam_day=8, required_sessions=3, difficulty=4, priority=1.9),
            SubjectSpec(name="Linear Algebra", exam_day=9, required_sessions=3, difficulty=3, priority=1.4),
            SubjectSpec(name="English Academic", exam_day=7, required_sessions=2, difficulty=2, priority=0.9),
        ]

    def test_valid_schedule_satisfies_all_constraints(self):
        blocked = {(1, 0), (2, 1), (3, 3)}
        csp = StudyScheduleCSP(
            subjects=self.subjects,
            total_days=9,
            max_sessions_per_day=3,
            max_same_subject_per_day=2,
            blocked_slots=blocked,
        )
        res = csp.solve(use_mrv=True, use_lcv=True, use_forward_checking=True)
        self.assertTrue(res.solved)
        self.assertEqual(len(res.assignments), 16)
        self.assertTrue(all(res.constraints_audit.values()))

    def test_infeasible_schedule_handled_gracefully(self):
        impossible = [
            SubjectSpec(name="Impossible Exam", exam_day=2, required_sessions=5, difficulty=5),
        ]
        csp = StudyScheduleCSP(subjects=impossible, total_days=5, max_sessions_per_day=3)
        res = csp.solve()
        self.assertFalse(res.solved)
        self.assertEqual(len(res.assignments), 0)
        self.assertTrue(len(res.message) > 0)

    def test_mrv_fc_improves_search_efficiency(self):
        blocked = {(1, 0), (1, 3), (2, 1), (4, 0), (4, 2)}
        csp = StudyScheduleCSP(
            subjects=self.subjects,
            total_days=9,
            max_sessions_per_day=3,
            blocked_slots=blocked,
        )
        naive = csp.solve(use_mrv=False, use_lcv=False, use_forward_checking=False, max_nodes=25000)
        smart = csp.solve(use_mrv=True, use_lcv=True, use_forward_checking=True, max_nodes=25000)
        self.assertTrue(smart.solved)
        self.assertLessEqual(smart.nodes_explored, naive.nodes_explored)

    def test_invalid_csp_inputs_raise_value_error(self):
        with self.assertRaises(ValueError):
            StudyScheduleCSP(subjects=self.subjects, total_days=0)
        with self.assertRaises(ValueError):
            StudyScheduleCSP(subjects=self.subjects, total_days=5, max_sessions_per_day=10)


class TestMDPOptimizer(unittest.TestCase):
    def setUp(self):
        self.mdp = StudyRestMDP(gamma=0.92, burnout_penalty=-22.0, mastery_reward=16.0, subject_difficulty=4)

    def test_transition_probabilities_sum_to_one(self):
        for state in self.mdp.states:
            for action in ["DEEP_STUDY", "LIGHT_REVIEW", "SHORT_BREAK", "FULL_REST"]:
                probs = [t[0] for t in self.mdp.get_transitions(state, action)]
                self.assertAlmostEqual(sum(probs), 1.0, places=6)

    def test_value_and_policy_iteration_converge_to_same_values(self):
        vi = self.mdp.value_iteration(theta=1e-6)
        pi = self.mdp.policy_iteration(theta=1e-6)
        self.assertTrue(np.allclose(vi.values, pi.values, atol=0.1))
        self.assertGreater(vi.iterations, 0)
        self.assertGreater(pi.iterations, 0)

    def test_optimal_policy_outperforms_no_rest_and_random(self):
        vi = self.mdp.value_iteration()
        sim = self.mdp.simulate_policies(vi.policy, start_state=(1, 0), horizon=12, n_episodes=300, seed=42)
        self.assertGreater(
            sim["MDP Optimal"]["avg_discounted_reward"],
            sim["Always Deep Study (Без отдыха)"]["avg_discounted_reward"],
        )
        self.assertGreater(
            sim["MDP Optimal"]["avg_discounted_reward"],
            sim["Random Policy"]["avg_discounted_reward"],
        )
        self.assertIn(vi.policy[(4, 2)], ["SHORT_BREAK", "FULL_REST"])

    def test_invalid_mdp_parameters(self):
        with self.assertRaises(ValueError):
            StudyRestMDP(gamma=1.05)
        with self.assertRaises(ValueError):
            StudyRestMDP(subject_difficulty=9)


class TestMLPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_student_dataset(n_samples=800, seed=42)
        cls.predictor = ExamScorePredictor(random_state=42)
        cls.metrics = cls.predictor.train_and_evaluate(cls.df)

    def test_regression_and_classification_quality(self):
        best_reg = self.metrics["best_regressor"]
        r2 = self.metrics["regression_metrics"][best_reg]["R2"]
        mae = self.metrics["regression_metrics"][best_reg]["MAE"]
        self.assertGreater(r2, 0.85)
        self.assertLess(mae, 4.0)

        best_f1 = max(m["Weighted_F1"] for m in self.metrics["classification_metrics"].values())
        self.assertGreater(best_f1, 0.85)

    def test_prediction_bounds_and_monotonic_study_effect(self):
        base = {
            "past_avg_grade": 74.0,
            "midterm_score": 70.0,
            "attendance_rate": 80.0,
            "assignments_completed": 78.0,
            "study_hours_planned": 6.0,
            "avg_sleep_hours": 7.5,
            "fatigue_index": 3.5,
            "subject_difficulty": 4,
        }
        more_study = dict(base, study_hours_planned=24.0)
        p_low = self.predictor.predict_subject(base, target_score=85.0, days_until_exam=5)
        p_high = self.predictor.predict_subject(more_study, target_score=85.0, days_until_exam=5)

        self.assertTrue(0.0 <= p_low.predicted_score <= 100.0)
        self.assertTrue(0.0 <= p_high.predicted_score <= 100.0)
        self.assertGreater(p_high.predicted_score, p_low.predicted_score)
        self.assertTrue(1 <= p_low.recommended_sessions <= 6)

    def test_input_validation_rejects_bad_ranges(self):
        bad = {
            "past_avg_grade": 150.0,
            "midterm_score": 70.0,
            "attendance_rate": 80.0,
            "assignments_completed": 78.0,
            "study_hours_planned": 10.0,
            "avg_sleep_hours": 7.0,
            "fatigue_index": 4.0,
            "subject_difficulty": 3,
        }
        with self.assertRaises(ValueError):
            self.predictor.predict_subject(bad)


if __name__ == "__main__":
    unittest.main()

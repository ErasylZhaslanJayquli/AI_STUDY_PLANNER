"""Synthetic academic dataset generator for the AI Study Planner ML module."""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_student_dataset(n_samples: int = 800, seed: int = 42, save_path: str | None = None) -> pd.DataFrame:
    """Generate a realistic student exam performance dataset.

    Features:
        past_avg_grade: Historical GPA-equivalent score on a 0-100 scale.
        midterm_score: Current semester midterm score (0-100).
        attendance_rate: Class attendance percentage (40-100).
        assignments_completed: Percentage of completed homework/labs (30-100).
        study_hours_planned: Total focused study hours allocated before exam (2-45).
        avg_sleep_hours: Average sleep hours per night during prep week (4.0-9.5).
        fatigue_index: Self-reported fatigue level from 1 (fresh) to 10 (exhausted).
        subject_difficulty: Course difficulty rating from 1 (easy) to 5 (very hard).

    Targets:
        final_score: Final exam score on a 0-100 scale.
        grade_category: Risk / performance category ("High Risk", "Pass", "Honors").
    """
    rng = np.random.default_rng(seed)

    past_avg_grade = np.clip(rng.normal(73, 13, n_samples), 38, 98)
    midterm_score = np.clip(0.72 * past_avg_grade + rng.normal(18, 8.5, n_samples), 32, 99)
    attendance_rate = np.clip(rng.normal(80, 14, n_samples), 40, 100)
    assignments_completed = np.clip(0.55 * attendance_rate + rng.normal(33, 11, n_samples), 30, 100)
    subject_difficulty = rng.integers(1, 6, size=n_samples)
    study_hours_planned = np.clip(rng.normal(15 + 2.0 * subject_difficulty, 7.5, n_samples), 2, 45)
    avg_sleep_hours = np.clip(rng.normal(6.8, 1.15, n_samples), 4.0, 9.5)
    fatigue_index = np.clip(
        10.0 - 0.85 * avg_sleep_hours + 0.08 * study_hours_planned + rng.normal(0, 1.0, n_samples),
        1.0,
        10.0,
    )

    # Nonlinear realistic relationship with diminishing study returns and burnout penalty
    effective_study = 13.5 * np.log1p(study_hours_planned / 5.0)
    sleep_effect = -1.6 * np.maximum(0.0, 6.5 - avg_sleep_hours) ** 1.5
    burnout_penalty = -1.35 * np.maximum(0.0, fatigue_index - 6.5) ** 1.4
    difficulty_penalty = -2.6 * (subject_difficulty - 3)

    raw_score = (
        0.34 * past_avg_grade
        + 0.36 * midterm_score
        + 0.09 * attendance_rate
        + 0.11 * assignments_completed
        + effective_study
        + sleep_effect
        + burnout_penalty
        + difficulty_penalty
        - 12.5
        + rng.normal(0, 2.9, n_samples)
    )
    final_score = np.round(np.clip(raw_score, 15, 100), 1)

    grade_category = np.where(
        final_score < 60,
        "High Risk",
        np.where(final_score >= 85, "Honors", "Pass"),
    )

    df = pd.DataFrame({
        "past_avg_grade": np.round(past_avg_grade, 1),
        "midterm_score": np.round(midterm_score, 1),
        "attendance_rate": np.round(attendance_rate, 1),
        "assignments_completed": np.round(assignments_completed, 1),
        "study_hours_planned": np.round(study_hours_planned, 1),
        "avg_sleep_hours": np.round(avg_sleep_hours, 2),
        "fatigue_index": np.round(fatigue_index, 1),
        "subject_difficulty": subject_difficulty.astype(int),
        "final_score": final_score,
        "grade_category": grade_category,
    })

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent.parent / "data" / "student_exam_history.csv"
    df = generate_student_dataset(n_samples=800, seed=42, save_path=str(out_file))
    print(f"Saved {len(df)} records to {out_file}")
    print(df["grade_category"].value_counts())

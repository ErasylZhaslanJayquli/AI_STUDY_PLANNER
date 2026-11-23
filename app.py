"""AI Study Planner — Умный планировщик сессии (QAIRU AI Fundamentals Final Project 2026).

Интерактивное веб-приложение, объединяющее 3 темы разных недель силлабуса:
- Week 3: Constraint Satisfaction Problem (CSP) — бесконфликтное расписание подготовки.
- Week 5: Markov Decision Process (MDP) — баланс между учебой и отдыхом с учетом усталости.
- Week 8: Rational Decision-Making & Classical ML — прогноз итогового балла и приоритетов.

Автор: Жаслан Ерасыл Жайыкұлы
"""

from pathlib import Path
import unittest
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.csp_scheduler import SLOTS_PER_DAY, SLOT_DURATION_HOURS, StudyScheduleCSP, SubjectSpec
from src.data_generator import generate_student_dataset
from src.mdp_optimizer import ACTION_LABELS_RU, ACTION_SHORT_RU, FATIGUE_LABELS, MASTERY_LABELS, StudyRestMDP
from src.ml_predictor import ExamScorePredictor, FEATURE_LABELS_RU


st.set_page_config(
    page_title="AI Study Planner — Умный планировщик сессии",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def load_and_train_ml() -> tuple[pd.DataFrame, ExamScorePredictor, dict]:
    data_path = Path(__file__).resolve().parent / "data" / "student_exam_history.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = generate_student_dataset(n_samples=800, seed=42, save_path=str(data_path))
    predictor = ExamScorePredictor(random_state=42)
    metrics = predictor.train_and_evaluate(df)
    return df, predictor, metrics


def get_default_subjects() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Предмет": "Calculus II",
            "День экзамена": 5,
            "Сложность (1-5)": 5,
            "Средний балл ранее": 71.0,
            "Балл Midterm": 64.0,
            "Посещаемость (%)": 80.0,
            "ДЗ (%)": 75.0,
            "Часы подготовки (план)": 10.0,
        },
        {
            "Предмет": "AI Fundamentals",
            "День экзамена": 6,
            "Сложность (1-5)": 4,
            "Средний балл ранее": 82.0,
            "Балл Midterm": 78.0,
            "Посещаемость (%)": 90.0,
            "ДЗ (%)": 88.0,
            "Часы подготовки (план)": 12.0,
        },
        {
            "Предмет": "English Academic",
            "День экзамена": 7,
            "Сложность (1-5)": 2,
            "Средний балл ранее": 86.0,
            "Балл Midterm": 84.0,
            "Посещаемость (%)": 85.0,
            "ДЗ (%)": 90.0,
            "Часы подготовки (план)": 6.0,
        },
        {
            "Предмет": "Data Structures",
            "День экзамена": 8,
            "Сложность (1-5)": 4,
            "Средний балл ранее": 76.0,
            "Балл Midterm": 72.0,
            "Посещаемость (%)": 82.0,
            "ДЗ (%)": 80.0,
            "Часы подготовки (план)": 10.0,
        },
        {
            "Предмет": "Linear Algebra",
            "День экзамена": 9,
            "Сложность (1-5)": 3,
            "Средний балл ранее": 78.0,
            "Балл Midterm": 75.0,
            "Посещаемость (%)": 84.0,
            "ДЗ (%)": 82.0,
            "Часы подготовки (план)": 8.0,
        },
    ])


def main() -> None:
    st.title("🎓 AI Study Planner — Умный планировщик сессии")
    st.caption(
        "QAIRU AI Fundamentals Final Project 2026 | Участник: **Жаслан Ерасыл Жайыкұлы** | "
        "Темы: CSP (Week 3) + MDP (Week 5) + Rational Decision-Making & Classical ML (Week 8)"
    )

    df_hist, predictor, ml_metrics = load_and_train_ml()

    # --- Sidebar Global Controls ---
    with st.sidebar:
        st.header("⚙️ Параметры сессии")
        total_days = st.slider("Горизонт подготовки (дней)", min_value=4, max_value=14, value=9)
        target_score = st.slider("Целевой балл на экзамене", min_value=60, max_value=100, value=85)
        avg_sleep = st.slider("Средний сон (часов/ночь)", min_value=4.0, max_value=9.5, value=7.0, step=0.5)
        fatigue_idx = st.slider("Текущий индекс усталости (1–10)", min_value=1.0, max_value=10.0, value=4.0, step=0.5)

        st.divider()
        st.subheader("Ограничения CSP (Week 3)")
        max_sessions_day = st.slider("Макс. 2-часовых сессий в день", 1, 4, 3)
        max_same_subj_day = st.slider("Макс. сессий одного предмета в день", 1, 3, 2)
        use_mrv = st.checkbox("Эвристика MRV", value=True)
        use_fc = st.checkbox("Forward Checking", value=True)
        use_lcv = st.checkbox("Эвристика LCV", value=False)

        st.divider()
        st.subheader("Параметры MDP (Week 5)")
        gamma = st.slider("Коэффициент дисконтирования (γ)", 0.70, 0.98, 0.92, 0.02)
        burnout_penalty = st.slider("Штраф за выгорание", -40.0, -5.0, -22.0, 1.0)
        mdp_difficulty = st.slider("Сложность профиля для MDP (1–5)", 1, 5, 4)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Настройка предметов",
        "2. ML-прогноз (Week 8)",
        "3. CSP-расписание (Week 3)",
        "4. MDP баланс отдыха (Week 5)",
        "5. Тестирование и метрики",
    ])

    # ==========================
    # TAB 1: INPUT & SUBJECTS
    # ==========================
    with tab1:
        st.subheader("Редактирование экзаменов и текущих показателей")
        st.info(
            "Измените данные по вашим предметам ниже. ML-модель автоматически оценит ожидаемый балл "
            "и рассчитает рациональный бюджет 2-часовых сессий для CSP-планировщика."
        )
        if "subjects_df" not in st.session_state:
            st.session_state.subjects_df = get_default_subjects()

        edited_df = st.data_editor(
            st.session_state.subjects_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "День экзамена": st.column_config.NumberColumn(min_value=2, max_value=15, step=1),
                "Сложность (1-5)": st.column_config.NumberColumn(min_value=1, max_value=5, step=1),
                "Средний балл ранее": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
                "Балл Midterm": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
                "Посещаемость (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
                "ДЗ (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
                "Часы подготовки (план)": st.column_config.NumberColumn(min_value=0.0, max_value=60.0, step=1.0),
            },
        )
        st.session_state.subjects_df = edited_df

        st.subheader("Заблокированные слоты (пары, работа, личные дела)")
        blocked_options = [
            f"День {d} — {SLOTS_PER_DAY[s]}"
            for d in range(1, total_days + 1)
            for s in range(len(SLOTS_PER_DAY))
        ]
        default_blocked = [
            o for o in ["День 1 — Утро (09:00–11:00)", "День 2 — День (13:00–15:00)", "День 3 — Поздний вечер (20:00–22:00)"]
            if o in blocked_options
        ]
        selected_blocked_labels = st.multiselect(
            "Выберите занятые временные слоты:",
            options=blocked_options,
            default=default_blocked,
        )
        blocked_slots = set()
        for lbl in selected_blocked_labels:
            parts = lbl.split(" — ")
            day_num = int(parts[0].replace("День ", "").strip())
            slot_idx = SLOTS_PER_DAY.index(parts[1].strip())
            blocked_slots.add((day_num, slot_idx))

    # Compute ML predictions for all valid rows
    predictions_rows = []
    csp_subjects = []
    validation_errors = []

    for idx, row in edited_df.iterrows():
        try:
            name = str(row["Предмет"]).strip()
            if not name or name.lower() == "nan":
                continue
            exam_day = int(row["День экзамена"])
            diff = int(row["Сложность (1-5)"])
            features = {
                "past_avg_grade": float(row["Средний балл ранее"]),
                "midterm_score": float(row["Балл Midterm"]),
                "attendance_rate": float(row["Посещаемость (%)"]),
                "assignments_completed": float(row["ДЗ (%)"]),
                "study_hours_planned": float(row["Часы подготовки (план)"]),
                "avg_sleep_hours": float(avg_sleep),
                "fatigue_index": float(fatigue_idx),
                "subject_difficulty": diff,
            }
            pred = predictor.predict_subject(features, target_score=float(target_score), days_until_exam=max(1, exam_day - 1))
            predictions_rows.append({
                "Предмет": name,
                "День экзамена": exam_day,
                "Прогноз балла": pred.predicted_score,
                "Категория риска": pred.risk_category,
                "Отставание от цели": pred.score_gap,
                "Приоритет (Expected Utility)": pred.expected_utility_priority,
                "Реком. сессий (по 2ч)": pred.recommended_sessions,
            })
            csp_subjects.append(
                SubjectSpec(
                    name=name,
                    exam_day=exam_day,
                    required_sessions=pred.recommended_sessions,
                    difficulty=diff,
                    priority=pred.expected_utility_priority,
                )
            )
        except Exception as exc:
            validation_errors.append(f"Строка {idx + 1}: {exc}")

    if validation_errors:
        for err in validation_errors:
            st.error(err)

    # ==========================
    # TAB 2: CLASSICAL ML (WEEK 8)
    # ==========================
    with tab2:
        st.subheader("Прогноз итогового балла и рациональная приоритизация (Week 8)")
        if predictions_rows:
            pred_df = pd.DataFrame(predictions_rows).sort_values("Приоритет (Expected Utility)", ascending=False)
            c1, c2, c3 = st.columns(3)
            c1.metric("Лучшая регрессионная модель", ml_metrics["best_regressor"])
            best_r2 = ml_metrics["regression_metrics"][ml_metrics["best_regressor"]]["R2"]
            best_mae = ml_metrics["regression_metrics"][ml_metrics["best_regressor"]]["MAE"]
            c2.metric("Точность регрессии (R² / MAE)", f"{best_r2:.3f} / ±{best_mae:.2f}")
            best_f1 = max(m["Weighted_F1"] for m in ml_metrics["classification_metrics"].values())
            c3.metric("Классификация риска (Weighted F1)", f"{best_f1:.3f}")

            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            st.markdown("#### Интерактивный симулятор «Что если?» (What-If Analysis)")
            chosen_subj = st.selectbox("Выберите предмет для анализа:", pred_df["Предмет"].tolist())
            row_sel = edited_df[edited_df["Предмет"] == chosen_subj].iloc[0]
            col_a, col_b, col_c = st.columns(3)
            sim_hours = col_a.slider("Часы подготовки (симуляция)", 2.0, 40.0, float(row_sel["Часы подготовки (план)"]), 1.0)
            sim_sleep = col_b.slider("Сон в неделю сессии (ч)", 4.0, 9.5, float(avg_sleep), 0.5)
            sim_fatigue = col_c.slider("Усталость (1–10)", 1.0, 10.0, float(fatigue_idx), 0.5)

            sim_pred = predictor.predict_subject(
                {
                    "past_avg_grade": float(row_sel["Средний балл ранее"]),
                    "midterm_score": float(row_sel["Балл Midterm"]),
                    "attendance_rate": float(row_sel["Посещаемость (%)"]),
                    "assignments_completed": float(row_sel["ДЗ (%)"]),
                    "study_hours_planned": sim_hours,
                    "avg_sleep_hours": sim_sleep,
                    "fatigue_index": sim_fatigue,
                    "subject_difficulty": int(row_sel["Сложность (1-5)"]),
                },
                target_score=float(target_score),
                days_until_exam=max(1, int(row_sel["День экзамена"]) - 1),
            )
            st.success(
                f"Прогноз для **{chosen_subj}**: **{sim_pred.predicted_score} / 100** "
                f"(категория: **{sim_pred.risk_category}**, рекомендовано сессий: **{sim_pred.recommended_sessions}**)"
            )

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Сравнение моделей регрессии")
            st.dataframe(pd.DataFrame(ml_metrics["regression_metrics"]).T, use_container_width=True)
            st.markdown("#### Классификация риска (High Risk / Pass / Honors)")
            st.dataframe(pd.DataFrame(ml_metrics["classification_metrics"]).T, use_container_width=True)
        with col_r:
            st.markdown("#### Важность признаков (Random Forest)")
            fi = ml_metrics["feature_importances"]
            sorted_fi = sorted(fi.items(), key=lambda kv: kv[1])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh([FEATURE_LABELS_RU[k] for k, _ in sorted_fi], [v * 100 for _, v in sorted_fi], color="#0EA5E9")
            ax.set_xlabel("Вклад (%)")
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            st.pyplot(fig)
            plt.close(fig)

    # ==========================
    # TAB 3: CSP SCHEDULER (WEEK 3)
    # ==========================
    with tab3:
        st.subheader("Генерация бесконфликтного расписания подготовки (Week 3: CSP)")
        if not csp_subjects:
            st.warning("Добавьте хотя бы один предмет на вкладке 1.")
        else:
            try:
                csp = StudyScheduleCSP(
                    subjects=csp_subjects,
                    total_days=total_days,
                    max_sessions_per_day=max_sessions_day,
                    max_same_subject_per_day=max_same_subj_day,
                    blocked_slots=blocked_slots,
                )
                res = csp.solve(use_mrv=use_mrv, use_lcv=use_lcv, use_forward_checking=use_fc)
                naive_res = csp.solve(use_mrv=False, use_lcv=False, use_forward_checking=False, max_nodes=25000)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Статус", "Решено ✅" if res.solved else "Конфликт ❌")
                m2.metric("Исследовано узлов", f"{res.nodes_explored} (Naive: {naive_res.nodes_explored})")
                m3.metric("Откаты (Backtracks)", str(res.backtracks))
                m4.metric("Время решения", f"{res.runtime_ms:.2f} мс")

                if res.solved:
                    st.success(f"{res.message} Метод: **{res.method}**.")
                    # Build visual timetable matrix
                    grid = {f"День {d}": [""] * len(SLOTS_PER_DAY) for d in range(1, total_days + 1)}
                    for (d, s) in blocked_slots:
                        if 1 <= d <= total_days:
                            grid[f"День {d}"][s] = "⛔ Занято"
                    for subj in csp_subjects:
                        if 1 <= subj.exam_day <= total_days:
                            grid[f"День {subj.exam_day}"][0] = (
                                (grid[f"День {subj.exam_day}"][0] + " | " if grid[f"День {subj.exam_day}"][0] else "")
                                + f"📝 ЭКЗАМЕН: {subj.name}"
                            )
                    for (subj_name, sess_idx), (d, s) in res.assignments.items():
                        grid[f"День {d}"][s] = f"📘 {subj_name} (#{sess_idx + 1})"

                    timetable_df = pd.DataFrame(grid, index=SLOTS_PER_DAY)
                    st.dataframe(timetable_df, use_container_width=True)

                    st.markdown("#### Аудит соблюдения жестких ограничений (Constraints Audit)")
                    audit_cols = st.columns(2)
                    items = list(res.constraints_audit.items())
                    for i, (cname, ok) in enumerate(items):
                        audit_cols[i % 2].write(f"{'✅' if ok else '❌'} {cname}")
                else:
                    st.error(res.message)
            except ValueError as err:
                st.error(f"Ошибка параметров CSP: {err}")

    # ==========================
    # TAB 4: MDP OPTIMIZER (WEEK 5)
    # ==========================
    with tab4:
        st.subheader("Оптимизация баланса учебы и отдыха с учетом усталости (Week 5: MDP)")
        mdp = StudyRestMDP(
            gamma=gamma,
            burnout_penalty=burnout_penalty,
            subject_difficulty=mdp_difficulty,
        )
        vi = mdp.value_iteration(theta=1e-6)
        pi = mdp.policy_iteration(theta=1e-6)
        sim = mdp.simulate_policies(vi.policy, start_state=(1, 0), horizon=12, n_episodes=400, seed=42)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Value Iteration", f"{vi.iterations} итераций ({vi.runtime_ms:.1f} мс)")
        k2.metric("Policy Iteration", f"{pi.iterations} итераций ({pi.runtime_ms:.1f} мс)")
        k3.metric("Совпадение π*(s)", "100% ✅" if vi.policy == pi.policy else "Близко")
        k4.metric("Освоение MDP vs Grind", f"{sim['MDP Optimal']['avg_final_mastery_pct']}% vs {sim['Always Deep Study (Без отдыха)']['avg_final_mastery_pct']}%")

        col_hm, col_sim = st.columns([1.25, 1.0])
        with col_hm:
            st.markdown("#### Оптимальная политика π*(s) и карта ценности V*(s)")
            fig, ax = plt.subplots(figsize=(7.5, 4.3))
            im = ax.imshow(vi.values, cmap="YlGnBu", aspect="auto")
            ax.set_xticks(range(len(MASTERY_LABELS)))
            ax.set_xticklabels(MASTERY_LABELS)
            ax.set_yticks(range(len(FATIGUE_LABELS)))
            ax.set_yticklabels(FATIGUE_LABELS)
            ax.set_xlabel("Уровень освоения материала")
            ax.set_ylabel("Уровень усталости")
            mid = (vi.values.min() + vi.values.max()) / 2.0
            for f in range(mdp.n_fatigue):
                for m in range(mdp.n_mastery):
                    val = vi.values[f, m]
                    ax.text(
                        m, f, f"{ACTION_SHORT_RU[vi.policy[(f, m)]]}\n({val:.1f})",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color="white" if val > mid else "#0F172A",
                    )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close(fig)

        with col_sim:
            st.markdown("#### Симуляция Монте-Карло (400 эпизодов, 12 слотов)")
            sim_df = pd.DataFrame(sim).T.rename(columns={
                "avg_discounted_reward": "Средняя награда",
                "avg_final_mastery_pct": "Итоговое освоение (%)",
                "avg_burnout_slots": "Слоты в выгорании",
            })
            st.dataframe(sim_df, use_container_width=True)

        st.markdown("#### Персональный пошаговый план чередования учебы и отдыха")
        c_sf, c_sm, c_st = st.columns(3)
        start_f = c_sf.selectbox("Начальная усталость", range(5), format_func=lambda x: FATIGUE_LABELS[x], index=1)
        start_m = c_sm.selectbox("Начальное освоение", range(6), format_func=lambda x: MASTERY_LABELS[x], index=0)
        steps_n = c_st.slider("Число слотов планирования", 4, 14, 8)
        plan = mdp.generate_slot_plan(vi.policy, start_fatigue=start_f, start_mastery=start_m, steps=steps_n)
        st.dataframe(pd.DataFrame(plan), use_container_width=True, hide_index=True)

    # ==========================
    # TAB 5: TESTS & METRICS
    # ==========================
    with tab5:
        st.subheader("Автоматическое тестирование модулей (Unit Tests & Validation)")
        st.write("Запустите встроенный набор из 11 модульных тестов для проверки CSP, MDP и Classical ML.")
        if st.button("▶️ Запустить все 11 unit-тестов"):
            buf = io.StringIO()
            loader = unittest.TestLoader()
            suite = loader.discover(str(Path(__file__).resolve().parent / "tests"))
            runner = unittest.TextTestRunner(stream=buf, verbosity=2)
            test_res = runner.run(suite)
            if test_res.wasSuccessful():
                st.success(f"Все тесты успешно пройдены: {test_res.testsRun}/{test_res.testsRun} OK ✅")
            else:
                st.error(f"Обнаружены ошибки: failures={len(test_res.failures)}, errors={len(test_res.errors)}")
            st.code(buf.getvalue(), language="text")


if __name__ == "__main__":
    main()

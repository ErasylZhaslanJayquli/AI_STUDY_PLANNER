# AI Study Planner — Умный планировщик сессии

**QAZAQ AI RESEARCH UNIVERSITY (QAIRU)**  
**Дисциплина:** AI Fundamentals — Final Project 2026 (*From AI Theory to a Working Product*)  
**Участник:** Жаслан Ерасыл Жайыкұлы  

---

## 1. Цель проекта

**AI Study Planner** — интерактивное веб-приложение на базе Python и Streamlit, помогающее студенту автоматически спланировать подготовку к экзаменационной сессии. Система объединяет алгоритмы из **трёх различных недель силлабуса** в единый сквозной конвейер (pipeline):

1. **Неделя 8 — Rational Decision-Making & Classical ML (`src/ml_predictor.py`):**  
   Прогнозирует ожидаемый итоговый балл (регрессия) и категорию риска (*High Risk / Pass / Honors*, классификация) по прошлым оценкам, баллу Midterm, посещаемости, выполнению ДЗ, сну, усталости и сложности предмета. На основе принципа максимальной ожидаемой полезности (Expected Utility) вычисляет приоритет и бюджет 2-часовых учебных сессий.
2. **Неделя 3 — Constraint Satisfaction Problems / CSP (`src/csp_scheduler.py`):**  
   Составляет бесконфликтное расписание подготовки с учетом дедлайнов экзаменов, дневного лимита нагрузки, занятых слотов и ограничений когнитивной перегрузки с помощью **Backtracking Search + MRV + LCV + Forward Checking**.
3. **Неделя 5 — Markov Decision Processes / MDP (`src/mdp_optimizer.py`):**  
   Оптимизирует баланс между интенсивной учебой, легким повторением и отдыхом в стохастической среде с учетом усталости и риска выгорания с помощью алгоритмов **Value Iteration** и **Policy Iteration**.

---

## 2. Архитектура системы

```text
Входные данные студента (предметы, оценки, дедлайны, сон, усталость, занятые слоты)
        │
        ▼
[Week 8: Classical ML + Rational Utility] ──► Прогноз балла, риск, бюджет сессий
        │
        ▼
[Week 3: CSP Solver (BT + MRV + FC)]      ──► Бесконфликтная сетка расписания
        │
        ▼
[Week 5: MDP (Value & Policy Iteration)]  ──► Оптимальная политика учебы и отдыха π*(s)
        │
        ▼
[Streamlit Interactive Dashboard]         ──► Визуализация, What-If анализ, аудит и тесты
```

### Структура репозитория

```text
ai_study_planner/
├── app.py                         # Веб-приложение Streamlit (5 интерактивных вкладок)
├── src/
│   ├── csp_scheduler.py           # Week 3: CSP Backtracking + MRV + LCV + Forward Checking
│   ├── mdp_optimizer.py           # Week 5: MDP Value Iteration, Policy Iteration, Monte Carlo
│   ├── ml_predictor.py            # Week 8: Ridge/Linear/DT/RF (GridSearchCV) + Risk Classifier
│   └── data_generator.py          # Генератор воспроизводимого датасета (800 записей)
├── data/
│   └── student_exam_history.csv   # Учебный датасет (8 признаков + 2 целевые переменные)
├── tests/
│   └── test_algorithms.py         # 11 модульных тестов (unittest / pytest)
├── assets/                        # Графики экспериментов и benchmark_summary.json
├── docs/
│   ├── PROJECT_PROPOSAL.md        # Заявка на утверждение темы (Раздел 13)
│   ├── AI_USAGE_DECLARATION.md    # Декларация использования ИИ (Раздел 10)
│   ├── TEAM_CONTRIBUTION.md       # Отчет о личном вкладе (Раздел 10)
│   └── DEMO_VIDEO_SCRIPT.md       # Сценарий 2–3 мин демо-видео и 10-мин защиты
├── requirements.txt
└── README.md
```

---

## 3. Установка и запуск

Требуется **Python 3.10+**.

```bash
# 1. Клонировать репозиторий или распаковать архив
cd ai_study_planner

# 2. Создать и активировать виртуальное окружение (рекомендуется)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить веб-приложение
streamlit run app.py
```

Приложение откроется в браузере по адресу `http://localhost:8501`.

### Запуск автоматических тестов

```bash
python -m unittest discover -s tests -v
```

---

## 4. Результаты тестирования и метрики

Все **11/11 unit-тестов** пройдены успешно. Основные экспериментальные метрики:

### Week 3 — CSP Scheduler (5 предметов, 17 сессий, 7 занятых слотов)
| Конфигурация солвера | Узлы поиска | Откаты | Время (мс) | Ограничения |
|---|---:|---:|---:|:---:|
| Basic Backtracking | 153 | 0 | 0.30 | 100% |
| BT + MRV | 153 | 0 | 0.40 | 100% |
| **BT + MRV + Forward Checking** | **17** | **0** | **4.57** | **100%** |
| BT + MRV + LCV + Forward Checking | 44 | 27 | 6.95 | 100% |

*Результат:* Включение **MRV + Forward Checking** сокращает число исследуемых узлов в **9.0 раз** (со 153 до 17).

### Week 5 — MDP Balance Optimizer (500 симуляций Монте-Карло, горизонт 12 слотов)
- **Value Iteration:** сходимость за 195 итераций (476.2 мс).
- **Policy Iteration:** сходимость к **идентичной** оптимальной политике $\pi^*(s)$ за 4 итерации (364.3 мс).

| Стратегия | Средняя награда | Итоговое освоение | Слотов в выгорании |
|---|---:|---:|---:|
| **MDP Optimal Policy** | **+98.53** | **96.6%** | **0.00** |
| Random Policy | +27.52 | 74.2% | 1.25 |
| Fixed 2:1 Heuristic | -53.71 | 53.6% | 6.20 |
| Always Deep Study (без отдыха) | -127.49 | 61.8% | 10.74 |

### Week 8 — Classical ML (800 записей, 80/20 stratified split)
| Модель регрессии | MAE | RMSE | Test $R^2$ | 4-Fold CV $R^2$ |
|---|---:|---:|---:|---:|
| **Ridge Regression** | **2.738** | **3.483** | **0.8870** | **0.8763** |
| Linear Regression | 2.742 | 3.483 | 0.8869 | 0.8763 |
| Random Forest (GridSearchCV) | 3.598 | 4.595 | 0.8033 | 0.7892 |
| Decision Tree | 5.275 | 6.527 | 0.6030 | 0.5427 |

- **Классификация риска (High Risk / Pass / Honors):** Logistic Regression — **Accuracy = 86.88%**, **Weighted F1 = 0.8680**; Random Forest Classifier — Accuracy = 81.25%, Weighted F1 = 0.7819.

---

## 5. Источники и лицензия

1. Russell, S., & Norvig, P. *Artificial Intelligence: A Modern Approach* (4th ed., Pearson, 2020) — главы по CSP, MDP и Machine Learning.
2. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018).
3. Scikit-learn Documentation (`https://scikit-learn.org/`).
4. Streamlit Documentation (`https://docs.streamlit.io/`).

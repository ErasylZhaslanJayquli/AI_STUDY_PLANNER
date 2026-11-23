"""Week 3: Constraint Satisfaction Problem (CSP) Exam Study Scheduler.

Implements Backtracking Search with MRV (Minimum Remaining Values),
LCV (Least Constraining Value), and Forward Checking to build a
conflict-free exam preparation schedule respecting deadlines,
daily hour limits, blocked slots, and subject variety constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Set, Tuple


SLOTS_PER_DAY = ["Утро (09:00–11:00)", "День (13:00–15:00)", "Вечер (17:00–19:00)", "Поздний вечер (20:00–22:00)"]
SLOT_DURATION_HOURS = 2


@dataclass(frozen=True)
class SubjectSpec:
    """Subject requirements for CSP scheduling."""
    name: str
    exam_day: int               # 1-indexed day of the exam (sessions must be strictly < exam_day)
    required_sessions: int      # Number of 2-hour study blocks needed
    difficulty: int = 3         # 1 (easy) to 5 (very hard)
    priority: float = 1.0       # Derived from ML risk/utility analysis


@dataclass
class ScheduleResult:
    """Output container with schedule, constraint audit, and solver diagnostics."""
    solved: bool
    assignments: Dict[Tuple[str, int], Tuple[int, int]]  # (subject_name, session_idx) -> (day, slot_idx)
    nodes_explored: int
    backtracks: int
    runtime_ms: float
    method: str
    constraints_audit: Dict[str, bool] = field(default_factory=dict)
    message: str = ""


class StudyScheduleCSP:
    """CSP formulation and solver for exam preparation timetabling."""

    def __init__(
        self,
        subjects: List[SubjectSpec],
        total_days: int = 10,
        max_sessions_per_day: int = 3,
        max_same_subject_per_day: int = 2,
        blocked_slots: Optional[Set[Tuple[int, int]]] = None,
        slots_count: int = 4,
    ) -> None:
        if total_days < 1:
            raise ValueError("Количество дней должно быть не меньше 1.")
        if max_sessions_per_day < 1 or max_sessions_per_day > slots_count:
            raise ValueError(f"Максимум сессий в день должен быть от 1 до {slots_count}.")
        if max_same_subject_per_day < 1:
            raise ValueError("Лимит сессий одного предмета в день должен быть >= 1.")

        self.subjects = subjects
        self.subject_map = {s.name: s for s in subjects}
        self.total_days = total_days
        self.max_sessions_per_day = max_sessions_per_day
        self.max_same_subject_per_day = max_same_subject_per_day
        self.blocked_slots = blocked_slots or set()
        self.slots_count = slots_count

        # Variables: (subject_name, session_index)
        self.variables: List[Tuple[str, int]] = []
        for subj in subjects:
            if subj.required_sessions < 0:
                raise ValueError(f"Отрицательное число сессий для предмета {subj.name}.")
            for k in range(subj.required_sessions):
                self.variables.append((subj.name, k))

        # Initial Domains: (day, slot_idx) pairs before deadline and not blocked
        self.initial_domains: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
        for var in self.variables:
            subj = self.subject_map[var[0]]
            last_prep_day = min(self.total_days, subj.exam_day - 1)
            domain = [
                (day, slot)
                for day in range(1, last_prep_day + 1)
                for slot in range(self.slots_count)
                if (day, slot) not in self.blocked_slots
            ]
            self.initial_domains[var] = domain

    def is_consistent(
        self,
        var: Tuple[str, int],
        value: Tuple[int, int],
        assignment: Dict[Tuple[str, int], Tuple[int, int]],
    ) -> bool:
        """Check all hard constraints for assigning `value=(day, slot)` to `var=(subject, idx)`."""
        subj_name, sess_idx = var
        day, slot = value
        subj = self.subject_map[subj_name]

        # 1. Deadline constraint
        if day >= subj.exam_day or day < 1 or day > self.total_days:
            return False

        # 2. Blocked slot constraint
        if (day, slot) in self.blocked_slots:
            return False

        # 3. Symmetry breaking for identical sessions of the same subject
        if sess_idx > 0 and (subj_name, sess_idx - 1) in assignment:
            if value <= assignment[(subj_name, sess_idx - 1)]:
                return False
        if (subj_name, sess_idx + 1) in assignment:
            if value >= assignment[(subj_name, sess_idx + 1)]:
                return False

        day_total = 0
        same_subject_on_day = 0
        hard_subjects_on_day = 0

        for other_var, other_val in assignment.items():
            # 4. No overlap constraint (All-Different)
            if other_val == value:
                return False
            o_day, o_slot = other_val
            if o_day == day:
                day_total += 1
                if other_var[0] == subj_name:
                    same_subject_on_day += 1
                    if subj.difficulty >= 4 and abs(o_slot - slot) == 1:
                        return False
                if self.subject_map[other_var[0]].difficulty >= 4:
                    hard_subjects_on_day += 1

        # 5. Daily max load constraint
        if day_total + 1 > self.max_sessions_per_day:
            return False

        # 6. Max sessions of the same subject per day
        if same_subject_on_day + 1 > self.max_same_subject_per_day:
            return False

        # 7. Cognitive overload constraint: max 2 hard-subject sessions (difficulty >= 4) per day
        if subj.difficulty >= 4 and hard_subjects_on_day + 1 > 2:
            return False

        return True

    def _select_unassigned_variable(
        self,
        assignment: Dict[Tuple[str, int], Tuple[int, int]],
        domains: Dict[Tuple[str, int], List[Tuple[int, int]]],
        use_mrv: bool,
    ) -> Tuple[str, int]:
        unassigned = [v for v in self.variables if v not in assignment]
        if not use_mrv:
            return unassigned[0]

        return min(
            unassigned,
            key=lambda v: (
                len(domains[v]),
                self.subject_map[v[0]].exam_day,
                -self.subject_map[v[0]].difficulty,
                v[1],
            ),
        )

    def _order_domain_values(
        self,
        var: Tuple[str, int],
        assignment: Dict[Tuple[str, int], Tuple[int, int]],
        domains: Dict[Tuple[str, int], List[Tuple[int, int]]],
        use_lcv: bool,
    ) -> List[Tuple[int, int]]:
        values = domains[var]
        if not use_lcv:
            return values

        day_counts: Dict[int, int] = {}
        for _, (d, _) in assignment.items():
            day_counts[d] = day_counts.get(d, 0) + 1

        unassigned = [v for v in self.variables if v not in assignment and v != var]

        def score(val: Tuple[int, int]) -> Tuple[int, int, int]:
            d, s = val
            conflicts = sum(1 for u in unassigned if val in domains[u])
            return (conflicts, day_counts.get(d, 0), s)

        return sorted(values, key=score)

    def _forward_check(
        self,
        var: Tuple[str, int],
        value: Tuple[int, int],
        assignment: Dict[Tuple[str, int], Tuple[int, int]],
        domains: Dict[Tuple[str, int], List[Tuple[int, int]]],
    ) -> Optional[Dict[Tuple[str, int], List[Tuple[int, int]]]]:
        """Prune inconsistent values from unassigned variables' domains."""
        new_domains: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
        for u in self.variables:
            if u in assignment:
                continue
            filtered = [val for val in domains[u] if self.is_consistent(u, val, assignment)]
            if not filtered:
                return None
            new_domains[u] = filtered
        return new_domains

    def verify_solution(self, assignment: Dict[Tuple[str, int], Tuple[int, int]]) -> Dict[str, bool]:
        """Independently verify every constraint on the resulting assignment."""
        all_assigned = len(assignment) == len(self.variables)
        no_overlap = len(set(assignment.values())) == len(assignment)
        deadlines_ok = all(
            day < self.subject_map[subj].exam_day and 1 <= day <= self.total_days
            for (subj, _), (day, _) in assignment.items()
        )
        blocked_ok = all(val not in self.blocked_slots for val in assignment.values())

        day_counts: Dict[int, int] = {}
        subj_day_counts: Dict[Tuple[str, int], int] = {}
        hard_day_counts: Dict[int, int] = {}
        no_consecutive_hard = True

        slots_by_subj_day: Dict[Tuple[str, int], List[int]] = {}
        for (subj, _), (day, slot) in assignment.items():
            day_counts[day] = day_counts.get(day, 0) + 1
            subj_day_counts[(subj, day)] = subj_day_counts.get((subj, day), 0) + 1
            if self.subject_map[subj].difficulty >= 4:
                hard_day_counts[day] = hard_day_counts.get(day, 0) + 1
                slots_by_subj_day.setdefault((subj, day), []).append(slot)

        for slots in slots_by_subj_day.values():
            slots_sorted = sorted(slots)
            for i in range(len(slots_sorted) - 1):
                if slots_sorted[i + 1] - slots_sorted[i] == 1:
                    no_consecutive_hard = False

        daily_limit_ok = all(c <= self.max_sessions_per_day for c in day_counts.values())
        same_subj_ok = all(c <= self.max_same_subject_per_day for c in subj_day_counts.values())
        hard_limit_ok = all(c <= 2 for c in hard_day_counts.values())

        return {
            "Все сессии распределены": all_assigned,
            "Нет пересечений слотов": no_overlap,
            "Все дедлайны экзаменов соблюдены": deadlines_ok,
            "Заблокированные слоты не заняты": blocked_ok,
            "Соблюден дневной лимит нагрузки": daily_limit_ok,
            "Соблюден лимит повторов предмета в день": same_subj_ok,
            "Нет когнитивной перегрузки (сложные предметы)": hard_limit_ok and no_consecutive_hard,
        }

    def solve(
        self,
        use_mrv: bool = True,
        use_lcv: bool = True,
        use_forward_checking: bool = True,
        max_nodes: int = 50000,
    ) -> ScheduleResult:
        """Run Backtracking CSP solver with configurable heuristics."""
        method_parts = ["Backtracking"]
        if use_mrv:
            method_parts.append("MRV")
        if use_lcv:
            method_parts.append("LCV")
        if use_forward_checking:
            method_parts.append("Forward Checking")
        method_name = " + ".join(method_parts)

        start = time.perf_counter()

        if not self.variables:
            return ScheduleResult(
                solved=True,
                assignments={},
                nodes_explored=0,
                backtracks=0,
                runtime_ms=0.0,
                method=method_name,
                constraints_audit=self.verify_solution({}),
                message="Нет запланированных сессий.",
            )

        for var, dom in self.initial_domains.items():
            if not dom:
                return ScheduleResult(
                    solved=False,
                    assignments={},
                    nodes_explored=0,
                    backtracks=0,
                    runtime_ms=(time.perf_counter() - start) * 1000,
                    method=method_name,
                    constraints_audit={},
                    message=f"Невозможно составить расписание: нет доступных слотов до экзамена по предмету «{var[0]}».",
                )

        stats = {"nodes": 0, "backtracks": 0, "aborted": False}

        def backtrack(
            assignment: Dict[Tuple[str, int], Tuple[int, int]],
            domains: Dict[Tuple[str, int], List[Tuple[int, int]]],
        ) -> Optional[Dict[Tuple[str, int], Tuple[int, int]]]:
            if len(assignment) == len(self.variables):
                return assignment

            if stats["nodes"] >= max_nodes:
                stats["aborted"] = True
                return None

            var = self._select_unassigned_variable(assignment, domains, use_mrv)
            for value in self._order_domain_values(var, assignment, domains, use_lcv):
                stats["nodes"] += 1
                if self.is_consistent(var, value, assignment):
                    assignment[var] = value
                    if use_forward_checking:
                        next_domains = self._forward_check(var, value, assignment, domains)
                        if next_domains is not None:
                            result = backtrack(assignment, next_domains)
                            if result is not None:
                                return result
                    else:
                        result = backtrack(assignment, domains)
                        if result is not None:
                            return result
                    del assignment[var]
                    stats["backtracks"] += 1
            return None

        domains_copy = {k: list(v) for k, v in self.initial_domains.items()}
        solution = backtrack({}, domains_copy)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if solution is not None:
            audit = self.verify_solution(solution)
            return ScheduleResult(
                solved=True,
                assignments=solution,
                nodes_explored=stats["nodes"],
                backtracks=stats["backtracks"],
                runtime_ms=round(elapsed_ms, 3),
                method=method_name,
                constraints_audit=audit,
                message="Расписание успешно составлено без конфликтов.",
            )

        msg = (
            "Превышен лимит узлов поиска (попробуйте включить MRV + Forward Checking или ослабить ограничения)."
            if stats["aborted"]
            else "Конфликт ограничений: с текущими дедлайнами и лимитами расписание невозможно. Увеличьте дневной лимит или уберите заблокированные слоты."
        )
        return ScheduleResult(
            solved=False,
            assignments={},
            nodes_explored=stats["nodes"],
            backtracks=stats["backtracks"],
            runtime_ms=round(elapsed_ms, 3),
            method=method_name,
            constraints_audit={},
            message=msg,
        )

"""Week 5: Markov Decision Process (MDP) for Study vs. Rest Balance."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Tuple
import numpy as np


FATIGUE_LABELS = ["0: Свежий", "1: Бодрый", "2: Умеренная", "3: Сильная", "4: Выгорание"]
MASTERY_LABELS = ["0%", "20%", "40%", "60%", "80%", "100%"]

ACTIONS = ["DEEP_STUDY", "LIGHT_REVIEW", "SHORT_BREAK", "FULL_REST"]
ACTION_LABELS_RU = {
    "DEEP_STUDY": "Глубокая учеба",
    "LIGHT_REVIEW": "Легкое повторение",
    "SHORT_BREAK": "Короткий перерыв",
    "FULL_REST": "Полноценный отдых/сон",
}
ACTION_SHORT_RU = {
    "DEEP_STUDY": "Глуб. учеба",
    "LIGHT_REVIEW": "Повторение",
    "SHORT_BREAK": "Перерыв",
    "FULL_REST": "Полн. отдых",
}


@dataclass
class MDPResult:
    """Result of solving the Study-Rest MDP."""
    values: np.ndarray            # shape (5, 6): fatigue x mastery
    policy: Dict[Tuple[int, int], str]
    iterations: int
    runtime_ms: float
    method: str
    gamma: float
    theta: float


class StudyRestMDP:
    """Finite-state Markov Decision Process for balancing exam study and recovery."""

    def __init__(
        self,
        gamma: float = 0.92,
        burnout_penalty: float = -22.0,
        mastery_reward: float = 16.0,
        goal_bonus: float = 35.0,
        subject_difficulty: int = 3,
    ) -> None:
        if not (0.1 <= gamma < 1.0):
            raise ValueError("Коэффициент дисконтирования gamma должен быть в интервале [0.1, 1.0).")
        if not (1 <= subject_difficulty <= 5):
            raise ValueError("Сложность предмета должна быть от 1 до 5.")

        self.n_fatigue = 5   # 0..4
        self.n_mastery = 6   # 0..5 (5 = 100% mastered)
        self.gamma = gamma
        self.burnout_penalty = burnout_penalty
        self.mastery_reward = mastery_reward
        self.goal_bonus = goal_bonus
        self.subject_difficulty = subject_difficulty
        self.states: List[Tuple[int, int]] = [
            (f, m) for f in range(self.n_fatigue) for m in range(self.n_mastery)
        ]

    def get_transitions(self, state: Tuple[int, int], action: str) -> List[Tuple[float, Tuple[int, int], float]]:
        """Return list of (probability, next_state, immediate_reward) tuples."""
        f, m = state
        if action not in ACTIONS:
            raise ValueError(f"Неизвестное действие: {action}")

        diff_shift = (3 - self.subject_difficulty) * 0.04
        outcomes: Dict[Tuple[int, int], float] = {}

        def add_prob(nf: int, nm: int, p: float) -> None:
            nf_c = max(0, min(self.n_fatigue - 1, nf))
            nm_c = max(0, min(self.n_mastery - 1, nm))
            outcomes[(nf_c, nm_c)] = outcomes.get((nf_c, nm_c), 0.0) + p

        if action == "DEEP_STUDY":
            if f <= 1:
                p_jump2 = np.clip(0.35 + diff_shift, 0.15, 0.55)
                p_jump1 = np.clip(0.55, 0.35, 0.70)
                p_stay = max(0.0, 1.0 - p_jump2 - p_jump1)
                add_prob(f + 2, m + 2, p_jump2 * 0.65)
                add_prob(f + 1, m + 2, p_jump2 * 0.35)
                add_prob(f + 1, m + 1, p_jump1 * 0.70)
                add_prob(f + 2, m + 1, p_jump1 * 0.30)
                add_prob(f + 1, m, p_stay)
            elif f == 2:
                p_gain = np.clip(0.55 + diff_shift, 0.35, 0.75)
                p_stay = 1.0 - p_gain
                add_prob(f + 1, m + 1, p_gain * 0.55)
                add_prob(f + 2, m + 1, p_gain * 0.45)
                add_prob(f + 2, m, p_stay * 0.70)
                add_prob(f + 1, m, p_stay * 0.30)
            else:
                p_gain = np.clip(0.20 + diff_shift, 0.08, 0.30)
                add_prob(4, m + 1, p_gain)
                add_prob(4, m, 1.0 - p_gain)

        elif action == "LIGHT_REVIEW":
            if f <= 2:
                p_gain = np.clip(0.62 + diff_shift, 0.40, 0.80)
                p_stay = 1.0 - p_gain
                add_prob(f + 1, m + 1, p_gain * 0.60)
                add_prob(f, m + 1, p_gain * 0.40)
                add_prob(f + 1, m, p_stay * 0.50)
                add_prob(f, m, p_stay * 0.50)
            else:
                p_gain = np.clip(0.25 + diff_shift, 0.12, 0.35)
                add_prob(min(4, f + 1), m + 1, p_gain)
                add_prob(min(4, f + 1), m, 1.0 - p_gain)

        elif action == "SHORT_BREAK":
            add_prob(f - 1, m, 0.60)
            add_prob(f - 2, m, 0.32)
            add_prob(f - 1, m - 1 if m > 1 else m, 0.08)

        elif action == "FULL_REST":
            add_prob(0, m, 0.68)
            add_prob(max(0, f - 2), m, 0.17)
            add_prob(0, m - 1 if m > 1 else m, 0.15)

        transitions: List[Tuple[float, Tuple[int, int], float]] = []
        effort_cost = {
            "DEEP_STUDY": -2.5,
            "LIGHT_REVIEW": -1.0,
            "SHORT_BREAK": -0.3,
            "FULL_REST": -0.6,
        }[action]

        for (nf, nm), prob in outcomes.items():
            if prob <= 1e-12:
                continue
            reward = effort_cost
            delta_m = nm - m
            if delta_m > 0:
                reward += delta_m * self.mastery_reward
            elif delta_m < 0:
                reward += delta_m * (self.mastery_reward * 0.6)

            if nm == self.n_mastery - 1 and nf < 4:
                reward += self.goal_bonus * (1.0 if m < self.n_mastery - 1 else 0.25)

            if nf == self.n_fatigue - 1:
                reward += self.burnout_penalty

            if f == self.n_fatigue - 1 and nf < self.n_fatigue - 1:
                reward += 4.0

            transitions.append((prob, (nf, nm), reward))

        total_p = sum(t[0] for t in transitions)
        transitions = [(p / total_p, ns, r) for p, ns, r in transitions]
        return transitions

    def _q_value(self, state: Tuple[int, int], action: str, V: np.ndarray) -> float:
        return sum(
            prob * (reward + self.gamma * V[ns[0], ns[1]])
            for prob, ns, reward in self.get_transitions(state, action)
        )

    def value_iteration(self, theta: float = 1e-6, max_iter: int = 1000) -> MDPResult:
        """Compute optimal value function V* and policy pi* via Value Iteration."""
        start = time.perf_counter()
        V = np.zeros((self.n_fatigue, self.n_mastery), dtype=float)
        iterations = 0

        for it in range(1, max_iter + 1):
            iterations = it
            delta = 0.0
            new_V = np.copy(V)
            for state in self.states:
                best_q = max(self._q_value(state, a, V) for a in ACTIONS)
                new_V[state[0], state[1]] = best_q
                delta = max(delta, abs(best_q - V[state[0], state[1]]))
            V = new_V
            if delta < theta:
                break

        policy: Dict[Tuple[int, int], str] = {}
        for state in self.states:
            policy[state] = max(ACTIONS, key=lambda a: self._q_value(state, a, V))

        elapsed_ms = (time.perf_counter() - start) * 1000
        return MDPResult(
            values=np.round(V, 2),
            policy=policy,
            iterations=iterations,
            runtime_ms=round(elapsed_ms, 3),
            method="Value Iteration",
            gamma=self.gamma,
            theta=theta,
        )

    def policy_iteration(self, theta: float = 1e-6, max_iter: int = 100) -> MDPResult:
        """Compute optimal value function V* and policy pi* via Policy Iteration."""
        start = time.perf_counter()
        V = np.zeros((self.n_fatigue, self.n_mastery), dtype=float)
        policy: Dict[Tuple[int, int], str] = {s: "LIGHT_REVIEW" for s in self.states}
        iterations = 0

        for it in range(1, max_iter + 1):
            iterations = it
            while True:
                delta = 0.0
                for state in self.states:
                    v_old = V[state[0], state[1]]
                    v_new = self._q_value(state, policy[state], V)
                    V[state[0], state[1]] = v_new
                    delta = max(delta, abs(v_old - v_new))
                if delta < theta:
                    break

            policy_stable = True
            for state in self.states:
                old_action = policy[state]
                best_action = max(ACTIONS, key=lambda a: self._q_value(state, a, V))
                policy[state] = best_action
                if best_action != old_action:
                    policy_stable = False
            if policy_stable:
                break

        elapsed_ms = (time.perf_counter() - start) * 1000
        return MDPResult(
            values=np.round(V, 2),
            policy=policy,
            iterations=iterations,
            runtime_ms=round(elapsed_ms, 3),
            method="Policy Iteration",
            gamma=self.gamma,
            theta=theta,
        )

    def simulate_policies(
        self,
        optimal_policy: Dict[Tuple[int, int], str],
        start_state: Tuple[int, int] = (1, 0),
        horizon: int = 12,
        n_episodes: int = 400,
        seed: int = 42,
    ) -> Dict[str, Dict[str, float]]:
        """Compare MDP optimal policy against baseline study strategies via Monte Carlo simulation."""
        rng = np.random.default_rng(seed)

        def choose_action(strategy: str, state: Tuple[int, int], step: int) -> str:
            if strategy == "MDP Optimal":
                return optimal_policy[state]
            if strategy == "Always Deep Study (Без отдыха)":
                return "DEEP_STUDY"
            if strategy == "Fixed 2:1 Heuristic":
                return "SHORT_BREAK" if (step + 1) % 3 == 0 else "DEEP_STUDY"
            if strategy == "Random Policy":
                return str(rng.choice(ACTIONS))
            raise ValueError(strategy)

        strategies = [
            "MDP Optimal",
            "Fixed 2:1 Heuristic",
            "Always Deep Study (Без отдыха)",
            "Random Policy",
        ]
        summary: Dict[str, Dict[str, float]] = {}

        for strat in strategies:
            total_rewards = []
            final_masteries = []
            burnout_steps_list = []

            for _ in range(n_episodes):
                state = start_state
                ep_reward = 0.0
                burnouts = 0
                for step in range(horizon):
                    action = choose_action(strat, state, step)
                    transitions = self.get_transitions(state, action)
                    probs = [t[0] for t in transitions]
                    idx = int(rng.choice(len(transitions), p=probs))
                    _, next_state, reward = transitions[idx]
                    ep_reward += (self.gamma ** step) * reward
                    if next_state[0] == self.n_fatigue - 1:
                        burnouts += 1
                    state = next_state

                total_rewards.append(ep_reward)
                final_masteries.append(state[1] * 20.0)
                burnout_steps_list.append(burnouts)

            summary[strat] = {
                "avg_discounted_reward": round(float(np.mean(total_rewards)), 2),
                "avg_final_mastery_pct": round(float(np.mean(final_masteries)), 1),
                "avg_burnout_slots": round(float(np.mean(burnout_steps_list)), 2),
            }
        return summary

    def generate_slot_plan(
        self,
        optimal_policy: Dict[Tuple[int, int], str],
        start_fatigue: int = 1,
        start_mastery: int = 1,
        steps: int = 8,
    ) -> List[Dict[str, object]]:
        """Generate an expected step-by-step study/rest recommendation trace."""
        state = (
            max(0, min(self.n_fatigue - 1, start_fatigue)),
            max(0, min(self.n_mastery - 1, start_mastery)),
        )
        plan = []
        for step in range(1, steps + 1):
            action = optimal_policy[state]
            transitions = self.get_transitions(state, action)
            best_prob, next_state, _ = max(transitions, key=lambda t: t[0])
            exp_reward = sum(p * r for p, _, r in transitions)
            plan.append({
                "Шаг": step,
                "Усталость": FATIGUE_LABELS[state[0]],
                "Освоение": MASTERY_LABELS[state[1]],
                "Рекомендация MDP": ACTION_LABELS_RU[action],
                "Код действия": action,
                "Ожидаемая награда": round(exp_reward, 2),
                "Прогноз следующего состояния": f"{FATIGUE_LABELS[next_state[0]]} / {MASTERY_LABELS[next_state[1]]}",
            })
            state = next_state
        return plan

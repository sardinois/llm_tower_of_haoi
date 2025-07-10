import random
from typing import Any, List, Optional, Dict
from pydantic import BaseModel
from dataclasses import dataclass, field
from llm_reasoning.models import HistoricalEvaluationModel


class StateModel(BaseModel):
    """Base class for task-specific state models."""

    pass


class ActionModel(BaseModel):
    """Base class for task-specific action models."""

    pass


@dataclass
class GenericState:
    state_data: Dict[str, Any]
    moves_made: List[Any] = field(default_factory=list)
    parent: Optional["GenericState"] = None
    depth: int = 0
    evaluation_history: List[HistoricalEvaluationModel] = field(default_factory=list)

    def is_goal(self) -> bool:
        raise NotImplementedError

    def get_valid_actions(self) -> List[Any]:
        raise NotImplementedError

    def apply_action(self, action: Any) -> "GenericState":
        raise NotImplementedError

    def to_structured_input(self) -> StateModel:
        raise NotImplementedError

    def add_evaluation(self, score: float, reasoning: str, depth: int):
        """Add an evaluation to this state's history"""
        eval_record = HistoricalEvaluationModel(
            state_description=(
                str(self.state_data)[:100] + "..."
                if len(str(self.state_data)) > 100
                else str(self.state_data)
            ),
            score=score,
            reasoning=reasoning,
            depth=depth,
        )
        # Create new list to avoid shared references
        self.evaluation_history = list(self.evaluation_history) + [eval_record]


def _limit_evaluation_history(
    evaluation_history: List[HistoricalEvaluationModel], max_size: int = 50
) -> List[HistoricalEvaluationModel]:
    """
    Limit evaluation history to prevent context explosion.

    New Strategy for > 10 states:
    - 3 highest scoring states (show what good looks like)
    - 3 lowest scoring states (show what bad looks like)
    - 4 random other states (show variety)
    - Preserve temporal order within each group
    """
    if len(evaluation_history) <= 10:
        return evaluation_history

    # Sort by score to identify top and bottom states
    sorted_by_score = sorted(evaluation_history, key=lambda x: x.score, reverse=True)

    # Get top 3 and bottom 3 states
    top_3 = sorted_by_score[:3]
    bottom_3 = sorted_by_score[-3:]

    # Get the middle states (excluding top 3 and bottom 3)
    middle_states = sorted_by_score[3:-3]

    # Randomly sample 4 from middle states, or take all if fewer than 4
    if len(middle_states) >= 4:
        import random

        random_4 = random.sample(middle_states, 4)
    else:
        random_4 = middle_states

    # Combine all selected states
    selected_states = top_3 + bottom_3 + random_4

    # Sort by original temporal order (preserve chronology within the selection)
    original_indices = {id(state): i for i, state in enumerate(evaluation_history)}
    selected_states.sort(key=lambda x: original_indices[id(x)])

    return selected_states


class GenericLLMGuidedSolver:
    def __init__(
        self,
        evaluator,
        action_ranker,
        max_depth: int = 50,
        beam_width: int = 3,
        game_description: str = None,
        batch_evaluator=None,
        batch_action_ranker=None,
        use_batch_processing: bool = True,
        backup_pool_size: int = None,
        max_concurrent_llm_calls: int = 8,
        max_history_size: int = 10,
    ):
        self.evaluator = evaluator
        self.action_ranker = action_ranker
        self.batch_evaluator = batch_evaluator
        self.batch_action_ranker = batch_action_ranker
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.game_description = game_description
        self.use_batch_processing = use_batch_processing
        # Default backup pool size to 5x beam width
        self.backup_pool_size = backup_pool_size or (beam_width * 5)
        self.max_concurrent_llm_calls = max_concurrent_llm_calls
        self.max_history_size = max_history_size

    def _evaluate_states_batch(self, states: list, evaluation_history: list) -> list:
        """Batch evaluation of multiple states in a single LLM call"""
        if not self.batch_evaluator:
            return self._evaluate_states_sequential(states, evaluation_history)

        import time

        start_time = time.time()

        try:
            print(f"   🚀 Batch evaluating {len(states)} states in single LLM call...")

            # Prepare batch inputs
            structured_states = []
            action_histories = []
            evaluation_histories = []
            goal_state = None

            for state in states:
                if hasattr(state, "to_structured_input"):
                    structured_states.append(state.to_structured_input())
                else:
                    structured_states.append(state)

                action_histories.append(getattr(state, "moves_made", []))

                # Limit evaluation history for each state
                limited_history = _limit_evaluation_history(
                    getattr(state, "evaluation_history", []), self.max_history_size
                )
                evaluation_histories.append(limited_history)

                if goal_state is None and hasattr(state, "goal_state"):
                    goal_state = state.goal_state

            # Make single batch LLM call
            kwargs = dict(
                states=structured_states,
                goal_state=goal_state,
                action_histories=action_histories,
                evaluation_histories=evaluation_histories,
                game_description=self.game_description or "Generic puzzle game",
            )

            result = self.batch_evaluator(**kwargs)

            # Extract scores from batch result
            if hasattr(result, "batch_evaluation") and hasattr(
                result.batch_evaluation, "state_scores"
            ):
                scores = result.batch_evaluation.state_scores
                reasoning = getattr(
                    result.batch_evaluation, "reasoning", "Batch evaluation"
                )
                individual_reasoning = getattr(
                    result.batch_evaluation, "individual_reasoning", []
                )
                best_index = getattr(result.batch_evaluation, "best_state_index", 0)
            else:
                print("   ⚠️  Unexpected batch result structure, using fallback scores")
                scores = [0.5] * len(states)
                reasoning = "Fallback evaluation"
                individual_reasoning = ["Fallback reasoning"] * len(states)
                best_index = 0

            # Store evaluation details and update state histories
            scored_states = []
            for i, (state, score) in enumerate(zip(states, scores)):
                individual_reason = (
                    individual_reasoning[i]
                    if i < len(individual_reasoning)
                    else reasoning
                )

                eval_details = {
                    "reasoning": f"Batch eval (#{i}): {individual_reason}",
                    "best_action": "N/A",
                    "batch_best": i == best_index,
                }
                evaluation_history.append((str(state.state_data), score, eval_details))

                # Add this evaluation to the state's history for future reference
                state.add_evaluation(score, individual_reason, state.depth)
                scored_states.append((score, state))

            # Sort by score (highest first) but return ALL states, not just top beam_width
            scored_states.sort(reverse=True, key=lambda x: x[0])

            eval_time = time.time() - start_time
            print(f"   ⏱️  Batch evaluation completed in {eval_time:.2f}s")
            print(f"   📊 Score range: {min(scores):.3f} - {max(scores):.3f}")

            # Return all scored states
            return [s for _, s in scored_states]

        except Exception as e:
            print(f"   ⚠️  Batch evaluation failed: {e}")
            print("   🔄 Falling back to sequential evaluation...")
            return self._evaluate_states_sequential(states, evaluation_history)

    def _evaluate_states_sequential(
        self, states: list, evaluation_history: list
    ) -> list:
        """Sequential evaluation (original implementation)"""
        if self.evaluator is None:
            return states

        scored_states = []
        for state in states:
            # Limit evaluation history to prevent context explosion
            limited_history = _limit_evaluation_history(
                getattr(state, "evaluation_history", []), self.max_history_size
            )

            kwargs = dict(
                current_state=(
                    state.to_structured_input()
                    if hasattr(state, "to_structured_input")
                    else state
                ),
                goal_state=getattr(state, "goal_state", None),
                valid_actions=state.get_valid_actions(),
                action_history=getattr(state, "moves_made", []),
                evaluation_history=limited_history,
            )
            if self.game_description is not None:
                kwargs["game_description"] = self.game_description

            result = self.evaluator(**kwargs)
            score = (
                getattr(result.evaluation, "score", 0)
                if hasattr(result, "evaluation")
                else 0
            )
            reasoning = (
                getattr(result.evaluation, "reasoning", "N/A")
                if hasattr(result, "evaluation")
                else "N/A"
            )

            # Store evaluation details for history
            eval_details = {
                "reasoning": reasoning,
                "best_action": (
                    getattr(result.evaluation, "best_action", "N/A")
                    if hasattr(result, "evaluation")
                    else "N/A"
                ),
            }
            evaluation_history.append((str(state.state_data), score, eval_details))

            # Add this evaluation to the state's history
            state.add_evaluation(score, reasoning, state.depth)

            scored_states.append((score, state))

        scored_states.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored_states]

    def _rank_actions_batch(self, state_action_pairs: list) -> list:
        """Batch action ranking for multiple states in a single LLM call"""
        if not self.batch_action_ranker or len(state_action_pairs) == 1:
            return self._rank_actions_sequential(state_action_pairs)

        import time

        start_time = time.time()

        try:
            print(
                f"   🚀 Batch ranking actions for {len(state_action_pairs)} states..."
            )

            # Prepare batch inputs
            structured_states = []
            valid_actions_per_state = []
            depths = []
            evaluation_histories = []
            goal_state = None

            for state, valid_actions in state_action_pairs:
                if hasattr(state, "to_structured_input"):
                    structured_states.append(state.to_structured_input())
                else:
                    structured_states.append(state)

                valid_actions_per_state.append(valid_actions)
                depths.append(getattr(state, "depth", 0))

                # Limit evaluation history for each state
                limited_history = _limit_evaluation_history(
                    getattr(state, "evaluation_history", []), self.max_history_size
                )
                evaluation_histories.append(limited_history)

                if goal_state is None and hasattr(state, "goal_state"):
                    goal_state = state.goal_state

            # Make single batch LLM call
            kwargs = dict(
                states=structured_states,
                valid_actions_per_state=valid_actions_per_state,
                goal_state=goal_state,
                depths=depths,
                evaluation_histories=evaluation_histories,
                game_description=self.game_description or "Generic puzzle game",
            )

            result = self.batch_action_ranker(**kwargs)

            # Extract rankings from batch result
            if hasattr(result, "batch_ranking") and hasattr(
                result.batch_ranking, "state_rankings"
            ):
                rankings = result.batch_ranking.state_rankings
            else:
                print("   ⚠️  Unexpected batch ranking structure, using original order")
                rankings = [valid_actions for _, valid_actions in state_action_pairs]

            # Convert rankings to proper models
            final_rankings = []
            for i, ((state, valid_actions), ranked_actions) in enumerate(
                zip(state_action_pairs, rankings)
            ):
                if valid_actions:
                    model_cls = type(valid_actions[0])

                    def to_model(a):
                        # FIXED: Handle various return types from LLM
                        if isinstance(a, (int, float)):
                            print(
                                f"   ⚠️  WARNING: Got numeric value {a} instead of action - using first valid action"
                            )
                            return valid_actions[0] if valid_actions else None
                        elif isinstance(a, str):
                            print(
                                f"   ⚠️  WARNING: Got string value '{a}' instead of action - using first valid action"
                            )
                            return valid_actions[0] if valid_actions else None
                        elif isinstance(a, dict):
                            action_data = a.get("action", a)
                            try:
                                return model_cls(**action_data)
                            except Exception:
                                for valid_action in valid_actions:
                                    match = True
                                    for field_name in model_cls.__fields__.keys():
                                        if hasattr(
                                            valid_action, field_name
                                        ) and action_data.get(field_name) != getattr(
                                            valid_action, field_name
                                        ):
                                            match = False
                                            break
                                    if match:
                                        return valid_action
                                return valid_actions[0]
                        elif hasattr(a, "from_tower") and hasattr(a, "to_tower"):
                            # It's already a proper MoveModel
                            return a
                        else:
                            print(
                                f"   ⚠️  WARNING: Unexpected action type {type(a)}: {a} - using first valid action"
                            )
                            return valid_actions[0] if valid_actions else None

                    converted_ranking = [to_model(a) for a in ranked_actions]
                    final_rankings.append((state, converted_ranking))
                else:
                    final_rankings.append((state, []))

            ranking_time = time.time() - start_time
            print(f"   ⏱️  Batch action ranking completed in {ranking_time:.2f}s")

            return final_rankings

        except Exception as e:
            print(f"   ⚠️  Batch action ranking failed: {e}")
            print("   🔄 Falling back to sequential ranking...")
            return self._rank_actions_sequential(state_action_pairs)

    def _rank_actions_sequential(self, state_action_pairs: list) -> list:
        """Sequential action ranking"""
        results = []
        for state, valid_actions in state_action_pairs:
            goal_state = None
            if hasattr(state, "goal_state"):
                goal_state = state.goal_state
            depth = getattr(state, "depth", 0)
            if hasattr(state, "to_structured_input"):
                structured_state = state.to_structured_input()
            else:
                structured_state = state

            # Limit evaluation history to prevent context explosion
            limited_history = _limit_evaluation_history(
                getattr(state, "evaluation_history", []), self.max_history_size
            )

            kwargs = dict(
                current_state=structured_state,
                valid_actions=valid_actions,
                goal_state=goal_state,
                depth=depth,
                evaluation_history=limited_history,
            )
            if self.game_description is not None:
                kwargs["game_description"] = self.game_description

            try:
                result = self.action_ranker(**kwargs)
                if hasattr(result, "ranking") and hasattr(
                    result.ranking, "ranked_actions"
                ):
                    ranked = list(result.ranking.ranked_actions)
                elif hasattr(result, "ranking"):
                    ranked = list(result.ranking)
                elif hasattr(result, "ranked_actions"):
                    ranked = list(result.ranked_actions)
                else:
                    ranked = list(valid_actions)
            except Exception as e:
                print(f"   ⚠️  Action ranking failed: {e}")
                ranked = list(valid_actions)

            # Convert to models
            if len(valid_actions) > 0:
                model_cls = type(valid_actions[0])

                def to_model(a):
                    # FIXED: Handle various return types from LLM
                    if isinstance(a, (int, float)):
                        print(
                            f"   ⚠️  WARNING: Got numeric value {a} instead of action - using first valid action"
                        )
                        return valid_actions[0] if valid_actions else None
                    elif isinstance(a, str):
                        print(
                            f"   ⚠️  WARNING: Got string value '{a}' instead of action - using first valid action"
                        )
                        return valid_actions[0] if valid_actions else None
                    elif isinstance(a, dict):
                        action_data = a.get("action", a)
                        try:
                            return model_cls(**action_data)
                        except Exception:
                            for valid_action in valid_actions:
                                match = True
                                for field_name in model_cls.__fields__.keys():
                                    if hasattr(
                                        valid_action, field_name
                                    ) and action_data.get(field_name) != getattr(
                                        valid_action, field_name
                                    ):
                                        match = False
                                        break
                                if match:
                                    return valid_action
                            return valid_actions[0]
                    elif hasattr(a, "from_tower") and hasattr(a, "to_tower"):
                        # It's already a proper MoveModel
                        return a
                    else:
                        print(
                            f"   ⚠️  WARNING: Unexpected action type {type(a)}: {a} - using first valid action"
                        )
                        return valid_actions[0] if valid_actions else None

                ranked = [to_model(a) for a in ranked]

            results.append((state, ranked))
        return results

    def solve(self, initial_state: GenericState) -> tuple[list, bool]:
        """Enhanced beam search with backup state pool and batch processing"""
        beam = [initial_state]
        backup_states = []  # Pool of alternative states with their scores
        visited = {str(initial_state.state_data)}
        states_explored = 0
        evaluation_history = []
        beam_replenishments = 0

        processing_mode = (
            "batch"
            if self.use_batch_processing and self.batch_evaluator
            else "sequential"
        )

        print(f"\n🎯 Starting enhanced beam search in {processing_mode} mode...")
        print(f"   Max depth: {self.max_depth}, Beam width: {self.beam_width}")
        print(f"   Backup pool: {self.backup_pool_size}")
        print(f"   History limit: {self.max_history_size} evaluations per state")
        print(f"   Initial state: {initial_state.state_data}")
        print("=" * 60)

        for depth in range(self.max_depth):
            # If beam is empty but we have backup states, replenish the beam
            if not beam and backup_states:
                beam_replenishments += 1
                print(
                    f"\n🔄 BEAM EXHAUSTED at depth {depth} - Replenishing from backup pool..."
                )
                print(f"   Backup pool has {len(backup_states)} states available")

                # Sort backup states by score (highest first) and take the best ones for the beam
                backup_states.sort(key=lambda x: x[0], reverse=True)
                beam = [state for _, state in backup_states[: self.beam_width]]
                backup_states = backup_states[self.beam_width :]

                print(
                    f"   📈 Restored beam with {len(beam)} states (replenishment #{beam_replenishments})"
                )
                print(
                    f"   📊 Backup pool now has {len(backup_states)} states remaining"
                )

            if not beam:
                print(f"\n💀 No more states to explore at depth {depth}")
                break

            print(
                f"\n📊 Depth {depth:3d} | Beam size: {len(beam):2d} | "
                f"Backup pool: {len(backup_states):3d} | States explored: {states_explored:4d}"
            )

            # Show latest evaluations from recent states
            if evaluation_history:
                recent_evals = evaluation_history[-3:]  # Get last 3 evaluations
                print("   📋 Latest 3 state evaluations:")
                for i, (state_desc, score, eval_details) in enumerate(recent_evals):
                    reasoning = eval_details.get("reasoning", "No reasoning available")
                    # FIXED: Show the BEGINNING of reasoning, not everything after char 240
                    short_reasoning = (
                        reasoning[:240] + "..." if len(reasoning) > 240 else reasoning
                    )
                    print(
                        f"     #{len(evaluation_history) - len(recent_evals) + i + 1}: {score:.3f} - {short_reasoning}"
                    )

            # Check for goal states
            for state in beam:
                if state.is_goal():
                    print("\n🎉 SOLUTION FOUND! 🎉")
                    print(f"Steps explored: {states_explored}")
                    print(f"Solution depth: {depth}")
                    print(f"Solution moves: {len(state.moves_made)}")
                    print(f"Beam replenishments used: {beam_replenishments}")
                    print("\nEvaluation trajectory for solution path:")
                    for i, eval_record in enumerate(state.evaluation_history):
                        print(
                            f"  Step {eval_record.depth}: Score={eval_record.score:.3f} - "
                            f"{eval_record.reasoning[:240]}"  # FIXED: Show beginning, not end
                        )
                    return state.moves_made, True

            next_states = []
            beam_start_size = len(beam)

            # Prepare state-action pairs for ranking
            state_action_pairs = []
            for state in beam:
                valid_actions = state.get_valid_actions()
                if valid_actions:
                    state_action_pairs.append((state, valid_actions))
                elif depth < 10:
                    print("   ⚠️  State has no valid actions")

            # Batch or sequential action ranking
            if (
                self.use_batch_processing
                and self.batch_action_ranker
                and len(state_action_pairs) > 1
            ):
                state_rankings = self._rank_actions_batch(state_action_pairs)
            else:
                state_rankings = self._rank_actions_sequential(state_action_pairs)

            # Process ranked actions to generate next states
            for state, ranked_actions in state_rankings:
                states_explored += 1

                # Additional safety: filter out any invalid actions that might have been suggested
                valid_actions_for_state = state.get_valid_actions()

                # Check if this is a high-scoring state that warrants sequence exploration
                is_high_scoring = (
                    hasattr(state, "evaluation_history")
                    and state.evaluation_history
                    and state.evaluation_history[-1].score > 0.65
                )

                if is_high_scoring:
                    current_score = state.evaluation_history[-1].score
                    print(
                        f"   🔍 High-scoring state detected ({current_score:.3f}) - exploring move sequences"
                    )

                # Standard single-move exploration
                for action in ranked_actions[: self.beam_width]:
                    # Double-check that this action is actually valid before applying
                    if hasattr(state, "is_valid_action") and not state.is_valid_action(
                        action
                    ):
                        print(
                            f"   🚫 FILTERED INVALID ACTION: {action} (not in valid set)"
                        )
                        continue

                    # Also check if action is in the original valid actions list
                    action_is_valid = any(
                        getattr(valid_act, "from_tower", None)
                        == getattr(action, "from_tower", None)
                        and getattr(valid_act, "to_tower", None)
                        == getattr(action, "to_tower", None)
                        for valid_act in valid_actions_for_state
                    )

                    if not action_is_valid:
                        print(f"   🚫 FILTERED ACTION NOT IN VALID SET: {action}")
                        continue

                    new_state = state.apply_action(action)
                    # Copy evaluation history to new state
                    new_state.evaluation_history = list(state.evaluation_history)
                    if str(new_state.state_data) not in visited:
                        visited.add(str(new_state.state_data))
                        next_states.append(new_state)

                # ENHANCED: Multi-move sequence exploration for high-scoring states
                if is_high_scoring:
                    sequence_count = 0

                    # Explore 2-move sequences
                    for first_action in ranked_actions[
                        : min(3, len(ranked_actions))
                    ]:  # Top 3 first moves
                        if not state.is_valid_action(first_action):
                            continue

                        intermediate_state = state.apply_action(first_action)

                        for second_action in intermediate_state.get_valid_actions()[
                            :2
                        ]:  # Top 2 second moves
                            final_state = intermediate_state.apply_action(second_action)
                            final_state.evaluation_history = list(
                                state.evaluation_history
                            )

                            config_key = str(final_state.state_data)
                            if config_key not in visited:
                                visited.add(config_key)
                                next_states.append(final_state)
                                sequence_count += 1
                                if depth < 10:  # Only show details for early depths
                                    print(
                                        f"     🎯 Added 2-move sequence: {first_action.from_tower}→{first_action.to_tower} → {second_action.from_tower}→{second_action.to_tower}"
                                    )

                    # Explore 3-move sequences (more selective)
                    for first_action in ranked_actions[
                        : min(2, len(ranked_actions))
                    ]:  # Top 2 first moves
                        if not state.is_valid_action(first_action):
                            continue

                        intermediate_state = state.apply_action(first_action)

                        for second_action in intermediate_state.get_valid_actions()[
                            :2
                        ]:  # Top 2 second moves
                            second_state = intermediate_state.apply_action(
                                second_action
                            )

                            for third_action in second_state.get_valid_actions()[
                                :1
                            ]:  # Top 1 third move
                                final_state = second_state.apply_action(third_action)
                                final_state.evaluation_history = list(
                                    state.evaluation_history
                                )

                                config_key = str(final_state.state_data)
                                if config_key not in visited:
                                    visited.add(config_key)
                                    next_states.append(final_state)
                                    sequence_count += 1
                                    if depth < 10:  # Only show details for early depths
                                        print(
                                            f"     🚀 Added 3-move sequence: {first_action.from_tower}→{first_action.to_tower} → {second_action.from_tower}→{second_action.to_tower} → {third_action.from_tower}→{third_action.to_tower}"
                                        )

                    if sequence_count > 0:
                        print(
                            f"   📈 Generated {sequence_count} additional sequence states from high-scoring state"
                        )

            # Show state generation progress
            if depth < 10:
                print(
                    f"   🌱 Generated {len(next_states)} new states from {beam_start_size} beam states"
                )

            if next_states:
                # Batch or sequential state evaluation - get ALL evaluated states
                if self.use_batch_processing and self.batch_evaluator:
                    all_evaluated_states = self._evaluate_states_batch(
                        next_states, evaluation_history
                    )
                else:
                    all_evaluated_states = self._evaluate_states_sequential(
                        next_states, evaluation_history
                    )

                # Split states: top beam_width for beam, rest for backup pool
                beam = all_evaluated_states[: self.beam_width]
                potential_backups = all_evaluated_states[self.beam_width :]

                # Add potential backups to backup pool with their scores
                for state in potential_backups:
                    if evaluation_history:
                        # Find the score for this state from recent evaluations
                        recent_evals = evaluation_history[-len(all_evaluated_states) :]
                        state_score = next(
                            (
                                score
                                for eval_state, score, _ in recent_evals
                                if eval_state == str(state.state_data)
                            ),
                            0.5,
                        )
                        backup_states.append((state_score, state))

                # Limit backup pool size to prevent memory explosion
                if len(backup_states) > self.backup_pool_size:
                    backup_states.sort(key=lambda x: x[0], reverse=True)
                    backup_states = backup_states[: self.backup_pool_size]
                    if depth < 10:
                        print(
                            f"   🗂️  Trimmed backup pool to {self.backup_pool_size} best states"
                        )

                if depth < 5 and evaluation_history:
                    recent_scores = [
                        score for _, score, _ in evaluation_history[-len(beam) :]
                    ]
                    if recent_scores:
                        print(
                            f"   📊 Beam score range: {min(recent_scores):.3f} - {max(recent_scores):.3f}"
                        )

                # Show backup pool status
                if depth < 10 and backup_states:
                    backup_scores = [score for score, _ in backup_states]
                    print(
                        f"   💾 Backup pool: {len(backup_states)} states, "
                        f"score range: {min(backup_scores):.3f} - {max(backup_scores):.3f}"
                    )

                # Show evaluation trend for early depths
                if depth < 5 and beam:
                    for i, state in enumerate(beam[:3]):  # Show top 3 states
                        if state.evaluation_history:
                            latest_eval = state.evaluation_history[-1]
                            trend = ""
                            if len(state.evaluation_history) > 1:
                                prev_score = state.evaluation_history[-2].score
                                if latest_eval.score > prev_score:
                                    trend = "📈"
                                elif latest_eval.score < prev_score:
                                    trend = "📉"
                                else:
                                    trend = "➡️"
                            print(
                                f"     State {i+1}: {latest_eval.score:.3f} {trend} "
                                f"({len(state.evaluation_history)} evals)"
                            )
            else:
                beam = []
                print(f"   💀 No new states generated at depth {depth}")

        print("\n" + "=" * 60)
        print("❌ SEARCH FAILED")
        print(f"Steps explored: {states_explored}")
        print(f"Max depth reached: {self.max_depth}")
        print(f"Final beam size: {len(beam)}")
        print(f"Final backup pool size: {len(backup_states)}")
        print(f"Beam replenishments used: {beam_replenishments}")
        print(f"Total evaluations: {len(evaluation_history)}")

        if evaluation_history:
            scores = [score for _, score, _ in evaluation_history]
            print("Score statistics:")
            print(f"  Best score: {max(scores):.3f}")
            print(f"  Average score: {sum(scores)/len(scores):.3f}")
            print(f"  Worst score: {min(scores):.3f}")

        # Show evaluation trajectories for best final states
        all_final_states = beam + [state for _, state in backup_states[:3]]
        if all_final_states:
            print("Evaluation trajectories for best final states:")
            for i, state in enumerate(all_final_states[:3]):
                print(
                    f"\nState {i+1} trajectory ({len(state.evaluation_history)} evaluations):"
                )
                for eval_record in state.evaluation_history[-5:]:  # Show last 5
                    print(
                        f"  Depth {eval_record.depth}: {eval_record.score:.3f} - "
                        f"{eval_record.reasoning[:120]}..."  # This one was already correct
                    )

        return [], False


def generate_solvable_checker_jumping_board(num_squares: int) -> list:
    """
    Generate a list of obstacle positions for a guaranteed-solvable Checker Jumping board.
    Obstacles are placed at every other position (1, 3, 5, ...) so the main piece can always jump forward.
    Usage:
        obstacles = generate_solvable_checker_jumping_board(num_squares)
        state = create_initial_checker_jumping_state(num_squares, obstacles)
    """
    return [i for i in range(1, num_squares - 1, 2)]

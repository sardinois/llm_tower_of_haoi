import dspy
from llm_reasoning.models import (
    StateModel,
    ActionModel,
    EvaluationModel,
    RankedActionsModel,
    BatchEvaluationModel,
    BatchRankedActionsModel,
    HistoricalEvaluationModel,
    TowersModel,
    MoveModel,
)
from typing import List


class StateEvaluator(dspy.Signature):
    """
    Generic State Evaluator - Universal Problem-Solving Principles

    Evaluate this state by reasoning about fundamental problem-solving concepts.

    HISTORICAL CONTEXT:
    You have access to previous evaluations along the path that led to this state.
    Use this information to understand:
    - Whether we're making consistent progress or stagnating
    - If recent moves were strategic or counterproductive
    - How this state compares to recent alternatives
    - Whether we're following a coherent strategy or wandering

    SCORING PHILOSOPHY - AVOID CONVERGENCE EXTREMES:

    CRITICAL: Use the full 0.0-1.0 range meaningfully! Early states should typically score in the 0.3-0.7 range.
    Reserve 0.8+ scores for genuinely advanced states, and 0.9+ for near-solutions only.

    RELATIVE PROGRESS APPROACH:
    - Consider how far we've come from the starting position (don't start too high)
    - Consider how far we still need to go (don't score too optimistically)
    - Look for incremental but meaningful differences between similar states
    - Factor in the complexity/difficulty of the remaining problem

    SCORE DISTRIBUTION GUIDELINES:

    0.85-1.0: NEAR COMPLETION - Very few moves away from goal
    - Solution is 1-3 optimal moves away
    - All major obstacles overcome, just finishing touches remain
    - Reserve 0.95+ for states where goal is immediately obvious

    0.70-0.84: SUBSTANTIAL PROGRESS - Major milestones achieved
    - Significant portion of the problem solved (maybe 60-80%)
    - Key structural achievements unlocked
    - Clear path forward visible, though work remains

    0.55-0.69: MODERATE PROGRESS - Meaningful advancement
    - Noticeable progress from starting state (maybe 30-60% complete)
    - Some important sub-goals completed
    - Mix of opportunities and challenges ahead

    0.40-0.54: EARLY PROGRESS - Foundation building
    - Better than starting position but substantial work remains
    - Basic groundwork laid, main challenges still ahead
    - Some preparatory moves made

    0.25-0.39: MINIMAL PROGRESS - Small steps forward
    - Slight improvement over starting position
    - Exploratory moves that don't create major advantages
    - Learning/positioning without clear advancement

    0.10-0.24: COUNTERPRODUCTIVE - Moving backwards
    - Worse than recent states or starting position
    - Actions that complicate or undermine progress
    - May require significant correction

    0.0-0.09: CRITICAL FAILURE - Major setbacks or dead ends
    - Severely compromised position
    - Dead end or unsolvable configuration
    - Major strategic blunders

    COMPARATIVE CONTEXT:
    - If evaluation history shows all recent scores are 0.95+, you're probably being too generous
    - Look for meaningful differences: a slightly better state might score 0.52 vs 0.48
    - Early in search (depth < 10), most states should be in 0.3-0.7 range
    - Don't let historical high scores inflate current scoring - be objective about actual progress
    """

    current_state: StateModel = dspy.InputField(
        desc="Current state as structured model"
    )
    goal_state: StateModel = dspy.InputField(desc="Goal state as structured model")
    valid_actions: List[ActionModel] = dspy.InputField(
        desc="List of valid actions as structured models"
    )
    action_history: List[ActionModel] = dspy.InputField(
        desc="Actions taken so far as structured models"
    )
    evaluation_history: List[HistoricalEvaluationModel] = dspy.InputField(
        desc="Previous state evaluations along this path, showing the trajectory of scores and reasoning"
    )
    evaluation: EvaluationModel = dspy.OutputField(
        desc="Evaluation result as structured model"
    )


class BatchStateEvaluator(dspy.Signature):
    """
    Batch State Evaluator - Evaluate Multiple States in One Call

    Evaluate ALL the provided states by comparing them to the number of moves needed to reach the goal state.
    Evaluation should be based on the goal state and the number of moves needed to reach it.

    CRITICAL SCORING GUIDANCE:
    - Spread scores across the full range from 0.0 to 1.0 - don't cluster around middle values
    - Use meaningful differences between states: if one state is clearly better, score it noticeably higher
    - Use precise decimal values like 0.42, 0.58, 0.73 rather than round numbers
    - Don't be afraid to use the extremes when appropriate

    HISTORICAL CONTEXT:
    Each state has its own evaluation history showing the path taken to reach it.
    Use this to understand which paths show consistent progress vs. stagnation.
    If you see previous evaluations consistently scoring very high, you may need to recalibrate to maintain meaningful differences.

    COMPARATIVE EVALUATION:
    - First score states based on actual progress toward the goal
    - Then adjust scores to ensure clear differentiation between different quality states
    - Factor in trajectory trends, but focus on current state quality
    - Aim for significant score differences between clearly different quality states

    SCORING PHILOSOPHY:

    NEAR COMPLETION: Very few moves away from goal
    - Solution is immediately visible with just final positioning needed
    - All major structural work complete
    - Score in the highest range when truly close to finishing

    SUBSTANTIAL PROGRESS: Major milestones achieved
    - Most of the puzzle structure is correctly built
    - Key breakthrough moments accomplished
    - Clear path to completion visible

    MODERATE PROGRESS: Meaningful advancement visible
    - Noticeable improvement from starting position
    - Important intermediate goals achieved
    - Good foundation laid for future moves

    EARLY PROGRESS: Foundation building phase
    - Better than starting position with some structure emerging
    - Preparatory work underway
    - Beginning to organize toward solution

    MINIMAL PROGRESS: Small steps forward
    - Slight improvement over starting position
    - Exploratory moves without major structural change
    - Learning phase with modest gains

    COUNTERPRODUCTIVE: Moving away from solution
    - Worse than recent states or complicating the path forward
    - Actions that undermine previous progress
    - May require backtracking to recover

    CRITICAL FAILURE: Major setbacks or impossible positions
    - Severely compromised or unsolvable configurations
    - Fundamental strategic errors
    - Dead ends requiring significant correction

    CALIBRATION GUIDELINES:
    - If recent evaluation history shows clustering around high scores, be more selective with top scores
    - Early in the search, most states should be in the lower-to-middle ranges
    - Reserve the highest scores for states that are genuinely close to completion
    - Use the full scoring range to maximize differentiation between states

    OUTPUT FORMAT:
    Return scores in the same order as the input states.
    """

    states: List[StateModel] = dspy.InputField(
        desc="List of states to evaluate as structured models"
    )
    goal_state: StateModel = dspy.InputField(desc="Goal state as structured model")
    action_histories: List[List[ActionModel]] = dspy.InputField(
        desc="Action history for each state (same order as states)"
    )
    evaluation_histories: List[List[HistoricalEvaluationModel]] = dspy.InputField(
        desc="Evaluation history for each state, showing the trajectory of scores and reasoning"
    )
    game_description: str = dspy.InputField(
        desc="Description of the game rules and objectives"
    )
    batch_evaluation: BatchEvaluationModel = dspy.OutputField(
        desc="Batch evaluation result with scores for all states"
    )


class ActionRanker(dspy.Signature):
    """
    Generic Action Ranking - Strategic Decision Making

    Rank these actions by reasoning about which moves demonstrate the best strategic thinking.

    HISTORICAL CONTEXT:
    Consider the evaluation history to understand:
    - What strategies have been working vs. failing
    - Whether we need to continue current approach or change direction
    - If previous high-scoring states suggest certain types of moves
    - Whether we're in a pattern that needs to be broken

    EVALUATION CRITERIA:

    PURPOSEFULNESS: Does this action serve a clear strategic purpose?
    TIMING: Is this the right moment for this type of action?
    RISK/REWARD: What are the potential consequences?
    ALTERNATIVES: How does this compare to other available options?
    TRAJECTORY: Does this action continue positive momentum or break negative patterns?

    Think like a strategic planner: which action represents the most intelligent
    next step given our current situation, ultimate objectives, and recent history?
    """

    current_state: StateModel = dspy.InputField(
        desc="Current state as structured model"
    )
    valid_actions: List[ActionModel] = dspy.InputField(
        desc="List of valid actions as structured models"
    )
    goal_state: StateModel = dspy.InputField(desc="Goal state as structured model")
    depth: int = dspy.InputField(desc="Current search depth")
    evaluation_history: List[HistoricalEvaluationModel] = dspy.InputField(
        desc="Previous state evaluations along this path, showing what has worked well"
    )
    ranking: RankedActionsModel = dspy.OutputField(
        desc="Ranked actions and strategy explanation as structured model"
    )


class BatchActionRanker(dspy.Signature):
    """
    Batch Action Ranking - Rank Actions for Multiple States

    For each state provided, rank its available actions from best to worst.
    Consider the strategic context, goals, and evaluation history for each state independently.

    HISTORICAL CONTEXT:
    Each state has its own evaluation history. Use this to understand what approaches
    have been successful and what patterns should be continued or avoided.

    Return rankings in the same order as input states.
    """

    states: List[StateModel] = dspy.InputField(
        desc="List of states as structured models"
    )
    valid_actions_per_state: List[List[ActionModel]] = dspy.InputField(
        desc="List of valid actions for each state (same order as states)"
    )
    goal_state: StateModel = dspy.InputField(desc="Goal state as structured model")
    depths: List[int] = dspy.InputField(
        desc="Search depth for each state (same order as states)"
    )
    evaluation_histories: List[List[HistoricalEvaluationModel]] = dspy.InputField(
        desc="Evaluation history for each state (same order as states)"
    )
    game_description: str = dspy.InputField(
        desc="Description of the game rules and objectives"
    )
    batch_ranking: BatchRankedActionsModel = dspy.OutputField(
        desc="Batch ranking result with ranked actions for all states"
    )


# Tower of Hanoi specializations
class HanoiStateEvaluator(StateEvaluator):
    """
    Tower of Hanoi State Evaluator - Reasoning-Based Approach

    Your task is to evaluate how promising this game state is by reasoning about
    the puzzle structure and what needs to happen next.

    Think about this state as a step in a larger problem-solving process.
    Consider the constraints, the goal, what meaningful progress looks like,
    and how this state fits into the trajectory shown by the evaluation history.

    TOWER OF HANOI SPECIFIC GUIDANCE:
    - For N disks, optimal solution requires 2^N - 1 moves
    - Early states (first 10-20% of optimal moves) should score 0.3-0.6
    - Mid-game states (20-70% progress) should score 0.5-0.8
    - Near-endgame states (80%+ progress) can score 0.8+
    - Only states 1-3 moves from completion should score 0.9+
    """

    __doc__ = StateEvaluator.__doc__ + __doc__

    game_description: str = dspy.InputField(
        default=(
            "Tower of Hanoi: Move all disks from the source peg to the target peg, using the auxiliary peg. "
            "Only one disk can be moved at a time, and a larger disk cannot be placed on a smaller disk."
        ),
        desc="Description of the Tower of Hanoi game and rules.",
    )
    current_state: TowersModel = dspy.InputField(desc="Current Tower configuration")
    goal_state: TowersModel = dspy.InputField(desc="Target Tower configuration")
    valid_actions: List[MoveModel] = dspy.InputField(desc="List of valid moves")
    action_history: List[MoveModel] = dspy.InputField(desc="Moves made so far")
    evaluation_history: List[HistoricalEvaluationModel] = dspy.InputField(
        desc="Previous state evaluations showing the scoring trajectory"
    )


class HanoiBatchStateEvaluator(BatchStateEvaluator):
    """
    Tower of Hanoi Batch State Evaluator

    CRITICAL TOWER OF HANOI UNDERSTANDING:
    Tower of Hanoi requires moving disks in a specific strategic order. Having smaller disks
    on the target peg is NOT always progress - they often need to be moved off to place larger disks.
    Some states might superficially look like they are making progress, but in reality they are not. R
    Reason about the state and effort required to reach the goal state and base your score on that.

    STRATEGIC EVALUATION PRINCIPLES:

    OPTIMAL POSITIONING:
    - Larger disks positioned closer to their final location on target peg
    - Smaller disks positioned to allow larger disk movement
    - Clear path available for the next logical large disk placement

    SUBGOAL COMPLETION:
    - Successfully moved a large disk to the target peg permanently
    - Created proper foundation with largest disks in correct order
    - Smaller disks properly staged on auxiliary peg for efficient transfer

    SCORING CALIBRATION GUIDANCE:

    Use the EVALUATION HISTORY to understand progress context:
    - If you see many previous scores above 0.8, you may be scoring too generously
    - Look at the action history length to gauge how far into the puzzle we are
    - Compare current states to the best and worst states you've seen before

    PROGRESS BENCHMARKS FOR SCORING:

    EXCEPTIONAL PROGRESS (reserve for truly outstanding states):
    - Multiple large disks correctly and permanently positioned on target peg
    - Nearly all structural work complete, just final organizing moves needed
    - Clear and immediate path to completion visible
    - Example: For 5 disks, having disks 5,4,3 correctly stacked on target peg

    SUBSTANTIAL PROGRESS (majority of structural goals achieved, clear steps toward completion):
    - The majority of large disks are correctly and permanently placed on the target peg
    - Visible, concrete steps have been taken toward the final goal, not just isolated progress
    - Smaller disks are staged to enable efficient completion, with minimal backtracking required
    - Example: For 5 disks, disks 5, 4, and possibly 3 are correctly stacked on the target peg, and the remaining disks are positioned to allow straightforward completion

    SOLID PROGRESS (good strategic positioning):
    - Effective grouping of smaller disks to enable large disk movement
    - Clear preparation for next major disk placement
    - Efficient intermediate configuration
    - Example: All smaller disks grouped on auxiliary peg, ready to move large disk

    MODERATE PROGRESS (mixed efficiency):
    - Some strategic advancement but with suboptimal elements
    - Partial progress toward goals
    - Better than typical but room for improvement
    - Example: Some disks well-positioned, others scattered

    EARLY PROGRESS (basic preparation):
    - Initial organizational moves
    - Slight improvement from starting position
    - Foundation-laying without major structural change
    - Example: One or two disks moved, main stack largely intact

    SCORING DISCIPLINE:
    - Most states should score in the middle ranges unless truly exceptional
    - Reserve scores above 0.8 for states that are genuinely close to completion
    - Use the full scoring range - don't cluster around middle values
    - If action history shows many moves taken, higher scores may be appropriate
    - If action history shows few moves taken, be conservative with scoring

    The goal is to move all disks from the source peg to the target peg, using the auxiliary peg as needed.

    ABSOLUTE PROGRESS EVALUATION:
    Score states based on their ABSOLUTE distance from completion, not relative to other states in the batch.

    REFERENCE ANCHORS for calibration:
    - INITIAL STATE (all disks on source peg): Should score around 0.200-0.300
    - FINAL GOAL STATE (all disks on target peg in order): Should score 1.000
    - HALFWAY PROGRESS (largest disk moved, others positioned strategically): Should score around 0.600-0.700

    EVALUATION METHODOLOGY:
    1. Estimate minimum moves needed to complete this state
    2. Compare to optimal total moves for this puzzle size
    3. Assign a score that reflects how close this state is to the goal in terms of moves required, with 1.0 representing the completed target state and lower values indicating greater distance from completion.

    SPREAD YOUR SCORES:
    - Use the FULL range 0.000-1.000
    - Don't cluster scores in narrow bands
    - Clearly bad states should score 0.200-0.400
    - Clearly good states should score 0.800-1.000
    """

    __doc__ = BatchStateEvaluator.__doc__ + __doc__

    states: List[TowersModel] = dspy.InputField(desc="List of tower states to evaluate")
    goal_state: TowersModel = dspy.InputField(desc="Target Tower configuration")
    action_histories: List[List[MoveModel]] = dspy.InputField(
        desc="Move history for each state"
    )
    evaluation_histories: List[List[HistoricalEvaluationModel]] = dspy.InputField(
        desc="Evaluation history for each state"
    )
    game_description: str = dspy.InputField(
        default=(
            "Tower of Hanoi: Reach the target state by moving disks one by one, using the auxiliary peg to reach the desired state "
            "Most important is that the right order of disks is reached in the target state."
            "Only one disk can be moved at a time, and a larger disk cannot be placed on a smaller disk."
        ),
        desc="Description of the Tower of Hanoi game and rules.",
    )


class HanoiActionRanker(ActionRanker):
    """
    Tower of Hanoi Action Ranker with Constraint Validation

    CRITICAL CONSTRAINT VALIDATION:
    In Tower of Hanoi, you can ONLY place a smaller disk on a larger disk or empty peg.
    Before ranking any action, verify: disk_being_moved < top_disk_destination (or destination is empty)

    INVALID MOVES TO REJECT:
    - Placing disk 3 on disk 1 or 2
    - Placing disk 4 on disk 1, 2, or 3
    - Any move where source disk number >= destination top disk number

    The goal is to Reach the target state by moving disks one by one, using the auxiliary peg to reach the desired state.
    Most important is that the right order of disks is reached in the target state, the disks should from the largest to the smallest on the target peg.
    """

    __doc__ = ActionRanker.__doc__ + __doc__

    game_description: str = dspy.InputField(
        default=(
            "Tower of Hanoi: Move all disks from the source peg to the target peg, using the auxiliary peg. "
            "CRITICAL RULE: Only one disk can be moved at a time, and a larger disk cannot be placed on a smaller disk. "
            "This means you can only move a disk to an empty peg or onto a larger disk."
        ),
        desc="Description of the Tower of Hanoi game and rules.",
    )
    current_state: TowersModel = dspy.InputField(
        desc="Tower configuration as structured model"
    )
    valid_actions: List[MoveModel] = dspy.InputField(
        desc="List of valid moves as structured models (pre-filtered for constraints)"
    )
    goal_state: TowersModel = dspy.InputField(
        desc=("Target tower configuration as structured model")
    )
    evaluation_history: List[HistoricalEvaluationModel] = dspy.InputField(
        desc="Previous state evaluations showing what strategies have worked"
    )


class HanoiBatchActionRanker(BatchActionRanker):
    """
    Tower of Hanoi Batch Action Ranker

    CRITICAL CONSTRAINT VALIDATION:
    In Tower of Hanoi, you can ONLY place a smaller disk on a larger disk or empty peg.
    Before ranking any action, verify: disk_being_moved < top_disk_destination (or destination is empty)

    INVALID MOVES TO REJECT:
    - Placing disk 3 on disk 1 or 2
    - Placing disk 4 on disk 1, 2, or 3
    - Any move where source disk number >= destination top disk number

    RANKING STRATEGY:
    Only rank moves that follow the constraint. If an action violates the rule,
    it should be ranked last or excluded from consideration entirely.
    """

    __doc__ = BatchActionRanker.__doc__ + __doc__

    states: List[TowersModel] = dspy.InputField(desc="List of tower states")
    valid_actions_per_state: List[List[MoveModel]] = dspy.InputField(
        desc="Valid moves for each state (pre-filtered to follow Tower of Hanoi constraints)"
    )
    goal_state: TowersModel = dspy.InputField(desc="Target Tower configuration")
    evaluation_histories: List[List[HistoricalEvaluationModel]] = dspy.InputField(
        desc="Evaluation history for each state"
    )
    game_description: str = dspy.InputField(
        default=(
            "Tower of Hanoi: Move all disks from the source peg to the target peg, using the auxiliary peg. "
            "CRITICAL RULE: Only one disk can be moved at a time, and a larger disk cannot be placed on a smaller disk. "
            "This means you can only move a disk to an empty peg or onto a larger disk."
        ),
        desc="Description of the Tower of Hanoi game and rules.",
    )

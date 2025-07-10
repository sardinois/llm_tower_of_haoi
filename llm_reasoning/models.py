from pydantic import BaseModel, Field
from typing import List, Any, Literal, Optional


class StateModel(BaseModel):
    pass


class ActionModel(BaseModel):
    pass


class HistoricalEvaluationModel(BaseModel):
    """Historical evaluation data for a previous state"""

    state_description: str = Field(description="Brief description of the state")
    score: float = Field(
        description="The score that was assigned to this state (0.0000-1.0000)"
    )
    reasoning: str = Field(description="The reasoning behind the score")
    depth: int = Field(description="The depth/step when this evaluation was made")


class EvaluationModel(BaseModel):
    score: float = Field(
        description="The score of the current state between 0.0000 and 1.0000, 1.0000 being the best, closest to the goal state. The goal is to move all disks from Peg A to Peg C",
        min_value=0.0000,
        max_value=1.0000,
    )
    best_action: Any
    reasoning: str


class RankedActionsModel(BaseModel):
    ranked_actions: List[Any]
    strategy_explanation: str


# Batch evaluation models
class BatchEvaluationModel(BaseModel):
    state_scores: List[float] = Field(
        description="List of scores (0.000-1.0000) for each state in the same order as input states",
        min_value=0.0000,
        max_value=1.0000,
    )
    reasoning: str = Field(
        description="Overall reasoning explaining the comparative scoring of all states"
    )
    best_state_index: int = Field(
        description="Index (0-based) of the best state from the input list"
    )
    individual_reasoning: List[str] = Field(
        description="""Clearly articulate the reasoning for each state's score, 
        explain independantly why giving this score (do not refer to the other states in the batch in this reasoning)
        The state and reasoning + score should be readabile on its own, do not refer to the other states in the batch in this reasoning
        Do not refer to the state ordering or number in the batch (this will be read outside the context of the batch)""",  # noqa: E501
        default_factory=list,
    )


class BatchRankedActionsModel(BaseModel):
    state_rankings: List[List[Any]] = Field(
        description="List of ranked actions for each state, in the same order as input states"
    )
    strategy_explanation: str = Field(
        description="Overall strategy explanation for all action rankings"
    )


# Example for Tower of Hanoi
class TowersModel(StateModel):
    "This will represent the state of the towers, disks are represented by numbers, the last number in the list is the top disk"
    A: List[int]
    B: List[int]
    C: List[int]


class MoveModel(ActionModel):
    from_tower: Literal["A", "B", "C"]
    to_tower: Literal["A", "B", "C"]


class CheckerJumpingStateModel(StateModel):
    board: list[int]  # 0 = empty, 1 = obstacle, 2 = main piece
    main_pos: int


class CheckerJumpingActionModel(ActionModel):
    from_index: int
    to_index: int


class BlocksWorldStateModel(StateModel):
    stacks: list[list[str]]  # e.g., [['B', 'A'], ['C']]


class BlocksWorldActionModel(ActionModel):
    block: str
    destination: str  # another block or 'table'

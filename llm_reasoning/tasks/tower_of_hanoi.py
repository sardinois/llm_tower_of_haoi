from typing import List, Dict, Optional
from dataclasses import dataclass, field
from llm_reasoning.core import GenericState
from llm_reasoning.models import TowersModel, MoveModel


@dataclass
class TowerOfHanoiState(GenericState):
    state_data: Dict[str, List[int]]
    moves_made: List[MoveModel] = field(default_factory=list)
    parent: Optional["TowerOfHanoiState"] = None
    depth: int = 0

    def is_goal(self) -> bool:
        """Check if all disks are on the target tower C in correct order"""
        target_tower = "C"
        num_disks = sum(len(tower) for tower in self.state_data.values())
        goal_reached = len(self.state_data[target_tower]) == num_disks
        goal_state = list(range(num_disks, 0, -1))

        is_solved = goal_reached and self.state_data[target_tower] == goal_state

        # Debug: print when goal is detected
        if is_solved:
            print(
                f"🎯 GOAL DETECTED! All {num_disks} disks on tower {target_tower}: {self.state_data[target_tower]}"
            )

        return is_solved

    def get_valid_actions(self) -> List[MoveModel]:
        valid_moves = []
        for from_tower in ["A", "B", "C"]:
            # Skip empty towers
            if not self.state_data[from_tower]:
                continue

            # Get the top disk from the source tower
            top_disk = self.state_data[from_tower][-1]

            for to_tower in ["A", "B", "C"]:
                if from_tower == to_tower:
                    continue

                # Check Tower of Hanoi constraint: can only place on empty tower or larger disk
                to_tower_stack = self.state_data[to_tower]
                if not to_tower_stack:
                    # Empty tower - valid move
                    valid_moves.append(
                        MoveModel(from_tower=from_tower, to_tower=to_tower)
                    )
                elif top_disk < to_tower_stack[-1]:
                    # Smaller disk on larger disk - valid move
                    valid_moves.append(
                        MoveModel(from_tower=from_tower, to_tower=to_tower)
                    )
                # else: would place larger disk on smaller disk - invalid, skip

        return valid_moves

    def is_valid_action(self, action: MoveModel) -> bool:
        """Check if an action is valid in the current state"""
        # Check if towers exist
        if (
            action.from_tower not in self.state_data
            or action.to_tower not in self.state_data
        ):
            return False

        # Check if from_tower has disks
        if not self.state_data[action.from_tower]:
            return False

        # Check if move is allowed (smaller disk on larger disk)
        from_disk = self.state_data[action.from_tower][-1]
        to_tower_stack = self.state_data[action.to_tower]

        if to_tower_stack and from_disk >= to_tower_stack[-1]:
            return False

        return True

    def apply_action(self, action: MoveModel) -> "TowerOfHanoiState":
        # Validate the action before applying
        if not self.is_valid_action(action):
            print(f"⚠️  INVALID ACTION DETECTED: {action}")
            print(f"   Current state: {self.state_data}")
            print(
                f"   From tower '{action.from_tower}': {self.state_data.get(action.from_tower, 'MISSING')}"
            )
            print(
                f"   To tower '{action.to_tower}': {self.state_data.get(action.to_tower, 'MISSING')}"
            )

            # Return the current state unchanged instead of crashing
            return TowerOfHanoiState(
                state_data={k: v.copy() for k, v in self.state_data.items()},
                moves_made=self.moves_made.copy(),
                parent=self,
                depth=self.depth + 1,
            )

        new_towers = {k: v.copy() for k, v in self.state_data.items()}
        disk = new_towers[action.from_tower].pop()
        new_towers[action.to_tower].append(disk)
        new_moves = self.moves_made + [action]
        return TowerOfHanoiState(
            state_data=new_towers,
            moves_made=new_moves,
            parent=self,
            depth=self.depth + 1,
        )

    def to_structured_input(self) -> TowersModel:
        return TowersModel(**self.state_data)


def create_initial_hanoi_state(
    num_disks: int, source: str = "A", target: str = "C"
) -> TowerOfHanoiState:
    initial_towers = {source: list(range(num_disks, 0, -1)), "B": [], target: []}
    return TowerOfHanoiState(state_data=initial_towers)

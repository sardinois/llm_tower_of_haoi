import click
from llm_reasoning.core import GenericLLMGuidedSolver
from llm_reasoning.signatures import (
    HanoiStateEvaluator,
    HanoiActionRanker,
    HanoiBatchStateEvaluator,
    HanoiBatchActionRanker,
)
from llm_reasoning.tasks.tower_of_hanoi import create_initial_hanoi_state
import dspy


@click.command()
@click.option(
    "--num-disks",
    type=int,
    default=3,
    show_default=True,
    help="Number of disks for Tower of Hanoi",
)
@click.option(
    "--max-depth", type=int, default=20, show_default=True, help="Maximum search depth"
)
@click.option(
    "--beam-width", type=int, default=3, show_default=True, help="Beam width for search"
)
@click.option(
    "--use-batch",
    is_flag=True,
    default=True,
    show_default=True,
    help="Use batch evaluation (single LLM call for multiple states)",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=1,
    show_default=True,
    help="Max concurrent LLM calls (only used if --no-use-batch)",
)
def main(
    num_disks,
    max_depth,
    beam_width,
    use_batch,
    max_concurrent,
):
    # lm = dspy.LM(model="openai/o4-mini", temperature=1.0, max_tokens=20000)
    lm = dspy.LM(model="openai/gpt-4.1-mini")  # , temperature=1.0, max_tokens=20000)

    dspy.settings.configure(lm=lm)

    initial_state = create_initial_hanoi_state(num_disks)
    game_description = (
        "Tower of Hanoi: Reach the target state by moving disks one by one, using the auxiliary peg to reach the desired state "
        "Most important is that the right order of disks is reached in the target state, the disks should from the largest to the smallest on the target peg."
        "Only one disk can be moved at a time, and a larger disk cannot be placed on a smaller disk."
    )

    if use_batch:
        solver = GenericLLMGuidedSolver(
            evaluator=dspy.ChainOfThought(HanoiStateEvaluator),
            action_ranker=dspy.ChainOfThought(HanoiActionRanker),
            batch_evaluator=dspy.ChainOfThought(HanoiBatchStateEvaluator),
            batch_action_ranker=dspy.ChainOfThought(HanoiBatchActionRanker),
            max_depth=max_depth,
            beam_width=beam_width,
            game_description=game_description,
            use_batch_processing=True,
            max_concurrent_llm_calls=1,  # Not used when batch processing
        )
        print("🚀 Using batch processing mode (single LLM call per batch)")
    else:
        solver = GenericLLMGuidedSolver(
            evaluator=dspy.ChainOfThought(HanoiStateEvaluator),
            action_ranker=dspy.ChainOfThought(HanoiActionRanker),
            max_depth=max_depth,
            beam_width=beam_width,
            game_description=game_description,
            use_batch_processing=False,
            max_concurrent_llm_calls=max_concurrent,
        )
        print("🔄 Using sequential processing mode")

    moves, success = solver.solve(initial_state)
    if success:
        print(f"✅ Success! Solution in {len(moves)} moves:")
        for move in moves:
            print(f"  {move}")
        expected = 2**num_disks - 1
        print(f"Efficiency: {expected}/{len(moves)} = {expected/len(moves):.2f}")
    else:
        print("❌ Failed to solve.")


if __name__ == "__main__":
    main()

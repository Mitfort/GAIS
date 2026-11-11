from __future__ import annotations

import argparse
import hashlib
import json 
import math
import random 

from collections import Counter
from dataclasses import dataclass
from pathlib import Path 

import arc_agi
import numpy as np

from arcengine import GameAction, GameState


# STEP 1: Observation representation

def get_grid(observation) -> np.ndarray:
    """
    Extract the final visible frame. 
    """

    if observation is None or len(observation.frame) == 0:
        raise ValueError("Missing observation frame.")

    # Convert the frame to a numpy array
    grid: np.ndarray = np.array(observation.frame[-1])

    if grid.ndim != 2:
        raise ValueError(f"Unexpected grid shape: {grid.shape}.")

    return grid.copy()

def state_id(grid:np.ndarray) -> str:
    """
    Generate a unique identifier for the given grid state.
    """
    # Convert the grid to bytes and hash it
    grid_bytes = grid.tobytes()

    payload = (str(grid.shape).encode() + grid_bytes)

    return hashlib.blake2b(payload, digest_size=16).hexdigest()



# STEP 2: Action representation

@dataclass(frozen=True)
class Action: 
    """
    Represents a single action in the game.
    """
    kind: GameAction
    x:int | None = None
    y:int | None = None

    @property
    def key(self) -> str:
        """
        Generate a unique key for the action.
        """
        return f"{self.kind.name}:{self.x}:{self.y}"

    def payload(self) -> dict:
        """
        Convert the action to a dictionary payload.
        """
        if self.x is None:
            return {}

        return {"x": self.x, "y": self.y}



def legal_candidates(env: arc_agi.EnvironmentWrapper, grid: np.ndarray) -> list[Action]:
    """
    Generate a list of legal actions based on the current grid state.
    """
    height, width = grid.shape

    candidates: list[Action] = []

    for action in env.action_space:
        if action.is_complex():
            # Initial, coare coodinate probin
            for fy in (0.15, 0.5, 0.85):
                for fx in (0.15, 0.5, 0.85):
                    candidates.append(
                        Action(
                            kind=action,
                            x=int((width-1) * fx),
                            y=int((height-1) * fy)
                        )
                    )

        else:
            candidates.append(Action(kind=action))

    return candidates

def analyze_visual_change(
    before: np.ndarray,
    after: np.ndarray
) -> dict:
    """
    Analyze the visual change between two grid states.
    Returns the ratio of changed pixels.
    """
    if before.shape != after.shape:
        return {"scene_transition": True}

    diff = before != after
    h,w = diff.shape 

    # Split on the 8x8 boxes
    tile_h = max(1, h // 8)
    tile_w = max(1, w // 8)

    changed_regions = []

    for y in range(0, h, tile_h):
        for x in range(0, w, tile_w):
                tile = diff[
                    y:min(y + tile_h, h),
                    x:min(x + tile_w, w)
                ]

                if np.any(tile):
                    changed_regions.append({
                        "x": x,
                        "y": y,
                        "ratio": float(np.mean(tile))
                    })

    return {
        "scene_transition": False,
        "total_change": float(np.mean(diff)),
        "changed_regions": changed_regions
    }



# STEP 3: Empirical world model 

class WorldModel:
    def __init__(self):
        self.visited_states: set[str] = set()

        self.action_counts: Counter[str] = Counter()
        self.action_changes: Counter[str] = Counter()

        self.state_action_counts = Counter()
        self.total_actions: int = 0 


    def predict_change(self, action: Action) -> float:
        """
        Predict the probability of a state change given an action.

        P(visual change | action) = (# of times action led to change) / (# of times action was taken)
        """
        key = action.key
        total = self.action_counts[key]
        changes = self.action_changes[key]

        if total == 0:
            return 0.0

        return (changes + 1 ) / (total + 2) # Laplace smoothing

    def curiosity_score(
        self,
        state: str,
        action: Action
    ) -> float:
        """
        Calculate the curiosity score for a given state and action.
        """
        n:int = self.action_counts[action.key]

        # Empirical model prediction
        expected_change = self.predict_change(action)

        # Preference for insufficeintly tested actions.
        uncertainty = math.sqrt(
            math.log(self.total_actions + 2) / (n + 1)
        )

        # Avoid repeating identical actions in one state
        repeatitions = self.state_action_counts[
            (state, action.key)
        ]

        return (
            expected_change + 0.7 * uncertainty - 0.3 * repeatitions
        )


    def choose(
        self, 
        grid: np.ndarray,
        candidates: list[Action],
        rng: random.Random
    ) -> Action:
        """
        Choose the most curious action from the list of candidates.
        """
        state = state_id(grid)

        # Calculate curiosity scores for all candidates
        scores = [
            (self.curiosity_score(state, action), action)
            for action in candidates
        ]

        # Find the maximum score
        max_score = max(score for score, _ in scores)

        # Filter actions that have the maximum score
        best_actions = [
            action for score, action in scores if abs(score - max_score) < 1e-6
        ]

        # Randomly select one of the best actions
        chosen_action = rng.choice(best_actions)

        return chosen_action

    def update(
        self,
        before: np.ndarray,
        action: Action,
        after: np.ndarray
    ) -> dict:
        """
        Update the world model based on the observed transition.
        """
        previous_state = state_id(before)
        next_state = state_id(after)

        novel = next_state not in self.visited_states

        if before.shape == after.shape:
            changed_ratio = float(np.mean(before != after))

        else:
            changed_ratio = 1.0

        # Update visited states
        changed = changed_ratio > 0

        self.visited_states.add(previous_state)
        self.visited_states.add(next_state)

        self.action_counts[action.key] += 1
        self.action_changes[action.key] += int(changed)
        self.total_actions += 1
 
        self.state_action_counts[
            (previous_state, action.key)
        ] += 1

        return {
            "state_before": previous_state,
            "state_after": next_state,
            "action": action.key,
            "changed_ratio": changed_ratio,
            "new_visual_state": novel,
            "predicted_change_after_update": self.predict_change(action)
        }


# STEP 4: Random Baseline

def random_action(
    candidates: list[Action],
    rng: random.Random
) -> Action:
    """
    Sample action families uniformly so coordinate
    actions do not dominate the random baseline
    """
    families = sorted({a.kind.name for a in candidates})
    family = rng.choice(families)


    return rng.choice([
        a for a in candidates if a.kind.name == family
    ])

# STEP 5: Main loop

def run(game:str, policy:str, seed:int, steps:int):
    rng = random.Random(seed)
    model = WorldModel()

    arc = arc_agi.Arcade()
    env = arc.make(
        game,
        seed=seed,
        save_recording=True,
        render_mode="human" if policy == "curiosity" else "none"
    )

    if env is None: 
        raise ValueError(f"Failed to initialize environment for game: {game}")

    obs = env.observation_space
    if obs is None:
        obs = env.reset()

    if obs is None:
        raise ValueError("Failed to obtain initial observation.")

    output = Path("results") / f"{game}_{policy}_{seed}.json"
    output.mkdir(parents=True, exist_ok=True)

    trace_path = output / "trace.jsonl"
    executed = 0
    frames = [get_grid(obs)]
    action_history = []

    with trace_path.open("w", encoding="utf-8") as log:
        # Stopping info
        no_valid_move: bool = False
        no_observations: bool = False

        for step in range(steps):

            if obs.state in (GameState.WIN, GameState.GAME_OVER):
                print(f"Game ended at step {step}.")
                break

            before = get_grid(obs)
            candidates = legal_candidates(env, before)
            
            if not candidates:
                no_valid_move = True
                print(f"No legal candidates available at step {step}.")
                break

            if policy == "random":
                action = random_action(candidates, rng)
            else:
                action = model.choose(before, candidates, rng)

            next_obs = env.step(
                action.kind,
                data=action.payload()
            )

            if next_obs is None:
                no_observations = True
                print(f"Failed to obtain next observation at step {step}.")
                break

            after = get_grid(next_obs)

            frames.append(after.copy())
            action_history.append(action.key)

            event = model.update(
                before,action,after
            )

            event["visual_change"] = analyze_visual_change(before, after)
            event["step"] = step 

            log.write(json.dumps(event) + "\n")

            obs = next_obs
            executed += 1

            if step % 10 == 0:
                print(
                    f"Step: {step} | "
                    f"Unique states: {len(model.visited_states)}"
                )

        scorecard = arc.close_scorecard()
        np.savez_compressed(
            output / "observations.npz",
            frames=np.stack(frames),
            actions=np.asarray(action_history)
        )

        summary = {
            "game": game,
            "policy": policy,
            "seed": seed,
            "actions": executed,
            "unique_visual_states": len(model.visited_states),
            "won": obs.state == GameState.WIN,
            "levels_completed": obs.levels_completed,
            "official_score": (
                scorecard.score if scorecard is not None else None
            ),
            "stopping_conditions": {
                "no_valid_move": no_valid_move,
                "no_observations": no_observations,
                "steps_reached": executed >= steps,
                "game_ended": obs.state in (GameState.WIN, GameState.GAME_OVER)
            }
        }

        (output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--game", default="ls20")
    parser.add_argument(
        "--policy",
        choices=["random", "curiosity"],
        default="curiosity"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)

    args = parser.parse_args()

    run(
        game=args.game,
        policy=args.policy,
        seed=args.seed,
        steps=args.steps
    )
import numpy as np
import os
from kungfu_env import KUNGFU_ACTION_NAMES

# Configuration
INPUT_DIR = "recordings"
OUTPUT_DIR = "recordings_filtered"
NO_OP_INDEX = 0  # No-op action index
MAX_CONSECUTIVE_NO_OPS = 3  # Max consecutive No-ops allowed
NO_OP_KEEP_PROB = 0.1  # Keep 10% of No-op actions
MIN_FRAMES = 100  # Minimum frames to save a file
MAX_ACTION_PERCENTAGE = 0.5  # Cap any action at 50% of total
AUGMENT_ACTIONS = [1, 7, 8, 11, 12]  # Punch, Right, Kick, Crouch Kick, Crouch Punch
AUGMENT_FACTOR = 3  # Triple these actions

def filter_recording(input_path, output_path):
    """Filter a single .npz file to reduce No-op and balance actions."""
    try:
        # Load data with allow_pickle=True
        data = np.load(input_path, allow_pickle=True)
        frames = data["frames"]
        actions = data["actions"]
        rewards = data["rewards"]
        buttons = data.get("buttons", [])
        scenario = data.get("scenario", "scenario")
        timestamp = data.get("timestamp", "")
        game_speed = data.get("game_speed", 0.5)
        frame_skip = data.get("frame_skip", 2)

        # Clip invalid actions
        original_len = len(actions)
        actions = np.clip(actions, 0, len(KUNGFU_ACTION_NAMES) - 1)
        if np.any(actions != data["actions"]):
            print(f"Warning: Clipped invalid actions in {input_path}")

        # Initialize filtering
        keep_mask = np.ones(len(actions), dtype=bool)
        no_op_count = 0

        # Filter consecutive No-ops
        for i in range(len(actions)):
            if actions[i] == NO_OP_INDEX:
                no_op_count += 1
                if no_op_count > MAX_CONSECUTIVE_NO_OPS:
                    keep_mask[i] = False
                elif np.random.rand() > NO_OP_KEEP_PROB:
                    keep_mask[i] = False
            else:
                no_op_count = 0

        # Prioritize attack actions (assume attacks needed when enemies are close)
        attack_indices = np.isin(actions, [1, 8, 11, 12])  # Punch, Kick, Crouch Kick, Crouch Punch
        keep_mask[attack_indices] = True  # Keep all attack actions

        # Apply initial mask
        filtered_frames = frames[keep_mask]
        filtered_actions = actions[keep_mask]
        filtered_rewards = rewards[keep_mask]

        # Augment underrepresented actions
        aug_frames, aug_actions, aug_rewards = [], [], []
        for i in AUGMENT_ACTIONS:
            indices = np.where(filtered_actions == i)[0]
            for idx in indices:
                for _ in range(AUGMENT_FACTOR - 1):
                    aug_frames.append(filtered_frames[idx])
                    aug_actions.append(filtered_actions[idx])
                    aug_rewards.append(filtered_rewards[idx])
        if aug_frames:
            filtered_frames = np.concatenate([filtered_frames, aug_frames])
            filtered_actions = np.concatenate([filtered_actions, aug_actions])
            filtered_rewards = np.concatenate([filtered_rewards, aug_rewards])

        # Balance action distribution
        if len(filtered_actions) > 0:
            action_counts = np.bincount(filtered_actions, minlength=len(KUNGFU_ACTION_NAMES))
            total = len(filtered_actions)
            new_keep_mask = np.ones(len(filtered_actions), dtype=bool)
            for i in range(len(action_counts)):
                if action_counts[i] / total > MAX_ACTION_PERCENTAGE:
                    indices = np.where(filtered_actions == i)[0]
                    max_allowed = int(MAX_ACTION_PERCENTAGE * total)
                    if len(indices) > max_allowed:
                        np.random.shuffle(indices)
                        new_keep_mask[indices[max_allowed:]] = False
                        print(f"Capped action {KUNGFU_ACTION_NAMES[i]} from {action_counts[i]} to {max_allowed}")
            filtered_frames = filtered_frames[new_keep_mask]
            filtered_actions = filtered_actions[new_keep_mask]
            filtered_rewards = filtered_rewards[new_keep_mask]

        # Debugging output
        print(f"\nProcessing {input_path}:")
        print(f"Original: {original_len} frames")
        print("Original action distribution:")
        original_counts = np.bincount(actions, minlength=len(KUNGFU_ACTION_NAMES))
        for i, count in enumerate(original_counts):
            if count > 0:
                print(f"  {KUNGFU_ACTION_NAMES[i]}: {count} ({count/original_len*100:.2f}%)")
        
        if len(filtered_frames) < MIN_FRAMES:
            print(f"Filtered recording too short ({len(filtered_frames)} frames) - discarded")
            return False

        print(f"Filtered: {len(filtered_frames)} frames")
        print("Filtered action distribution:")
        filtered_counts = np.bincount(filtered_actions, minlength=len(KUNGFU_ACTION_NAMES))
        for i, count in enumerate(filtered_counts):
            if count > 0:
                print(f"  {KUNGFU_ACTION_NAMES[i]}: {count} ({count/len(filtered_actions)*100:.2f}%)")

        # Save filtered data
        np.savez_compressed(
            output_path,
            frames=filtered_frames,
            actions=filtered_actions,
            rewards=filtered_rewards,
            buttons=buttons,
            scenario=scenario,
            timestamp=timestamp,
            game_speed=game_speed,
            frame_skip=frame_skip
        )
        print(f"Saved filtered recording to {output_path}")
        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    processed = 0
    discarded = 0
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".npz"):
            input_path = os.path.join(INPUT_DIR, file)
            output_path = os.path.join(OUTPUT_DIR, file)
            if filter_recording(input_path, output_path):
                processed += 1
            else:
                discarded += 1
    
    print(f"\nSummary: Processed {processed} files, discarded {discarded} files")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
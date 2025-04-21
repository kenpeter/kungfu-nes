import numpy as np
import os
import argparse
from kungfu_env import KUNGFU_ACTION_NAMES

# Configuration
NO_OP_INDEX = 0  # No-op action index
MIN_FRAMES = 100  # Minimum frames to save a file
AUGMENT_ACTIONS = [1, 7, 8, 11, 12]  # Punch, Right, Kick, Crouch Kick, Crouch Punch
AUGMENT_FACTOR = 2  # Double these actions (reduced to avoid over-augmentation)

def filter_recording(input_path, output_path, target_noop_ratio=0.02):
    """Filter a single .npz file to achieve target No-op ratio and balance actions."""
    try:
        # Load data with allow_pickle=True
        data = np.load(input_path, allow_pickle=True)
        frames = data["frames"]
        actions = data["actions"]
        rewards = data.get("rewards", np.ones_like(data["actions"]))
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

        # Calculate current No-op ratio
        noop_count = np.sum(actions == NO_OP_INDEX)
        total_actions = len(actions)
        current_noop_ratio = noop_count / total_actions if total_actions > 0 else 0
        print(f"\nProcessing {input_path}:")
        print(f"Original: {original_len} frames, No-op ratio = {current_noop_ratio:.2%} ({noop_count}/{total_actions})")
        print("Original action distribution:")
        original_counts = np.bincount(actions, minlength=len(KUNGFU_ACTION_NAMES))
        for i, count in enumerate(original_counts):
            if count > 0:
                print(f"  {KUNGFU_ACTION_NAMES[i]}: {count} ({count/original_len*100:.2f}%)")

        # Target No-op count
        target_noop_count = int(total_actions * target_noop_ratio)

        # Initialize filtering
        keep_mask = np.ones(len(actions), dtype=bool)
        noop_kept = 0

        # Prioritize keeping No-op actions between active actions (strategic pauses)
        for i in range(len(actions)):
            if actions[i] == NO_OP_INDEX:
                # Check if No-op is between active actions (e.g., Punch -> No-op -> Kick)
                is_strategic = False
                if i > 0 and i < len(actions) - 1:
                    if actions[i-1] != NO_OP_INDEX and actions[i+1] != NO_OP_INDEX:
                        is_strategic = True
                if is_strategic and noop_kept < target_noop_count:
                    # Prefer strategic No-ops
                    keep_mask[i] = True
                    noop_kept += 1
                else:
                    # Randomly keep non-strategic No-ops to meet target
                    if noop_kept < target_noop_count and np.random.random() < (target_noop_count - noop_kept) / (noop_count - noop_kept + 1e-6):
                        keep_mask[i] = True
                        noop_kept += 1
                    else:
                        keep_mask[i] = False
            else:
                keep_mask[i] = True  # Keep all non-No-op actions

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

        # Verify filtered action distribution
        if len(filtered_actions) > 0:
            filtered_counts = np.bincount(filtered_actions, minlength=len(KUNGFU_ACTION_NAMES))
            new_total = len(filtered_actions)
            new_noop_count = filtered_counts[NO_OP_INDEX]
            new_noop_ratio = new_noop_count / new_total if new_total > 0 else 0
            print(f"Filtered: {new_total} frames, No-op ratio = {new_noop_ratio:.2%} ({new_noop_count}/{new_total})")
            print("Filtered action distribution:")
            for i, count in enumerate(filtered_counts):
                if count > 0:
                    print(f"  {KUNGFU_ACTION_NAMES[i]}: {count} ({count/new_total*100:.2f}%)")

            # Ensure No-op ratio is within target
            if new_noop_ratio > target_noop_ratio * 1.1:  # Allow 10% tolerance
                print(f"Warning: No-op ratio {new_noop_ratio:.2%} exceeds target {target_noop_ratio:.2%}")
                # Further reduce No-op if needed
                excess_noops = int(new_total * (new_noop_ratio - target_noop_ratio))
                noop_indices = np.where(filtered_actions == NO_OP_INDEX)[0]
                if len(noop_indices) > excess_noops:
                    np.random.shuffle(noop_indices)
                    new_keep_mask = np.ones(len(filtered_actions), dtype=bool)
                    new_keep_mask[noop_indices[:excess_noops]] = False
                    filtered_frames = filtered_frames[new_keep_mask]
                    filtered_actions = filtered_actions[new_keep_mask]
                    filtered_rewards = filtered_rewards[new_keep_mask]
                    filtered_counts = np.bincount(filtered_actions, minlength=len(KUNGFU_ACTION_NAMES))
                    new_total = len(filtered_actions)
                    new_noop_count = filtered_counts[NO_OP_INDEX]
                    new_noop_ratio = new_noop_count / new_total if new_total > 0 else 0
                    print(f"After adjustment: No-op ratio = {new_noop_ratio:.2%} ({new_noop_count}/{new_total})")

        # Check minimum frames
        if len(filtered_frames) < MIN_FRAMES:
            print(f"Filtered recording too short ({len(filtered_frames)} frames) - discarded")
            return False

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

def main(input_dir, output_dir, target_noop_ratio):
    os.makedirs(output_dir, exist_ok=True)
    
    processed = 0
    discarded = 0
    for file in os.listdir(input_dir):
        if file.endswith(".npz"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            if filter_recording(input_path, output_path, target_noop_ratio):
                processed += 1
            else:
                discarded += 1
    
    print(f"\nSummary: Processed {processed} files, discarded {discarded} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NPZ files to reduce No-op actions")
    parser.add_argument("--input_dir", default="recordings", help="Directory containing input NPZ files")
    parser.add_argument("--output_dir", default="recordings_filtered", help="Directory to save filtered NPZ files")
    parser.add_argument("--target_noop_ratio", type=float, default=0.02, help="Target ratio of No-op actions (e.g., 0.02 for 2%)")
    args = parser.parse_args()
    
    try:
        main(args.input_dir, args.output_dir, args.target_noop_ratio)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
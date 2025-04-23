import os
import numpy as np
import pygame
import time
import argparse
from pygame.locals import *


def load_recording(npz_path):
    """Safely load NPZ file with proper pickle handling"""
    try:
        # First try with allow_pickle=True for newer NumPy versions
        return np.load(npz_path, allow_pickle=True)
    except ValueError as e:
        # Fallback for older recordings
        print(f"Warning: Falling back to legacy NPZ loading for {npz_path}")
        return np.load(npz_path)


def play_recording(npz_path, speed=1.0):
    """Play back a recorded game session"""
    try:
        data = load_recording(npz_path)
        frames = data["frames"]
        actions = data["actions"]

        # Handle button names carefully
        if "buttons" in data:
            buttons = list(data["buttons"])  # Convert numpy array to list if needed
        else:
            # Default button order for Kung Fu Master
            buttons = [
                "UP",
                "DOWN",
                "LEFT",
                "RIGHT",
                "A",
                "B",
                "C",
                "X",
                "Y",
                "Z",
                "MODE",
                "START",
                "SELECT",
            ]
            print("Warning: No button names found in recording, using default")

        print(f"Playing back {npz_path}")
        print(f"Frames: {len(frames)}, Actions: {len(actions)}")
        print(f"Buttons: {buttons}")

        pygame.init()
        window = pygame.display.set_mode(
            (frames[0].shape[1] * 2, frames[0].shape[0] * 2)
        )
        pygame.display.set_caption(f"Playback: {os.path.basename(npz_path)}")

        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 18)

        running = True
        frame_idx = 0
        paused = False

        while running and frame_idx < len(frames):
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        paused = not paused
                    elif event.key == K_LEFT:
                        frame_idx = max(0, frame_idx - 10)
                    elif event.key == K_RIGHT:
                        frame_idx = min(len(frames) - 1, frame_idx + 10)
                    elif event.key == K_ESCAPE:
                        running = False

            if not paused:
                # Display frame
                frame = frames[frame_idx]
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                frame_surface = pygame.transform.scale(frame_surface, window.get_size())
                window.blit(frame_surface, (0, 0))

                # Display action info
                action = actions[frame_idx]
                action_text = " | ".join(
                    f"{btn}:{int(act)}" for btn, act in zip(buttons, action)
                )
                text_surface = font.render(
                    f"Frame: {frame_idx}/{len(frames)} | {action_text}",
                    True,
                    (255, 255, 255),
                )
                window.blit(text_surface, (10, 10))

                pygame.display.flip()
                frame_idx += 1
                clock.tick(60 * speed)
            else:
                # Show paused screen
                window.fill((0, 0, 0))
                pause_text = font.render(
                    "PAUSED (SPACE=resume, LEFT/RIGHT=seek, ESC=quit)",
                    True,
                    (255, 255, 255),
                )
                window.blit(pause_text, (10, 10))
                pygame.display.flip()
                clock.tick(60)

        pygame.quit()
    except Exception as e:
        print(f"Error during playback: {e}")
        pygame.quit()
        raise


def main():
    parser = argparse.ArgumentParser(description="Play back recorded game sessions")
    parser.add_argument("npz_path", help="Path to the NPZ recording file")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Playback speed multiplier"
    )
    args = parser.parse_args()

    if not os.path.exists(args.npz_path):
        print(f"Error: File not found - {args.npz_path}")
        return

    play_recording(args.npz_path, args.speed)


if __name__ == "__main__":
    main()

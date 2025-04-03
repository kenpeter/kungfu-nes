import keyboard  # Requires `pip install keyboard`

def capture_state(args):
    env = make_kungfu_env(render=True)
    obs = env.reset()
    done = False
    frame_time = 1/60

    print("Controls: Left/Right Arrows, Z (Punch), X (Kick), Up (Jump), Down (Duck)")
    print(f"Press 'S' to save state to '{args.state_file}', 'Q' to quit.")

    try:
        while not done:
            start_time = time.time()
            env.render()

            # Determine action
            action = 0
            if keyboard.is_pressed("left") or keyboard.is_pressed("a"):
                action = 1
            elif keyboard.is_pressed("right") or keyboard.is_pressed("d"):
                action = 2
            if keyboard.is_pressed("x"):
                action = 3 if action == 0 else action + 2
            elif keyboard.is_pressed("z"):
                action = 4 if action == 0 else action + 4
            if keyboard.is_pressed("up"):
                action = 10
            elif keyboard.is_pressed("down"):
                action = 9

            obs, reward, done, info = env.step(action)
            logging.info(f"Action: {env.action_names[action]}")

            if keyboard.is_pressed("s"):
                with open(args.state_file, "wb") as f:
                    f.write(env.unwrapped.get_state())
                print(f"State saved to '{args.state_file}'")

            if keyboard.is_pressed("q"):
                print("Quitting...")
                break

            # Maintain frame rate
            elapsed = time.time() - start_time
            time.sleep(max(0, frame_time - elapsed))

    finally:
        env.close()

import retro
env = retro.make(game="KungFu-Nes")  # Adjust to match the imported name
env.reset()
while True:
    env.step(env.action_space.sample())  # Random actions
    env.render()
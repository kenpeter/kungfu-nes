import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

# we use the projectile detection in projectile aware wrapper
from projectile_detection import enhance_observation_with_projectiles


class ProjectileAwareWrapper(gym.Wrapper):
    """A wrapper that adds projectile information to the observation space"""

    # init
    def __init__(self, env, max_projectiles=5):
        super().__init__(env)

        # Original observation space (image)
        self.image_obs_space = self.observation_space

        # Additional features for each projectile:
        # [relative_x, relative_y, velocity_x, velocity_y, size, distance, time_to_impact]
        self.projectile_features = 7

        # Maximum number of projectiles to track
        self.max_projectiles = max_projectiles

        # Create a hybrid observation space with both images and vector data
        self.observation_space = spaces.Dict(
            {
                "image": self.image_obs_space,
                "projectiles": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max_projectiles, self.projectile_features),
                    dtype=np.float32,
                ),
                "projectile_mask": spaces.Box(
                    low=0, high=1, shape=(max_projectiles,), dtype=np.int8
                ),
                "recommended_action": spaces.Box(
                    low=0,
                    high=13,  # Number of actions in Kung Fu Master
                    shape=(1,),
                    dtype=np.int32,
                ),
            }
        )

    # reset
    def reset(self, **kwargs):
        obs_result = self.env.reset(**kwargs)

        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            obs, info = obs_result
        else:
            obs = obs_result
            info = {}

        # Create empty projectile vector data
        projectile_data = np.zeros(
            (self.max_projectiles, self.projectile_features), dtype=np.float32
        )
        projectile_mask = np.zeros(self.max_projectiles, dtype=np.int8)
        recommended_action = np.zeros(1, dtype=np.int32)

        # Create enhanced observation
        enhanced_obs = {
            "image": obs,
            "projectiles": projectile_data,
            "projectile_mask": projectile_mask,
            "recommended_action": recommended_action,
        }

        return enhanced_obs, info

    # step
    def step(self, action):
        # Take step in environment
        step_result = self.env.step(action)

        # Handle different return types
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        elif len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            raise ValueError(f"Unexpected step result length: {len(step_result)}")

        # Get current player position
        player_position = (
            self.env.get_player_position()
            if hasattr(self.env, "get_player_position")
            else (0, 0)
        )

        # Initialize projectile data
        projectile_data = np.zeros(
            (self.max_projectiles, self.projectile_features), dtype=np.float32
        )
        projectile_mask = np.zeros(self.max_projectiles, dtype=np.int8)
        recommended_action = np.zeros(1, dtype=np.int32)

        # If we have enough frames, detect projectiles
        if (
            hasattr(self.env, "raw_observation_buffer")
            and len(self.env.raw_observation_buffer) >= 2
        ):
            # Convert buffer to numpy array for projectile detection
            frame_stack = np.array(self.env.raw_observation_buffer)

            # Detect projectiles from frame differences
            projectile_info = enhance_observation_with_projectiles(
                frame_stack, self.env.projectile_detector, player_position
            )
            projectiles = projectile_info["projectiles"]
            recommended_action[0] = projectile_info["recommended_action"]

            # Convert projectile info to feature vectors
            # Each projectile: [relative_x, relative_y, velocity_x, velocity_y, size, distance, time_to_impact]
            for i, proj in enumerate(projectiles[: self.max_projectiles]):
                # Extract projectile position and info
                x, y = proj["position"]
                vel_x, vel_y = proj.get("velocity", (0, 0))
                width, height = proj.get("size", (0, 0))
                size = max(width, height)

                # Calculate relative position to player
                player_x, player_y = player_position
                rel_x = x - player_x
                rel_y = y - player_y

                # Calculate distance and estimated time to impact
                distance = np.sqrt(rel_x**2 + rel_y**2)
                time_to_impact = distance / max(np.sqrt(vel_x**2 + vel_y**2), 1e-6)

                # Store features
                projectile_data[i] = [
                    rel_x,
                    rel_y,
                    vel_x,
                    vel_y,
                    size,
                    distance,
                    time_to_impact,
                ]
                projectile_mask[i] = 1  # Mark this projectile as valid

        # Create enhanced observation
        enhanced_obs = {
            "image": obs,
            "projectiles": projectile_data,
            "projectile_mask": projectile_mask,
            "recommended_action": recommended_action,
        }

        return enhanced_obs, reward, terminated, truncated, info


# this return the tire projectile aware wrapper
def wrap_projectile_aware(env, max_projectiles=5):
    """Wrap an environment to add projectile awareness"""
    # Handle VecEnv case - unwrap to get the base env
    if isinstance(env, VecEnv):
        # Temporarily disable to make changes
        env.close()

        # Get the base environment
        base_env = env.envs[0]

        # Apply our wrapper
        wrapped_env = ProjectileAwareWrapper(base_env, max_projectiles=max_projectiles)

        # Re-wrap in DummyVecEnv
        env = DummyVecEnv([lambda: wrapped_env])
    else:
        # Apply wrapper directly
        env = ProjectileAwareWrapper(env, max_projectiles=max_projectiles)

    return env

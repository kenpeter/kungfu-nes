import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Type, Union, Optional

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3 import PPO
from gymnasium import spaces


# Feature extractor for projectile awareness
class ProjectileAwareCNN(BaseFeaturesExtractor):
    """
    CNN for processing both game frames and projectile features.

    :param observation_space: Observation space
    :param features_dim: Number of features to extract
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # First call parent constructor with the provided features_dim
        super().__init__(observation_space, features_dim)

        # Extract dimensions from the image observation space
        if isinstance(observation_space, spaces.Dict):
            image_space = observation_space["image"]
            projectile_space = observation_space["projectiles"]
            n_input_channels = image_space.shape[
                0
            ]  # Number of input channels (e.g., 4 for frame stack)
            image_height, image_width = image_space.shape[1], image_space.shape[2]
            self.projectile_dim = (
                projectile_space.shape[0] * projectile_space.shape[1]
                if len(projectile_space.shape) > 1
                else projectile_space.shape[0]
            )
            print(
                f"Image space: {image_space.shape}, Projectile space: {projectile_space.shape}"
            )
            print(f"Projectile dim calculated: {self.projectile_dim}")
        else:
            # Fallback for standard observation spaces
            n_input_channels = observation_space.shape[0]
            image_height, image_width = (
                observation_space.shape[1],
                observation_space.shape[2],
            )
            self.projectile_dim = 0
            print(f"Standard observation space: {observation_space.shape}")

        # CNN for processing game frames
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output shape by doing one forward pass
        with torch.no_grad():
            try:
                # Try with regular dimensions first
                test_tensor = torch.zeros(
                    1, n_input_channels, image_height, image_width
                )
                n_flatten = self.cnn(test_tensor).shape[1]
                print(
                    f"CNN output shape determined with dimensions {image_height}x{image_width}"
                )
            except RuntimeError:
                try:
                    # If that fails, try with transposed dimensions
                    print(
                        f"Trying transposed dimensions for CNN input: {image_width}x{image_height}"
                    )
                    test_tensor = torch.zeros(
                        1, n_input_channels, image_width, image_height
                    )
                    n_flatten = self.cnn(test_tensor).shape[1]
                    print(f"CNN output shape determined with transposed dimensions")
                except RuntimeError:
                    # If both fail, use a hardcoded value that's likely to work
                    print("Could not determine CNN output shape, using estimated value")
                    n_flatten = 39936  # Common value for frame stacked observations

        print(f"CNN output features: {n_flatten}")

        # Linear layer for combining CNN features with projectile features
        # Calculate the total input size
        total_features = n_flatten + self.projectile_dim
        print(f"Total input features to linear layer: {total_features}")

        # Create the linear layer to combine features
        self.linear = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.ReLU(),
        )

        # Save dimensions for forward pass
        self.n_input_channels = n_input_channels
        self.image_height = image_height
        self.image_width = image_width
        self.n_flatten = n_flatten

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process image and projectile observations

        :param observations: Dict containing 'image' and 'projectiles' tensors
        :return: Tensor of extracted features
        """
        batch_size = observations["image"].shape[0]

        # Process image features
        try:
            # Try first with standard channel order
            try:
                # First try permuting from [batch, height, width, channels] to [batch, channels, height, width]
                image_tensor = observations["image"]
                if (
                    len(image_tensor.shape) == 4
                    and image_tensor.shape[3] < image_tensor.shape[1]
                ):
                    image_features = self.cnn(image_tensor.permute(0, 3, 1, 2) / 255.0)
                else:
                    # Already in the expected format or a different shape
                    image_features = self.cnn(image_tensor / 255.0)
            except RuntimeError as e:
                print(f"First attempt failed: {e}")
                # Try other permutations
                image_tensor = observations["image"]
                permutations = [
                    (0, 3, 1, 2),  # [batch, h, w, c] -> [batch, c, h, w]
                    None,  # No permutation, try as is
                    (0, 2, 3, 1),  # Another common permutation
                ]

                for perm in permutations:
                    try:
                        if perm is None:
                            tensor_to_try = image_tensor
                        else:
                            if len(image_tensor.shape) == 4:
                                tensor_to_try = image_tensor.permute(*perm)
                            else:
                                continue  # Skip this permutation if shape doesn't match

                        image_features = self.cnn(tensor_to_try / 255.0)
                        print(
                            f"Successfully processed image with {'no permutation' if perm is None else f'permutation {perm}'}"
                        )
                        break
                    except RuntimeError:
                        continue
                else:
                    # If all permutations fail, try reshaping
                    try:
                        # Try to reshape to expected format
                        reshaped = image_tensor.reshape(
                            batch_size,
                            self.n_input_channels,
                            self.image_height,
                            self.image_width,
                        )
                        image_features = self.cnn(reshaped / 255.0)
                    except RuntimeError as e:
                        print(f"All image processing attempts failed: {e}")
                        # Create empty features as last resort
                        image_features = torch.zeros(
                            (batch_size, self.n_flatten), device=image_tensor.device
                        )
        except Exception as e:
            print(f"Error processing image: {e}")
            image_features = torch.zeros(
                (batch_size, self.n_flatten), device=observations["image"].device
            )

        # Process projectile features
        try:
            projectile_features = observations["projectiles"]
            print(f"Original projectile features shape: {projectile_features.shape}")

            # Flatten projectile features if needed
            if len(projectile_features.shape) == 3:  # [batch, time, features]
                print(f"Reshaping projectile features from {projectile_features.shape}")
                # Flatten all dimensions except batch
                projectile_features = projectile_features.reshape(batch_size, -1)
            elif len(projectile_features.shape) == 1:  # [features]
                projectile_features = projectile_features.unsqueeze(
                    0
                )  # Add batch dimension

            print(
                f"Projectile features shape after processing: {projectile_features.shape}"
            )

        except Exception as e:
            print(f"Error processing projectiles: {e}")
            projectile_features = torch.zeros(
                (batch_size, self.projectile_dim), device=observations["image"].device
            )

        # Ensure both tensors have the same batch dimension before concatenating
        if image_features.shape[0] != projectile_features.shape[0]:
            print(
                f"Batch size mismatch: image={image_features.shape[0]}, projectile={projectile_features.shape[0]}"
            )
            if image_features.shape[0] > projectile_features.shape[0]:
                projectile_features = projectile_features.expand(
                    image_features.shape[0], -1
                )
            else:
                image_features = image_features.expand(projectile_features.shape[0], -1)

        # Check if projectile_features needs resizing to match expected dimensions
        if projectile_features.shape[1] != self.projectile_dim:
            print(
                f"Projectile feature dim mismatch: got {projectile_features.shape[1]}, expected {self.projectile_dim}"
            )
            # Resize to match expected dimensions
            if projectile_features.shape[1] < self.projectile_dim:
                # Pad with zeros if smaller
                padding = torch.zeros(
                    (batch_size, self.projectile_dim - projectile_features.shape[1]),
                    device=projectile_features.device,
                )
                projectile_features = torch.cat([projectile_features, padding], dim=1)
            else:
                # Truncate if larger
                projectile_features = projectile_features[:, : self.projectile_dim]

        # Combine features
        print(
            f"Before cat: image_features shape: {image_features.shape}, projectile_features shape: {projectile_features.shape}"
        )
        combined_features = torch.cat([image_features, projectile_features], dim=1)
        print(f"Combined features shape: {combined_features.shape}")

        # Verify the combined shape matches what our linear layer expects
        expected_input_size = self.linear[0].in_features
        actual_input_size = combined_features.shape[1]

        if expected_input_size != actual_input_size:
            print(
                f"Linear input size mismatch: got {actual_input_size}, expected {expected_input_size}"
            )
            if actual_input_size < expected_input_size:
                # Pad with zeros if too small
                padding = torch.zeros(
                    (batch_size, expected_input_size - actual_input_size),
                    device=combined_features.device,
                )
                combined_features = torch.cat([combined_features, padding], dim=1)
            else:
                # Truncate if too large
                combined_features = combined_features[:, :expected_input_size]

            print(f"Adjusted combined features shape: {combined_features.shape}")

        # Process through final layers
        return self.linear(combined_features)


# Policy that uses the ProjectileAwareCNN
class ProjectileAwarePolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        # Use our custom CNN feature extractor
        features_extractor_class = ProjectileAwareCNN
        features_extractor_kwargs = dict(features_dim=256)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs,
        )


def create_enhanced_kungfu_model(env, resume=False, model_path=None):
    """Create a custom PPO model with projectile awareness"""
    # Default model path if none provided
    if model_path is None:
        model_path = "model/kungfu_projectile_model.zip"

    # Resume from existing model if requested
    if resume and os.path.exists(model_path):
        print(f"Loading existing projectile-aware model from {model_path}")
        try:
            model = PPO.load(model_path, env=env, policy=ProjectileAwarePolicy)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model instead")

    # Create new model with custom policy
    print("Creating new projectile-aware model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Simplified initialization - compatible with older versions of stable-baselines3
    try:
        # Try with the most common parameters first
        model = PPO(
            policy=ProjectileAwarePolicy,
            env=env,
            learning_rate=0.0001,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./logs/tensorboard/",
            verbose=1,
            device=device,
        )
    except TypeError as e:
        print(f"Error with initial PPO parameters: {e}")
        # Try with even fewer parameters as a fallback
        model = PPO(
            policy=ProjectileAwarePolicy,
            env=env,
            learning_rate=0.0001,
            tensorboard_log="./logs/tensorboard/",
            verbose=1,
        )

    return model

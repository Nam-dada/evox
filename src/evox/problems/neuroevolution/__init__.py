from .torchvision_dataset import TorchvisionDataset

try:
    # optional dependency: torchvision, optax
    from .torchvision_dataset import TorchvisionDataset
except ImportError as e:
    original_erorr_msg = str(e)

    def TorchvisionDataset(*args, **kwargs):
        raise ImportError(
            f'TorchvisionDataset requires torchvision, optax but got "{original_erorr_msg}" when importing'
        )


try:
    # optional dependency: gym
    from .gym import Gym
    from .gym_mo import Gym_mo
except ImportError as e:
    original_erorr_msg = str(e)

    def Gym(*args, **kwargs):
        raise ImportError(
            f'Gym requires gym, ray but got "{original_erorr_msg}" when importing'
        )
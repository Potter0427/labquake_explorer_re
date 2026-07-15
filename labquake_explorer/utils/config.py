"""Configuration settings for Labquake Explorer"""
from dataclasses import dataclass

@dataclass
class LabquakeExplorerConfig:
    """Configuration settings for the application"""
    WINDOW_GAP: int = 100
    WINDOW_WIDTH: int = 300
    WINDOW_TITLE: str = "Labquake Explorer"
    MAX_ARRAY_DISPLAY: int = 1000
    DEFAULT_WINDOW_SIZE: float = 5.0
    FILE_TYPES: tuple = (
        ("HDF5 files", "*.h5 *.hdf5"),
        ("NPZ files", "*.npz"),
        ("All files", "*.*")
    )

    # Eddy current sensor positions (mm along fault)
    EDDY_POSITIONS_8CH_MM: tuple = (31, 93, 155, 217, 279, 341, 403, 465)
    EDDY_POSITIONS_5CH_MM: tuple = (50, 150, 250, 350, 450)

    # Standard color palette (Matplotlib tab10 first 8 colors)
    EDDY_COLORS: tuple = (
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
        'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray'
    )

    @classmethod
    def get_eddy_positions(cls, n_channels: int) -> list:
        """
        Get physical positions in mm for Eddy sensors along the fault.
        Ensures backward compatibility for legacy 5-channel data while
        supporting new 8-channel experiments.
        """
        if n_channels == 5:
            return list(cls.EDDY_POSITIONS_5CH_MM)
        elif n_channels == 8:
            return list(cls.EDDY_POSITIONS_8CH_MM)
        else:
            # Fallback linear spacing using 31mm + i * 62mm
            return [31 + 62 * i for i in range(n_channels)]
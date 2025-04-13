import os
from pathlib import Path
from typing import Union, Optional

def set_working_dir(directory: Union[str, Path] = os.getcwd(), subdir: str | None = None) -> Path:
    """
    Sets the working directory and returns the new path.
    
    Args:
        directory (Union[str, Path]): Directory to change to
        
    Returns:
        Path: Path object pointing to the new working directory
    """

    if subdir:
        working_dir = Path(directory) / subdir
    else:
        working_dir = Path(directory)

    working_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(working_dir)

    return working_dir.absolute()
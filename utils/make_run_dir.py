import datetime
import pathlib


def make_run_dir(mode: str, base: str = "runs") -> str:
    """
    Create a unique run directory like:
      runs/train_15_08_2025_19_30

    Args:
        mode: "train" | "valid" | "test" | etc.
        base: parent directory under which runs are saved.

    Returns:
        Absolute string path to the run directory.
    """
    ts = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
    run_name = f"{mode}_{ts}"
    run_dir = pathlib.Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)

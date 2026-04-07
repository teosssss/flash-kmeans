from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal


REPO_REMOTE_PATH = "/root/flash-kmeans"
DEFAULT_BENCH_OUT_DIR = "artifacts/cuda_vs_triton"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio",
        "python -m pip install triton numpy tqdm ninja",
    )
    .add_local_dir(
        ".",
        remote_path=REPO_REMOTE_PATH,
        copy=True,
        ignore=[".git", "__pycache__", "*.pyc", ".pytest_cache"],
    )
    .workdir(REPO_REMOTE_PATH)
    .run_commands("python -m pip install -e .")
)

app = modal.App("flash-kmeans-tests")


@app.function(image=image, gpu="L4", timeout=60 * 60)
def run_tests(command: str = "python examples/testapi.py") -> str:
    completed = subprocess.run(
        command,
        shell=True,
        cwd=REPO_REMOTE_PATH,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}\n{completed.stdout}")
    return completed.stdout


@app.function(image=image, gpu="L4", timeout=60 * 60)
def run_benchmark(
    command: str = f"python examples/benchmark_cuda_vs_triton.py --out-dir {DEFAULT_BENCH_OUT_DIR}",
    artifact_dir: str = DEFAULT_BENCH_OUT_DIR,
) -> dict[str, str]:
    completed = subprocess.run(
        command,
        shell=True,
        cwd=REPO_REMOTE_PATH,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}\n{completed.stdout}")

    remote_artifact_dir = Path(REPO_REMOTE_PATH) / artifact_dir
    artifacts: dict[str, str] = {"stdout.txt": completed.stdout}
    if remote_artifact_dir.exists():
        for path in sorted(remote_artifact_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(remote_artifact_dir).as_posix()
                artifacts[rel] = path.read_text()

    return artifacts


@app.local_entrypoint()
def main(
    command: str = "python examples/testapi.py",
    bench: bool = False,
    artifact_dir: str = DEFAULT_BENCH_OUT_DIR,
):
    if not bench:
        print(run_tests.remote(command=command))
        return

    artifacts = run_benchmark.remote(command=command, artifact_dir=artifact_dir)
    local_dir = Path(artifact_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, content in artifacts.items():
        target = local_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    summary = {
        "artifact_dir": str(local_dir),
        "files": sorted(artifacts.keys()),
    }
    print(json.dumps(summary, indent=2))

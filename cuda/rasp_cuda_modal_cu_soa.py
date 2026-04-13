import os
import pathlib
import subprocess

import modal

app = modal.App("rasp-cuda-benchmark-cu-soa")

DATA_VOL = modal.Volume.from_name("rasp-bench-data", create_if_missing=True)
HERE = pathlib.Path(__file__).resolve().parent
CU_FILE = HERE / "rasp_cuda_benchmark_soa.cu"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.2.0-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install("build-essential", "libboost-dev")
    .add_local_file(CU_FILE, "/workspace/rasp_cuda_benchmark_soa.cu", copy=True)
    .run_commands(
        "nvcc -O3 -std=c++17 -lineinfo -Xptxas=-v "
        "-gencode arch=compute_75,code=sm_75 "
        "-gencode arch=compute_75,code=compute_75 "
        "-gencode arch=compute_86,code=sm_86 "
        "-gencode arch=compute_86,code=compute_86 "
        "-gencode arch=compute_89,code=sm_89 "
        "-gencode arch=compute_89,code=compute_89 "
        "-gencode arch=compute_90,code=sm_90 "
        "-gencode arch=compute_90,code=compute_90 "
        "-gencode arch=compute_100,code=sm_100 "
        "-gencode arch=compute_100,code=compute_100 "
        "/workspace/rasp_cuda_benchmark_soa.cu -o /workspace/rasp_cuda_benchmark_soa"
    )
)


def upload_rasp_local(local_path: str, remote_name: str = "rasp_vms.bin") -> None:
    src = pathlib.Path(local_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    with DATA_VOL.batch_upload(force=True) as batch:
        batch.put_file(src, remote_path=f"/{remote_name}")
    print(f"Uploaded {src} -> volume://rasp-bench-data/{remote_name}")


@app.function(
    gpu="B200+:8",
    timeout=10 * 60 * 6,
    image=image,
    volumes={"/data": DATA_VOL},
)
def run_benchmark_remote() -> list[str]:
    if not os.path.exists("/data/rasp_vms.bin"):
        raise FileNotFoundError("Missing /data/rasp_vms.bin in Modal volume")

    subprocess.run(
        [
            "bash",
            "-lc",
            "echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES && "
            "nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader"
        ],
        check=False,
    )

    tuple_lines: list[str] = []
    proc = subprocess.Popen(
        ["/workspace/rasp_cuda_benchmark_soa", "/data/rasp_vms.bin"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        stripped = line.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            tuple_lines.append(stripped)

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"rasp_cuda_benchmark_soa exited with code {rc}")

    return tuple_lines


@app.local_entrypoint()
def main(rasp_file: str = "rasp_vms.bin"):
    upload_rasp_local(rasp_file)
    lines = run_benchmark_remote.remote()
    for line in lines:
        print(line)

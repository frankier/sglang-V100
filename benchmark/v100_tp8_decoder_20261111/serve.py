import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def request(path, data=None):
    payload = None if data is None else json.dumps(data).encode()
    with urllib.request.urlopen(
        urllib.request.Request(
            "http://127.0.0.1:18380" + path,
            data=payload,
            headers={"Content-Type": "application/json"},
        ),
        timeout=900,
    ) as response:
        return json.load(response)


for mode in ("baseline", "specialized"):
    tp = 8
    plan = json.loads(Path(f"tp{tp}-plan.json").read_text())
    command, env = plan["argv"], plan["env"]
    process_env = os.environ.copy()
    for key, value in env.items():
        # Reuse existing warm JIT caches from the tested serving environment.
        process_env["APPTAINERENV_" + key] = str(value)
    process_env["APPTAINERENV_TMPDIR"] = os.environ["TMPDIR"]
    for suffix in ("responses", "throughput", "capacity", "server-info"):
        Path(f"tp{tp}-{suffix}.json").unlink(missing_ok=True)
    if tp == 4:
        python_index = command.index("/opt/venv/bin/python")
        test_command = command[:python_index] + [
            "/opt/venv/bin/python",
            str(Path.cwd() / "test/manual/attention/test_sm70_qsa_tp.py"),
        ]
        subprocess.run(test_command, env=process_env, check=True)
    process_env["APPTAINER_BIND"] = os.environ["APPTAINER_BIND"]
    if mode == "baseline":
        process_env["APPTAINER_BIND"] = process_env["APPTAINER_BIND"].replace(
            f"{Path.cwd()}/python/sglang/srt/layers/moe/moe_runner/marlin.py:",
            f"{Path.cwd()}/image-marlin.py:",
        )
    verify = command[: command.index("/opt/venv/bin/python")] + [
        "/opt/venv/bin/python",
        "-c",
        "import hashlib; from pathlib import Path; p=Path('/opt/sglang/python/sglang/srt/layers/moe/moe_runner/marlin.py'); print(hashlib.sha256(p.read_bytes()).hexdigest())",
    ]
    bound_hash = (
        subprocess.check_output(verify, env=process_env, text=True)
        .strip()
        .splitlines()[-1]
    )
    expected = (
        Path("image-marlin.py")
        if mode == "baseline"
        else Path("python/sglang/srt/layers/moe/moe_runner/marlin.py")
    )
    assert bound_hash == hashlib.sha256(expected.read_bytes()).hexdigest(), (
        mode,
        bound_hash,
    )
    print(f"Verified {mode} runner SHA256 {bound_hash}", flush=True)
    log = open(f"{mode}-server.log", "w")
    print(
        f"Starting TP{tp} ({mode}) full checkpoint + MTP at {time.ctime()}", flush=True
    )
    proc = subprocess.Popen(
        command,
        env=process_env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"TP{tp} server exited {proc.returncode}; see tp{tp}-server.log"
                )
            try:
                models = request("/v1/models")
                break
            except Exception:
                time.sleep(10)
        else:
            raise RuntimeError("Server readiness timeout")
        print(f"TP{tp} READY {models}", flush=True)

        def generate(i):
            return request(
                "/v1/chat/completions",
                {
                    "model": "qwen",
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is 2 + 2? Answer with the number only.",
                        }
                    ],
                    "max_tokens": 64,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            answers = list(executor.map(generate, range(4)))
        Path(f"tp{tp}-responses.json").write_text(json.dumps(answers, indent=2))
        for answer in answers:
            content = answer["choices"][0]["message"]["content"]
            assert "4" in content, content
        measurements = []
        for clients in (1, 4):

            def bench(i):
                return request(
                    "/v1/chat/completions",
                    {
                        "model": "qwen",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Explain how a computer executes a Python program, in detail.",
                            }
                        ],
                        "max_tokens": 256,
                        "ignore_eos": True,
                        "temperature": 0,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )

            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as executor:
                outputs = list(executor.map(bench, range(clients)))
            elapsed = time.monotonic() - started
            tokens = sum(answer["usage"]["completion_tokens"] for answer in outputs)
            measurements.append(
                {
                    "clients": clients,
                    "completion_tokens": tokens,
                    "seconds": elapsed,
                    "aggregate_tokens_per_second": tokens / elapsed,
                    "responses": outputs,
                }
            )
            print(
                f"TP{tp} clients={clients} {tokens / elapsed:.2f} output tok/s ({tokens} tokens)",
                flush=True,
            )
        Path(f"tp{tp}-throughput.json").write_text(json.dumps(measurements, indent=2))
        if mode == "baseline":
            Path("baseline-throughput.json").write_text(
                json.dumps(measurements, indent=2)
            )
            continue
        # Distinct token streams prevent shared-prefix cache reuse between
        # clients. This is a capacity/admission test, not a quality benchmark.
        clients = 2 if tp == 8 else 1

        def capacity(i):
            return request(
                "/generate",
                {
                    "input_ids": [42 + i] * 262016,
                    "sampling_params": {
                        "max_new_tokens": 64,
                        "ignore_eos": True,
                        "temperature": 0,
                    },
                },
            )

        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as executor:
            outputs = list(executor.map(capacity, range(clients)))
        for output in outputs:
            assert output["meta_info"]["prompt_tokens"] == 262016, output["meta_info"]
            assert output["meta_info"]["completion_tokens"] == 64, output["meta_info"]
        Path(f"tp{tp}-capacity.json").write_text(
            json.dumps(
                {
                    "seconds": time.monotonic() - started,
                    "clients": clients,
                    "responses": outputs,
                },
                indent=2,
            )
        )
        print(
            f"PASS TP{tp} {clients} independent clients x 262016 prompt tokens + 64 output tokens",
            flush=True,
        )
        Path(f"tp{tp}-server-info.json").write_text(
            json.dumps(request("/get_server_info"), indent=2)
        )
        print(f"PASS TP{tp} full checkpoint, MTP, four concurrent requests", flush=True)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()

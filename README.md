# llama.cpp fork: tiered experts (for Wren_T3 / Wren_T3-v2)

This fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) adds **two expert groups per MoE layer**, each with its
own quantization type. The routing is unchanged. The fork is needed to run the per-expert mixed-precision GGUFs
of **Wren**: `Wren_T3.gguf`, `Wren_T3-v2.gguf` and `Wren_T4-*.gguf` from
[ohmysimo/Wren-GGUF](https://huggingface.co/ohmysimo/Wren-GGUF). (`Wren_Q4_K_M.gguf` and `Wren_T1.gguf` also run on
upstream llama.cpp.)

Wren is a 50%-expert-pruned and re-healed Swift1.5-Qwen3.8-Flash-Next. **Wren_T3-v2** is the recommended file:
- size: 60.4 GB;
- quiz: 83.0, against 80.7 for T3 on the same backend;
- perplexity: 7% lower than T3.

The model card has the details.

> Use the **`tiered-experts` branch** of this repository (the default branch). It contains the expert-tier support,
> the CUDA/HIP fix for an expert repeated within a token, and the RDNA2 flash-attention fix.

## Hardware

The setup this was tested on: one 12 GB GPU + 64 GB RAM.
- The routed experts and the n-gram table stay in system RAM, about 56 GB.
- Everything else goes on the GPU.
- With a q8_0 KV cache, the full 262,144-token context needs about 10 GB of VRAM.

CPU-only also works, but slowly (about 5 tokens/s).

## 1. Build

```bash
git clone -b tiered-experts https://github.com/OhMySimo/llama.cpp llama-tiered
cd llama-tiered
```

Pick **one** backend:

```bash
# NVIDIA (CUDA)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# AMD (ROCm/HIP). Set GPU_TARGETS to your GPU: gfx1030 = RX 6000 series, gfx1100 = RX 7900, ...
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# Any GPU (Vulkan). At run time add --no-op-offload (see below)
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# CPU only
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

The binaries end up in `build/bin/`.

## 2. Download the model

```bash
pip install -U huggingface_hub
hf download ohmysimo/Wren-GGUF Wren_T3-v2.gguf --local-dir models
```

## 3. Run

This starts an OpenAI-compatible server with a web UI at http://127.0.0.1:8080:

```bash
build/bin/llama-server -m models/Wren_T3-v2.gguf \
  -ngl 99 -ot "exps=CPU,per_layer_token_embd=CPU" -fa on \
  -c 262144 -ctk q8_0 -ctv q8_0 --jinja --reasoning-effort xhigh \
  --temp 1.0 --top-p 0.95 --top-k 20 --cache-ram 2048 --port 8080
```

What the flags do:

| Flag | Why |
|---|---|
| `-ngl 99 -ot "exps=CPU,per_layer_token_embd=CPU"` | Keeps the experts and the n-gram table in RAM and everything else on the GPU. Pass the overrides as **one comma-separated `-ot`**: with repeated `-ot` flags only the last one is applied. |
| `-c 262144 -ctk q8_0 -ctv q8_0` | Full native context with a q8_0 KV cache, about 10 GB of VRAM. Lower `-c` if you have less VRAM. |
| `--jinja --reasoning-effort xhigh` | The model's chat template, with reasoning at **xhigh**: the mode in which Swift was trained to reason concisely. |
| `--temp 1.0 --top-p 0.95 --top-k 20` | Recommended sampling. |
| `--cache-ram 2048` | Keep the prompt cache small on 64 GB machines. A large one on top of a 60 GB model pushes the system into swap and generation drops to almost zero. |

Backend notes:
- **Vulkan:** add `--no-op-offload`. Without it the expert matmuls offloaded to the GPU produce NaN.
- **AMD RDNA2 (RX 6000) with ROCm:** `-fa on` needs this branch (commit `daaf72c` or later).
- **CPU only:** drop `-ngl` and `-ot`.

For a quick test in the terminal, use the same flags with `build/bin/llama-cli -m models/Wren_T3-v2.gguf ...`.

**Measured speed** on an RX 6700 XT 12 GB + i9-14900KS + 64 GB RAM (ROCm):
- about 11 tokens/s generation;
- 234 tokens/s prefill.

## Troubleshooting

- **`unknown model architecture` or tensor-type errors:** you are not on the `tiered-experts` branch, or the build is
  older than the fork.
- **Very slow generation or a frozen machine:** RAM is full. Close other programs, keep `--cache-ram` small and use
  `-ot` exactly as above.
- **Garbage or NaN output on Vulkan:** add `--no-op-offload`.

---

*The original llama.cpp README follows.*

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

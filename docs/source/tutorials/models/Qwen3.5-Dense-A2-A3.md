# Qwen3.5-Dense on Atlas A2/A3

## 1 Introduction

This page provides an Atlas A2/A3 reference configuration for the dense `Qwen3.5-2B`, `Qwen3.5-4B`, and `Qwen3.5-9B` models. These models combine GDN linear-attention layers with full-attention layers and use the same serving pattern on A2/A3 after the model path and memory limits are adjusted.

For Atlas 300I DUO and Atlas 200I Pro, continue to use the platform-specific [Qwen3.5-Dense guide](Qwen3.5-Dense.md). The 310P image, checkpoint format, precision, and graph settings in that guide should not be reused unchanged on A2/A3.

Please also check the [supported-model matrix](../../user_guide/support_matrix/supported_models.md) for the feature set supported by the selected release.

## 2 Prerequisites

Use the latest compatible vLLM and vLLM Ascend release. Select the standard image for Atlas A2 and the matching `-a3` image for Atlas A3:

```bash
# Atlas A2
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

# Atlas A3
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
```

Use the BF16 Qwen3.5 checkpoints:

| Model | ModelScope path |
|-------|-----------------|
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` |

## 3 Start the container

Follow the general [prebuilt-image installation guide](../../getting_started/installation.md#installation-prebuilt-image) and map only the NPU devices allocated to the deployment. The following example exposes one NPU for a TP=1 baseline; add more `/dev/davinciN` mappings when increasing tensor parallelism.

```bash
docker run --rm \
    --name vllm-ascend \
    --shm-size=16g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

Verify device visibility inside the container before starting the service:

```bash
npu-smi info
```

## 4 Start the service

The following command is a conservative BF16 baseline for `Qwen3.5-9B`. The 2B and 4B models use the same pattern after changing `MODEL_PATH` and tuning the memory-related arguments.

```bash
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

export MODEL_PATH=Qwen/Qwen3.5-9B

vllm serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --served-model-name qwen3.5 \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --mamba-ssm-cache-dtype bfloat16 \
    --dtype bfloat16 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}'
```

Tune `--max-model-len`, `--max-num-batched-tokens`, `--max-num-seqs`, and `--gpu-memory-utilization` together according to the available HBM and workload. Increase `--tensor-parallel-size` only when the model dimensions and visible-device count support the requested degree.

### Optional MTP speculative decoding

MTP is not required for normal serving. Add the following option only when the checkpoint actually contains the Qwen3.5 MTP head under `mtp.*`:

```bash
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
```

Some fine-tuned or merged checkpoints contain only the language-model backbone. Enabling `qwen3_5_mtp` for such a checkpoint causes the draft model to look for tensors that are not present.

## 5 Verify the OpenAI-compatible API

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5",
        "messages": [{"role": "user", "content": "The future of AI is"}],
        "max_completion_tokens": 128,
        "temperature": 0
    }'
```

The expected result is HTTP 200 with a non-empty `choices` field. For common environment and container issues, refer to the [FAQs](../../faqs.md).

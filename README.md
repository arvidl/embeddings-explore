# embeddings-explore

2025-09-09

[Arvid Lundervold](https://www4.uib.no/en/find-employees/Arvid.Lundervold)

Exploration of *tokenization*, *embeddings*, and (cosine) *similarity* of **Colors**, the **Likert scale scores**, and **Neurological diagnoses** using `google/embeddinggemma-300m`

- https://huggingface.co/blog/embeddinggemma
- https://huggingface.co/google/embeddinggemma-300m
- https://ai.google.dev/gemma/docs/embeddinggemma

  
```bash
ollama pull embeddinggemma:300m
```

From https://ai.google.dev/gemma/docs/embeddinggemma:

> EmbeddingGemma is a 308M parameter multilingual text embedding model based on Gemma 3. It is optimized for use in everyday devices, such as phones, laptops, and tablets. The model produces numerical representations of text to be used for downstream tasks like information retrieval, semantic similarity search, classification, and clustering.
>
> EmbeddingGemma includes the following key features:
>
> - Multilingual support: Wide linguistic data understanding, trained in over 100 languages.
> - Flexible output dimensions: Customize your output dimensions from 768 to 128 for speed and storage tradeoffs using Matryoshka Representation Learning (MRL).
> - 2K token context: Substantial input context for processing text data and documents directly on your hardware.
> - Storage efficient: Run it on less than 200MB of RAM with quantization
> - Low latency: Generative embeddings in less than 22ms on EdgeTPU for fast and fluid applications.
> - Offline and secure: Generate embeddings of documents directly on your hardware, which works without an internet connection to keep sensitive data secure.

### Included functionality for running `gpt-oss-120b`

i.e. 

`MODEL_PATH = f"{os.environ['HOME']}/models/gpt-oss-120b/gpt-oss-120b-mxfp4.gguf"`

using `llama-cpp-python` (enabling MacOS Metal by deafult)
  - https://github.com/abetlen/llama-cpp-python)
  - https://llama-cpp-python.readthedocs.io/en

-----

# Multi-Token Prediction

> Please explain to me Muti-Token Prediction (e.g., gemma4:31b-coding-mtp-bf16)
>
> **macOS**<br>
> Ollama on macOS now supports native MTP support for Gemma 4 via [MLX](https://c.vialoops.com/CL0/https:%2F%2Follama.com%2Fblog%2Fmlx/1/0100019e059a38c0-9649d427-5e79-440e-85be-9bd9a500702a-000000/g1UnwPPt8Jc0is0u_avNjhVxKqWs7oP-Iu1Z6Yp69QU=452):
> 
> `ollama run gemma4:31b-coding-mtp-bf16`
> 
> For more information, please visit Ollama's [Gemma 4 model page](https://c.vialoops.com/CL0/https:%2F%2Follama.com%2Flibrary%2Fgemma4/2/0100019e059a38c0-9649d427-5e79-440e-85be-9bd9a500702a-000000/HloOl0xnm1Cm2gRctDzbPZSh3Q3jZpo-cXsUHndq4rI=452). 

## Core idea of Multi-Token Prediction (MTP)

Multi-Token Prediction in Gemma 4 is an inference-time speedup technique where a **small “drafter” model proposes several future tokens at once**, and the large “target” model (e.g. Gemma 4 31B) **verifies them in parallel** instead of generating one token at a time.  This is a particular implementation of speculative decoding, tuned for Gemma 4. [blog](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)

In practice, this can yield **up to roughly 2–3× higher tokens/second** for the same target model, while keeping the *exact same* output distribution as if you had run the big model alone step‑by‑step. [huggingface](https://huggingface.co/google/gemma-4-31B-it-assistant/blob/main/README.md)

***

## How it works step by step

1. **Two models in the pipeline**  
   - Target: the full Gemma 4 model, e.g. `gemma4:31b-coding`. [ai.google](https://ai.google.dev/gemma/docs/mtp/mtp)
   - Drafter (MTP head): a small assistant model, e.g. `gemma-4-31B-it-assistant`, that sits “on top” of the target’s last-layer activations and shared embeddings. [ai.google](https://ai.google.dev/gemma/docs/mtp/overview)

2. **Drafting multiple tokens**  
   - Given the current prefix, the drafter autoregressively predicts a sequence of \(N\) next tokens (say 4–16) much faster than the 31B model could. [scannn](https://scannn.com/multi-token-prediction-in-gemma-4/)
   - This is the “multi-token prediction”: the drafter is trained to generate a short continuation, not just one token. [arxiv](https://arxiv.org/abs/2404.19737)

3. **Parallel verification**  
   - The entire draft sequence is then fed to the target model, which computes a **single forward pass** over those positions. [ai.google](https://ai.google.dev/gemma/docs/mtp/mtp)
   - For each drafted token position, the target model’s distribution is compared against the drafter’s proposal; high‑probability matches are accepted, low‑probability ones are rejected. [ai.google](https://ai.google.dev/gemma/docs/mtp/overview)

4. **Guaranteeing correctness**  
   - If the target rejects a token, it *still* outputs its own token for that time step (so you never “lose” a step or sacrifice correctness). [tildes](https://tildes.net/~comp/1u1m/multi_token_prediction_mtp_with_gemma_4)
   - Because the target always runs at those positions, the final accepted sequence is exactly what you would have gotten from plain autoregressive decoding with the large model alone. [huggingface](https://huggingface.co/google/gemma-4-31B-it-assistant/blob/main/README.md)

5. **Speedup intuition**  
   - Normally, to produce \(N\) tokens you need \(N\) forward passes of the big model.  
   - With MTP, the drafter does those \(N\) steps cheaply, and the big model does **one** pass to check them (plus to add one new token of its own). [marktechpost](https://www.marktechpost.com/2026/05/06/google-ai-releases-multi-token-prediction-mtp-drafters-for-gemma-4-delivering-up-to-3x-faster-inference-without-quality-loss/?amp)
   - This breaks the usual one-token-at-a-time latency bottleneck and yields ~2–3× decoding speedup on supported runtimes (LiteRT-LM, vLLM, HF, MLX, etc.). [app.daily](https://app.daily.dev/posts/multi-token-prediction-in-gemma-4-p8wqk64sp)

***

## What “gemma4:31b-coding-mtp-bf16” means

- **`31b`**: 31 billion parameter target model (dense LLM, not the drafter size). [blog](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
- **`coding`**: instruction-tuned for code / programming use cases. [news.ycombinator](https://news.ycombinator.com/item?id=48024540)
- **`mtp`**: the model is packaged with an MTP drafter checkpoint and decoding configuration so that speculative multi-token prediction is enabled in the serving stack. [news.ycombinator](https://news.ycombinator.com/item?id=48024540)
- **`bf16`**: weights stored/served in bfloat16 for memory and throughput efficiency. [news.ycombinator](https://news.ycombinator.com/item?id=48024540)

In frameworks like Ollama or HF Transformers, this tag typically means: “load the 31B target together with its associated MTP drafter and use the MTP decoding pipeline by default.” [ai.google](https://ai.google.dev/gemma/docs/mtp/mtp)

***

## Training and architecture notes (at a high level)

- The drafter is **not a completely separate model**; in Gemma 4 it reuses the target’s input embeddings and the last-layer activations, plus a small stack of its own layers and an extra output head. [huggingface](https://huggingface.co/google/gemma-4-31B-it-assistant/blob/main/README.md)
- It is trained to predict several future tokens jointly, which empirically improves both sample efficiency and draft quality vs. simple one-step distillation. [arxiv](https://arxiv.org/abs/2404.19737)
- For dense models, verifying multiple drafted positions adds relatively little overhead, since the same weights are applied across all positions in a single pass. [scannn](https://scannn.com/multi-token-prediction-in-gemma-4/)

***

## When to use it (and when not)

**Good fits**  
- Interactive coding/assistant scenarios where *latency* matters more than slightly higher VRAM usage (you must load both target and drafter). [app.daily](https://app.daily.dev/posts/multi-token-prediction-in-gemma-4-p8wqk64sp)
- On-device or constrained environments where getting more tokens/s at the same quality is crucial. [reddit](https://www.reddit.com/r/LocalLLaMA/comments/1t4jq6h/gemma_4_mtp_released/)

**Trade-offs**  
- Slight VRAM increase (to hold the drafter) for a substantial wall‑clock speedup. [tildes](https://tildes.net/~comp/1u1m/multi_token_prediction_mtp_with_gemma_4)
- Best suited where the runtime (vLLM, HF, etc.) has explicit support for MTP/speculative decoding and for these specific Gemma 4 MTP checkpoints. [forums.developer.nvidia](https://forums.developer.nvidia.com/t/gemma-4-mtp/369123)

***

To connect this to your context: conceptually, MTP here is very close to a lightweight, activation-sharing speculative decoder head attached to your 31B target, not a fundamentally different LM; the key novelty is the tight integration and training so that the drafted runs are good enough that the verifier almost always accepts several tokens per pass.

What part would you like to dive into next: the **mathematical training objective for predicting multiple future tokens**, or the **systems side (how this plays with KV caches, batching, and runtimes like vLLM/HF)**?

# Using ollama and gemma4:31b-coding-mtp-bf16 on MacOS

With `ollama run gemma4:31b-coding-mtp-bf16` you’re starting an interactive, **multimodal** Gemma 4 31B session (text + image in, text out), with MTP giving you decoding speedups under the hood. [ollama](https://ollama.com/library/gemma4:31b-coding-mtp-bf16)

Below are concrete usage patterns from the CLI and via the local HTTP API.

***

## 1. Simple text-only coding use

After:

```bash
ollama pull gemma4:31b-coding-mtp-bf16
ollama run gemma4:31b-coding-mtp-bf16
```

You can just type:

```text
Write a Python function that loads a 3D NIfTI brain MRI with nibabel,
normalizes it to zero mean and unit variance, and returns a NumPy array.
Explain each step briefly.
```

The model will respond as a coding‑tuned assistant; MTP is transparent here (no special syntax needed). [ollama](https://ollama.com/library/gemma4:31b-coding-mtp-bf16)

***

## 2. Multimodal from the CLI (image + text)

Gemma 4 31B on Ollama supports **image input**.  The generic Ollama multimodal pattern is: [ollama](https://ollama.com/library/gemma4/tags)

```bash
ollama run gemma4:31b-coding-mtp-bf16 \
  --image /path/to/figure.png \
  "This is a sagittal T1‑weighted brain MRI. 
   Describe any visible artifacts and suggest preprocessing steps
   for a cortical thickness pipeline."
```

Notes:  
- `--image path` attaches the image; put the image **before** the text prompt for best results. [youtube](https://www.youtube.com/watch?v=zyirOtgICZs)
- You can pass multiple `--image` flags if you want to compare images. [gemma4-ai](https://gemma4-ai.com/blog/gemma4-multimodal)

Another example, more code‑centric:

```bash
ollama run gemma4:31b-coding-mtp-bf16 \
  --image ./screenshot_of_error.png \
  "This is a screenshot of a Python stack trace from a Jupyter notebook.
   Extract the error message and propose a fix."
```

***

## 3. Multimodal via the local HTTP API (Python)

Ollama exposes a local HTTP endpoint; images are sent as **base64 strings** in the `images` field. [datacamp](https://www.datacamp.com/tutorial/gemma-4-tutorial)

```python
import base64
import json
import requests

# Encode image
with open("brain_slice.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "gemma4:31b-coding-mtp-bf16",
    "prompt": (
        "You see an axial brain MR image. "
        "Classify whether this looks like T1, T2, or FLAIR, and explain why. "
        "Then suggest Python code using MONAI to load and minimally preprocess "
        "a dataset of similar images."
    ),
    "images": [img_b64],
}

resp = requests.post("http://localhost:11434/api/generate", json=payload, stream=False)
print(resp.json()["response"])
```

Same pattern for any other coding + image task (e.g., diagram-to-code, reading plots, OCR of console logs). [gemma4-ai](https://gemma4-ai.com/blog/gemma4-multimodal)

***

## 4. A few concrete multimodal prompt ideas (CLI-style)

Just to give you a feel for what’s reasonable to ask:

- **Diagram → code**  
  ```bash
  ollama run gemma4:31b-coding-mtp-bf16 \
    --image ./cnn_diagram.jpg \
    "Translate this architecture into PyTorch code using nn.Module. 
     Include conv, pooling, and fully connected layers as shown."
  ```

- **LaTeX / paper screenshot → explanation + code**  
  ```bash
  ollama run gemma4:31b-coding-mtp-bf16 \
    --image ./paper_equation.png \
    "This is an equation from a diffusion MRI paper. 
     Explain it in plain language and show Python code 
     that numerically computes the right-hand side."
  ```

- **Notebook cell screenshot → debugging**  
  ```bash
  ollama run gemma4:31b-coding-mtp-bf16 \
    --image ./notebook_error.png \
    "Read the error from this screenshot and suggest how to fix the code. 
     Provide a corrected cell."
  ```

All of these work the same way: `--image` + a clear text instruction; the MTP tag just makes generation faster. [ollama](https://ollama.com/library/gemma4:31b-coding-mtp-bf16)

***

Given your interests, what kind of multimodal coding workflow are you most likely to try first (e.g., “explain MRI figures”, “debug code from screenshots”, or “transcribe equations/plots into code”)?

Here are some concrete “diagram → code” examples you can run directly with:

```bash
ollama run gemma4:31b-coding-mtp-bf16
```

or via the `--image` flag.

***

## 1. CNN diagram → PyTorch model

Assume you have a whiteboard/photo of a CNN architecture (Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Softmax) in `cnn_diagram.png`.

```bash
ollama run gemma4:31b-coding-mtp-bf16 \
  --image ./cnn_diagram.png \
  "This diagram shows a convolutional neural network.
   1) Describe the architecture in words (layers, filter sizes, activations).
   2) Implement the network in PyTorch as a nn.Module called SimpleCNN
      that matches the diagram as closely as possible.
   3) Add a forward() method and a short example of how to instantiate
      and run a batch of dummy data through the network."
```

This uses the image as the “spec,” and your text prompt makes the task: interpret then implement.

***

## 2. U‑Net style architecture sketch → PyTorch/Lightning

Say you have a hand‑drawn U‑Net diagram saved as `unet_sketch.jpg`.

```bash
ollama run gemma4:31b-coding-mtp-bf16 \
  --image ./unet_sketch.jpg \
  "This is a U-Net style architecture diagram for 2D medical image segmentation.
   1) Infer the number of downsampling and upsampling stages.
   2) Write PyTorch code for a UNet2D class compatible with 1-channel input
      and a configurable number of output classes.
   3) Use Conv2d, BatchNorm2d, ReLU, MaxPool2d, and ConvTranspose2d as appropriate.
   4) At the end, show a short example constructing the model for 1 input channel
      and 3 output classes, and running a [1,1,256,256] tensor through it."
```

If you want a Lightning module:

```bash
"... Now wrap this architecture in a PyTorch Lightning module with a
cross-entropy loss and a simple training_step."
```

***

## 3. Flow diagram → Python pipeline

Suppose you have a block diagram: “Load DICOM → Convert to NIfTI → Bias correction → Registration → Segmentation” in `pipeline_diagram.png`.

```bash
ollama run gemma4:31b-coding-mtp-bf16 \
  --image ./pipeline_diagram.png \
  "This block diagram represents a brain MRI preprocessing pipeline.
   1) Read the diagram and list each step in order, with a one-line description.
   2) Implement a Python module that exposes a function
        run_pipeline(input_dir: str, output_dir: str) -> None
      that performs each step using common open-source tools
      (e.g., dcm2niix, nibabel, SimpleITK, MONAI transforms).
   3) Use placeholder commands or function calls where needed,
      but keep the structure realistic and idiomatic."
```

***

## 4. Class diagram / ER diagram → data classes

Imagine a UML class diagram or ER diagram showing entities like `Patient`, `Study`, `Series`, `Image`, stored in `dicom_uml.png`.

```bash
ollama run gemma4:31b-coding-mtp-bf16 \
  --image ./dicom_uml.png \
  "This UML diagram shows a data model for DICOM entities.
   1) Extract the classes, their fields, and relationships.
   2) Generate Python code using dataclasses to represent this model.
   3) Include type annotations and simple __str__ methods.
   4) At the end, provide a short example that instantiates a Patient
      with one Study, one Series, and two Images."
```

***

## 5. State machine diagram → implementation

Say you have a state diagram for a job scheduler (Queued → Running → Succeeded/Failed) in `state_machine.png`.

```bash
ollama run gemma4:31b-coding-mtp-bf16 \
  --image ./state_machine.png \
  "This is a state machine diagram for a job scheduler.
   1) Infer the states and transitions.
   2) Implement a Python class JobStateMachine with methods
      for each transition, enforcing valid transitions only.
   3) Add a short example showing creation of a job, running it,
      and handling both success and failure paths."
```

***

All of these follow the same pattern:

- Attach the diagram with `--image`.
- In the text, explicitly say: “interpret the diagram” → “translate into code” → “give a usage example”.

**What kind of diagram do you actually have in mind right now (CNN sketch, U‑Net, flowchart, UML, or something else)? If you describe it, a prompt can be tuned very tightly to your use case.**

# multimodal prompt with `gemma4:31b-coding-mtp-bf16`

You can test a multimodal prompt with `gemma4:31b-coding-mtp-bf16` in two main ways: CLI or HTTP API. Here’s a concrete, copy‑paste‑ready example for each.

***

## 1. Test via CLI

Assume you have an architecture sketch file `unet_diagram.png` in the current directory.

```bash
ollama run gemma4:31b-coding-mtp-bf16 \
  --image ./unet_diagram.png \
  "You see a U-Net style convolutional architecture for 2D medical image segmentation.

   1) Describe the architecture in words (downsampling path, bottleneck, upsampling path, skip connections).
   2) Translate the diagram into idiomatic PyTorch code by implementing a UNet2D(nn.Module)
      that supports 1 input channel and a configurable number of output classes.
   3) Show a complete forward() method and a short example that instantiates the model
      and runs a dummy tensor of shape [1, 1, 256, 256] through it.
   4) Finally, comment inline on any assumptions you had to make that are not explicit
      in the diagram."
```

This gives the model an image + an explicit coding task and forces it to both “read” the diagram and emit runnable code.

***

## 2. Test via HTTP API (Python)

Same idea, but from a Python script using the local Ollama endpoint:

```python
import base64
import requests

# Encode the image as base64
with open("unet_diagram.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "gemma4:31b-coding-mtp-bf16",
    "prompt": (
        "You see a U-Net style convolutional architecture for 2D medical image segmentation.\n"
        "1) Describe the architecture in words (down path, bottleneck, up path, skip connections).\n"
        "2) Implement a UNet2D in PyTorch that matches the diagram as closely as possible.\n"
        "3) Include a forward() method and an example call on a [1, 1, 256, 256] tensor."
    ),
    "images": [img_b64],
}

resp = requests.post("http://localhost:11434/api/generate", json=payload)
print(resp.json()["response"])
```

This is usually the easiest way to inspect the raw output and quickly tweak prompts.

***

If you tell a bit about the kind of diagram you actually have (e.g., CNN, U‑Net, flowchart, UML, or a plotting/analysis figure), a prompt can be tailored more tightly to your specific multimodal test.

----

Last updated: 2026-05-08

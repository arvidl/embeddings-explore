# embeddings-explore

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

Last updated: 2025-09-09

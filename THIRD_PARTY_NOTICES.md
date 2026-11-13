# Componentes de terceiros

O projeto integra componentes externos sem reivindicar a autoria dos respectivos pesos ou bibliotecas.

| Componente | Origem | Condições |
|---|---|---|
| SmolVLM2 | HuggingFaceTB/SmolVLM2-500M-Video-Instruct | Consultar licença Apache 2.0 e card original |
| Qwen3-VL | Qwen/Qwen3-VL-2B-Instruct | Consultar licença Apache 2.0 e card original |
| Stable Diffusion 1.5 | stable-diffusion-v1-5/stable-diffusion-v1-5 | CreativeML Open RAIL-M; preservar restrições e atribuições |
| IGNIS Image LoRA v1 | ViniciusKhan/ignis-image-lora-v1 | Adaptador derivado da base SD 1.5; consultar model card e documentos BASE_* no release |
| DiffusionDB | poloclub/diffusiondb | CC0 declarada no card da revisão utilizada; conteúdo sintético |
| Bibliotecas Python e frontend | requirements.txt e frontend/package.json | Cada dependência mantém sua própria licença |
| Imagem-base Docker | python:3.13-slim (Debian bookworm) | Python sob PSF License; pacotes Debian sob suas respectivas licenças |
| PyTorch/Torchvision CUDA | download.pytorch.org/whl/cu128 | BSD-3-Clause; as wheels incluem bibliotecas NVIDIA CUDA/cuDNN redistribuídas sob os termos da NVIDIA |
| Node.js (somente build) | node:22-bookworm-slim | Usado apenas para compilar o frontend; não está presente na imagem final |

Os pesos e o dataset não estão incluídos neste repositório Git nem na imagem Docker publicada em ghcr.io/viniciuskanh/ignis-multimodal. A lista exata de pacotes Python da imagem fica em `/opt/ignis-pip-freeze.txt` dentro do container. Downloads são feitos separadamente. O logotipo identifica IGNIS Multimodal; não substitui atribuições de terceiros.

As animações em docs/media derivam da gravação da aplicação fornecida pelo autor para esta documentação. O vídeo original não é redistribuído integralmente no repositório.

Não foi definida uma licença geral de código para a aplicação. A publicação no GitHub não implica que as licenças dos modelos se estendam a todo o código.

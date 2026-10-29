---
language: [en, pt]
license: creativeml-openrail-m
library_name: diffusers
pipeline_tag: text-to-image
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
base_model_relation: adapter
datasets: [poloclub/diffusiondb]
tags: [lora, stable-diffusion, text-to-image, diffusers, ignis, experimental]
---

<p align="center">
  <img src="assets/ignis-multimodal.png" alt="Logotipo IGNIS Multimodal" width="760" />
</p>

# IGNIS Image LoRA v1

**Adaptação eficiente de geração de imagens, com treinamento local e resultados auditáveis.**

[Experimentar no Space](https://huggingface.co/spaces/ViniciusKhan/IGNIS-Image-Lab) · [Pesos](pytorch_lora_weights.safetensors) · [Treinamento](training_report.json) · [Avaliação](evaluation_report.json) · [Proveniência](data_provenance.json)

Desenvolvido por **Vinicius de Souza Santos**, este é um adaptador **LoRA de rank 8 para o UNet do Stable Diffusion 1.5**, integrante do projeto IGNIS Multimodal. A contribuição própria compreende o adaptador treinado, os pipelines, a avaliação, a interface e a integração. A autoria dos pesos-base permanece atribuída aos respectivos responsáveis.

**Estado:** experimento concluído, com 550 atualizações do otimizador e 6 comparações visuais. As evidências disponíveis não demonstram superioridade geral sobre o modelo-base.

## Navegação

- [Ficha técnica](#ficha-técnica)
- [Arquitetura](#arquitetura)
- [Dataset e separação](#dataset-e-separação)
- [Treinamento executado](#treinamento-executado)
- [Resultados e avaliação](#resultados-e-avaliação)
- [Instalação e inferência](#instalação-e-inferência)
- [Edição de imagens](#edição-de-imagens)
- [Space e reprodução local](#space-e-reprodução-local)
- [Limitações](#limitações)
- [Arquivos e integridade](#arquivos-e-integridade)
- [Licença, autoria e referências](#licença-autoria-e-referências)

## Ficha técnica

| Item | Especificação |
|---|---|
| Repositório | `ViniciusKhan/ignis-image-lora-v1` |
| Artefato | Adaptador LoRA; requer os componentes da base |
| Base | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Revisão da base | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` |
| Pesos | `pytorch_lora_weights.safetensors` |
| Tamanho dos pesos | 6414992 bytes; aproximadamente 6.41 MB decimais |
| Parâmetros LoRA armazenados | 1594368 elementos nos tensores do adaptador |
| Módulos treinados | LoRA nas projeções de atenção do UNet |
| Módulos congelados | VAE, encoder de texto e parâmetros originais do UNet |
| Resolução de treino | 512 × 512 |
| Dataset | 82 imagens de treino; 14 de validação |
| Hardware local | RTX 4050 Laptop, 6 GB de VRAM; 32 GB de RAM; Windows |
| Idioma dos prompts | Inglês recomendado; português aceito como entrada, sem avaliação específica |
| Palavra de ativação | Nenhum token especial introduzido |

Este repositório publica o componente **Ignis Image**. SmolVLM2 e Qwen3-VL pertencem às integrações independentes de compreensão visual da plataforma local; seus pesos não estão incluídos neste adaptador. O Space apresenta geração e edição de imagens, sem expor todo o backend da plataforma multimodal.

## Arquitetura

O pipeline utiliza tokenizer e encoder CLIP para representar o texto, UNet para estimar o ruído no espaço latente e VAE para codificar/decodificar imagens. Esses componentes são fornecidos pela base. O arquivo LoRA contém apenas as atualizações de baixa ordem aprendidas.

Para uma matriz adaptada, a atualização tem a forma `W' = W + (α/r) BA`, com rank e alpha iguais a `8`. Os alvos são `to_q`, `to_k`, `to_v` e `to_out.0`, com inicialização gaussiana. O encoder de texto não recebeu LoRA.

**Entrada:** descrição textual e, nos pipelines de edição, imagem e máscara. **Saída:** imagem sintética. Os pesos aqui publicados não realizam OCR, compreensão visual ou conversação.

## Dataset e separação

| Campo | Registro |
|---|---|
| Fonte utilizada | Recorte de imagens sintéticas do DiffusionDB |
| Repositório | [poloclub/diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb) |
| Revisão | `fb620fbe49fa4420e0734bd9c0df11f51176b61f` |
| Arquivo de origem | `images/part-000001.zip` |
| Licença declarada pelos mantenedores | CC0 1.0 |
| Total | 96 pares imagem–legenda |
| Treino / validação | 82 / 14 |
| Seed da separação | `42` |
| Fingerprint | `90c1f9c61a6dfa9716dd866880f90b9973131bced5e87872f37bfd30bdf6f999` |

As legendas provêm dos prompts do DiffusionDB. As imagens são sintéticas, originalmente geradas por Stable Diffusion. A alternativa de coleta do Art Institute of Chicago não forneceu os dados desta execução.

O preparo seleciona um exemplar por prompt normalizado, calcula hashes dos pixels, remove duplicatas exatas, ordena e embaralha os registros com seed fixa e separa aproximadamente 85%/15%. A checagem desta versão encontrou **0 hashes de imagem em comum entre treino e validação**. A validação orienta a medição da loss e a parada, sem participar das atualizações por gradiente.

A exclusão textual é heurística, sem certificação de segurança visual. A deduplicação exata não elimina semelhanças semânticas nem vieses. O pequeno recorte não é representativo da diversidade de imagens possíveis. [data_provenance.json](data_provenance.json) registra legendas, hashes, origem e licença individual; o dataset completo não é redistribuído aqui.

## Treinamento executado

| Parâmetro | Valor |
|---|---|
| Rank / LoRA alpha | 8 / 8 |
| Resolução | 512 × 512 |
| Batch por dispositivo | 1 |
| Acumulação de gradientes | 4 |
| Batch efetivo em uma GPU | 4 |
| Otimizador | AdamW; weight decay `0.01` |
| Learning rate inicial | `0.0001` |
| Scheduler | Cosseno com warmup |
| Warmup | 50 atualizações |
| Limite configurado / executado | 800 / 550 |
| Seed | 42 |
| Checkpoint / validação | A cada 50 / 50 atualizações |
| Early stopping | Paciência 5; mínimo 300 passos; melhora mínima `1e-5` |
| Motivo da parada | `early_stopping_validation` |
| Tempo registrado | 842.06 segundos; exclui instalação, downloads e avaliação posterior |

Precisão mista FP16, parâmetros treináveis LoRA em FP32, gradient checkpointing e batch unitário com acumulação reduzem a memória necessária. Latentes do VAE e embeddings de texto são calculados em etapas separadas e armazenados em cache; os módulos congelados são liberados antes do treino do UNet. Imagens recebem correção EXIF, conversão RGB, ajuste quadrado e normalização para `[-1, 1]`.

O objetivo é a MSE entre a predição do UNet e o alvo do scheduler de difusão. A validação usa ruído e timesteps com sementes fixas por amostra. Checkpoints guardam LoRA e estado para retomada. A implementação admite redução de resolução após falta de memória; **esta execução terminou em 512 × 512**.

Ambiente observado: Python 3.13, PyTorch `2.11.0+cu128`, torchvision `0.26.0+cu128`, Diffusers `0.35.1`, Transformers `4.57.1`, PEFT `0.17.1` e Accelerate `1.10.1`. Essas versões descrevem o ambiente local; não representam certificação para todas as plataformas.

## Resultados e avaliação

| Medida | Valor registrado |
|---|---|
| Loss inicial de validação | 0.19390892872719892 |
| Melhor loss de validação | 0.19316272154849554 |
| Passo do melhor checkpoint | 300 |
| Loss de validação no adaptador final | 0.19317844051069447 |
| Última loss de treino | 0.14198050647974014 |
| Passo do adaptador publicado | 550 |

**Os pesos publicados são do adaptador final do passo 550.** O melhor checkpoint por loss foi observado no passo 300. A loss mede reconstrução de ruído no pequeno conjunto de validação; sua variação não corresponde a uma porcentagem de melhoria perceptual.

Foram gerados **6 pares, totalizando 12 imagens**, com três prompts e seeds `42` e `2026`, em `512 × 512` e `25` passos por imagem. Nos painéis, **base à esquerda e IGNIS LoRA à direita**. Não foram calculados FID, CLIPScore, preferências humanas ou significância estatística.

### Comparação 1: seed 42

Prompt: `A detailed oil painting of a quiet garden, soft morning light`

![Base à esquerda e IGNIS à direita, seed 42](samples/pair-1.png)

### Comparação 2: seed 2026

Prompt: `A detailed oil painting of a quiet garden, soft morning light`

![Base à esquerda e IGNIS à direita, seed 2026](samples/pair-2.png)

### Comparação 3: seed 42

Prompt: `A watercolor landscape with mountains and a lake`

![Base à esquerda e IGNIS à direita, seed 42](samples/pair-3.png)

### Comparação 4: seed 2026

Prompt: `A watercolor landscape with mountains and a lake`

![Base à esquerda e IGNIS à direita, seed 2026](samples/pair-4.png)

### Comparação 5: seed 42

Prompt: `A still life painting of flowers in a ceramic vase`

![Base à esquerda e IGNIS à direita, seed 42](samples/pair-5.png)

### Comparação 6: seed 2026

Prompt: `A still life painting of flowers in a ceramic vase`

![Base à esquerda e IGNIS à direita, seed 2026](samples/pair-6.png)

Os resultados se restringem a esses prompts e seeds. Não demonstram especialização artística geral, robustez em português ou melhoria em edição. [evaluation_report.json](evaluation_report.json) identifica os pares e o hash dos pesos avaliados. A reprodução exata depende também do scheduler, das bibliotecas e do hardware.

## Instalação e inferência

### Ambiente

Utilize um ambiente virtual e um PyTorch CUDA compatível com o driver. A execução local utilizou CUDA 12.8:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install diffusers==0.35.1 transformers==4.57.1 peft==0.17.1 accelerate==1.10.1 "huggingface-hub>=0.34,<1" safetensors Pillow
```

O bloco seguinte é Python: salve como `gerar_ignis.py` e execute com `.\.venv\Scripts\python.exe gerar_ignis.py`. O primeiro uso baixa a base e o adaptador; não requer novo treinamento.

### Geração com pouca VRAM

```python
from pathlib import Path
import os

# Configura o cache dentro do projeto antes de importar bibliotecas HF.
raiz = Path.cwd()
os.environ["HF_HOME"] = str(raiz / ".cache" / "huggingface")

import torch
from diffusers import StableDiffusionPipeline

if not torch.cuda.is_available():
    raise RuntimeError("Este exemplo requer GPU CUDA compatível.")

pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    revision="451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    variant="fp16",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipeline.load_lora_weights(
    "ViniciusKhan/ignis-image-lora-v1",
    revision="bdcba88b1d095e95ccbaa6066b48d9d0ca987e59",
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="ignis",
)
pipeline.set_adapters("ignis", adapter_weights=1.0)
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

resultado = pipeline(
    prompt="A watercolor landscape with mountains and a lake",
    negative_prompt="blurry, low quality",
    width=512,
    height=512,
    num_inference_steps=25,
    guidance_scale=7.5,
    generator=torch.Generator(device="cpu").manual_seed(42),
)
if resultado.nsfw_content_detected and any(resultado.nsfw_content_detected):
    raise RuntimeError("A saída foi bloqueada pelo filtro de segurança.")

destino = raiz / "outputs"
destino.mkdir(exist_ok=True)
resultado.images[0].save(destino / "ignis-exemplo.png")
```

Esse é um exemplo de uso, não uma reprodução pixel a pixel dos painéis. Para reduzir memória ainda mais, substitua `enable_model_cpu_offload()` por `enable_sequential_cpu_offload()`; evite combinar os dois métodos ou mover todo o pipeline para CUDA antes do offloading.

Para comparar com a base, use `pipeline.disable_lora()`, repita com uma **nova instância do generator usando a mesma seed** e reative com `pipeline.enable_lora()`. Explore intensidades entre `0.0` e `1.0`; o valor `1.0` é inicial, não um ótimo comprovado.

### Modo offline

Após o download, substitua os identificadores por diretórios locais e use `local_files_only=True` em ambos os carregamentos. Informe explicitamente `weight_name="pytorch_lora_weights.safetensors"`: o Diffusers pode não inferir esse nome offline. A base precisa conter os componentes completos do pipeline; o adaptador isolado não basta.

## Edição de imagens

O adaptador pode ser usado com image-to-image e inpainting compatíveis com SD 1.5. Não foi treinado especificamente para seguir instruções de edição ou preservar identidades. O projeto usa inpainting com o UNet de quatro canais da base, sem alegar um checkpoint especializado de nove canais.

Após carregar o `pipeline` do exemplo anterior:

```python
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# Remove hooks antes de compartilhar os componentes com outro pipeline.
pipeline.remove_all_hooks()
editor = StableDiffusionImg2ImgPipeline(**pipeline.components)
editor.enable_model_cpu_offload()

entrada = Image.open("entrada.png").convert("RGB").resize((512, 512))
resultado = editor(
    prompt="A watercolor landscape, warm sunset light, soft colors",
    image=entrada,
    strength=0.45,
    num_inference_steps=25,
    guidance_scale=7.5,
    generator=torch.Generator(device="cpu").manual_seed(42),
)
if resultado.nsfw_content_detected and any(resultado.nsfw_content_detected):
    raise RuntimeError("A saída foi bloqueada pelo filtro de segurança.")
resultado.images[0].save("outputs/ignis-edicao.png")
```

Para inpainting, utilize `StableDiffusionInpaintPipeline`, envie `image=entrada` e `mask_image=mascara`: branco na área a editar e preto no entorno, com máscara alinhada à imagem. Intensidade maior pode alterar mais a composição; não há garantia de preservação exata.

## Space e reprodução local

O [IGNIS Image Lab](https://huggingface.co/spaces/ViniciusKhan/IGNIS-Image-Lab) disponibiliza:

| Aba | Recursos |
|---|---|
| Estúdio | Prompt, estilos, seed, resolução, passos, CFG, intensidade LoRA e seleção base/adaptador |
| Comparar | Base e LoRA com parâmetros compartilhados e comparação deslizante |
| Editar | Imagem inteira ou região com máscara; intensidade da mudança |
| Minha sessão | Histórico temporário e receitas |
| Pesquisa e documentação | Relatórios e amostras reais |

Há exportação PNG, JSON e ZIP e importação de receitas. Os estilos são modificadores do prompt, não outros adaptadores treinados. ZeroGPU depende de autorização, cotas, fila e disponibilidade do serviço. Baixe os resultados: o histórico é temporário.

**Teste de edição:** envie uma paisagem, escolha Imagem inteira, intensidade `0.45`, seed `42`, `25` passos e LoRA `1.0`. Prompt: `A watercolor landscape, warm sunset light, soft colors`. Deixe a máscara vazia nessa operação.

Prompts em português são aceitos como texto, sem tradução automática nesta versão. O campo `language` do cabeçalho não certifica capacidade multilíngue; não foi realizado benchmark específico.

Os comandos abaixo pertencem ao **código do projeto local IGNIS Multimodal**, não ao repositório de pesos. Execute na raiz do projeto:

```powershell
.\scripts\bootstrap.ps1
.\scripts\download-models.ps1 -Models sd15
# Preparar dados somente para um novo experimento.
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --limit 96
.\scripts\train-lora.ps1
.\scripts\evaluate.ps1 -Adapter 'ignis-image-lora-v1/final'
.\scripts\status.ps1
.\scripts\start.ps1
```

Antes de reproduzir, preserve e compare o manifesto original, a configuração, a revisão da base e o fingerprint. O coletor pode tentar uma fonte alternativa quando uma API estiver indisponível; uma nova coleta não garante o mesmo recorte. As referências de [data_provenance.json](data_provenance.json) identificam os dados utilizados.

O treinamento retoma checkpoints completos. Use outro nome de execução para novos experimentos e preserve treino, validação e checkpoints anteriores. Avalie `best` e `final` separadamente antes de substituir uma versão. Seeds fixas não garantem igualdade numérica entre plataformas.

## Limitações

- **Escala:** 96 imagens sintéticas e uma execução não sustentam generalização nem superioridade estatística.
- **Herança da base:** artefatos anatômicos, texto ilegível, contagem incorreta de objetos, estereótipos e problemas de composição podem persistir.
- **Dados sintéticos:** podem reproduzir erros e vieses do gerador original; não acrescentam evidência factual sobre o mundo.
- **Português:** sem fine-tuning linguístico específico ou benchmark dedicado.
- **Edição:** não é um modelo especializado em instruções complexas, preservação de rostos, marcas ou documentos.
- **Avaliação:** sem FID, CLIPScore, estudo humano ou benchmark de segurança.
- **Safety checker:** mantido nos pipelines; pode falhar e não garante segurança.
- **Privacidade:** evite enviar dados pessoais e documentos confidenciais ao Space público. Imagens geradas não são prova de acontecimentos, diagnóstico ou evidência documental.

Usos pretendidos: pesquisa, demonstração técnica e experimentação criativa, sujeitos à licença-base e à revisão humana dos resultados.

## Arquivos e integridade

| Arquivo | Conteúdo |
|---|---|
| `pytorch_lora_weights.safetensors` | Pesos LoRA finais; não inclui a base completa |
| `assets/ignis-multimodal.png` | Logotipo do projeto |
| `training_config.json` | Configuração executada |
| `training_report.json` | Passos, losses, parada e duração registrada |
| `evaluation_report.json` | Prompts, seeds, amostras e hash do adaptador |
| `data_provenance.json` | Origem, licença, divisão e hashes dos dados |
| `base_model.json` | Origem e revisão da base |
| `samples/pair-*.png` | Painéis de comparação |
| `BASE_*` | Documentos de licença-base |
| `checksums.json` | SHA-256 dos arquivos do pacote |

SHA-256 dos pesos avaliados e publicados:

```text
f1a2811fbc685bb3059fb43c43c32d44d6ffab51fc5d492e7ff80ec4825f77e5
```

Verificação no Windows:

```powershell
Get-FileHash -Algorithm SHA256 .\pytorch_lora_weights.safetensors
```

Revisão imutável dos pesos usados nos exemplos: `bdcba88b1d095e95ccbaa6066b48d9d0ca987e59`. Uma atualização de documentação pode alterar `main` sem alterar os pesos.

## Licença, autoria e referências

**Autor do projeto e do adaptador:** Vinicius de Souza Santos — [ViniciusKhan](https://huggingface.co/ViniciusKhan). **Projeto:** IGNIS Multimodal. **Versão:** experimental, `ignis-image-lora-v1`.

O modelo derivado segue as condições CreativeML Open RAIL-M da base, incluindo restrições de uso e obrigações de redistribuição. A CC0 declarada pelo DiffusionDB refere-se ao dataset e não substitui a licença-base. Consulte os arquivos `BASE_*` e os cards originais. Este registro não constitui parecer jurídico sobre imagens específicas.

O logotipo identifica o projeto, sem atribuir a ele a autoria do Stable Diffusion, DiffusionDB, Diffusers ou outros componentes de terceiros.

- [Stable Diffusion 1.5 — base](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
- [DiffusionDB — revisão utilizada](https://huggingface.co/datasets/poloclub/diffusiondb/blob/fb620fbe49fa4420e0734bd9c0df11f51176b61f/README.md).
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752).
- [Diffusers — carregar adaptadores](https://huggingface.co/docs/diffusers/using-diffusers/loading_adapters).
- [Hugging Face — model cards](https://huggingface.co/docs/hub/model-cards).

Citação sugerida para este artefato, sem implicar publicação revisada por pares:

```bibtex
@misc{santos2026ignisimage,
  author = {Santos, Vinicius de Souza},
  title = {IGNIS Image LoRA v1: adaptador experimental para Stable Diffusion 1.5},
  year = {2026},
  howpublished = {Hugging Face},
  url = {https://huggingface.co/ViniciusKhan/ignis-image-lora-v1},
  note = {Adaptador experimental. Revisao dos pesos: bdcba88b1d095e95ccbaa6066b48d9d0ca987e59}
}
```

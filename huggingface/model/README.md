---
language:
  - en
  - pt
license: creativeml-openrail-m
library_name: diffusers
pipeline_tag: text-to-image
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
base_model_relation: adapter
tags:
  - lora
  - stable-diffusion
  - ignis
  - experimental
---

# IGNIS Image LoRA v1

**Rascunho de model card. Não há pesos treinados ou avaliação confirmados nesta versão do documento.**

IGNIS Image LoRA v1 é o primeiro adaptador de geração de imagens previsto para o projeto IGNIS Multimodal, desenvolvido por Vinicius de Souza Santos. A proposta é especializar o UNet do Stable Diffusion 1.5 por Low-Rank Adaptation (LoRA), mantendo os parâmetros originais congelados.

Este repositório se destina ao adaptador, não a um modelo-base treinado do zero. Ele não contém os componentes de compreensão visual SmolVLM2 ou Qwen3-VL. Esses modelos são integrações independentes da plataforma IGNIS Multimodal.

## Identificação e autoria

| Item | Descrição |
|---|---|
| Nome previsto | IGNIS Image LoRA v1 |
| Repositório proposto | `ViniciusKhan/ignis-image-lora-v1` |
| Base | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Tipo de contribuição | Adaptador LoRA, treinamento, integração e avaliação |
| Autor do projeto IGNIS | Vinicius de Souza Santos |
| Estado atual deste documento | Rascunho anterior à publicação dos pesos |

A autoria do projeto IGNIS e dos adaptadores não transfere a autoria dos pesos-base nem das obras utilizadas no experimento. As referências e condições da base devem acompanhar a distribuição.

## Objetivo e uso pretendido

Investigar adaptação eficiente de geração de imagens em hardware de consumo, com memória de GPU limitada. O experimento inicial utiliza obras de arte com metadados institucionais e indicador de domínio público, visando explorar adaptação visual. Não foi demonstrada uma melhoria geral sobre o modelo-base, nem uma especialização artística consistente.

Os usos previstos são pesquisa, demonstração e experimentação criativa dentro das condições de uso da base. Não utilizar as imagens como prova factual, reprodução documental exata, diagnóstico ou evidência científica sem verificação independente.

## Arquitetura e método

Stable Diffusion 1.5 é uma base de difusão latente condicionada por texto. O pipeline IGNIS congela VAE, codificador de texto e parâmetros-base do UNet. As atualizações LoRA são aplicadas às projeções `to_q`, `to_k`, `to_v` e `to_out.0` do UNet.

O pré-cálculo de latentes e embeddings ocorre em sequência para reduzir a coexistência dos componentes na GPU. O treinamento utiliza precisão mista, acumulação de gradientes e gradient checkpointing. Os pesos treináveis permanecem em FP32 para compatibilidade com o scaler FP16.

## Dados e proveniência

O coletor implementado consulta a API do Art Institute of Chicago e aceita somente registros com `is_public_domain=true` e imagem disponível. O plano inicial limita a coleta a 96 imagens; esse número é uma configuração, não uma quantidade já coletada.

As legendas são construídas com título, tipo de obra e materiais informados pela instituição; não são descrições visuais humanas exaustivas. Metadados podem conter lacunas, vieses de catalogação ou descrições inadequadas para treinamento.

As imagens são deduplicadas pelo hash dos pixels decodificados. Aproximadamente 15% das imagens únicas são reservadas para validação, com no mínimo duas imagens. O manifesto registra arquivo, legenda, hash, origem e licença. A publicação inicial não deve redistribuir o dataset por padrão.

Fontes: [API e política de imagens](https://api.artic.edu/docs/#copyright) e [modelo-base](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).

## Configuração inicial planejada

| Parâmetro | Valor inicial |
|---|---|
| Rank / alpha | 8 / 8 |
| Resolução | 512 × 512; tentativas separadas em 384 e 256 se necessário |
| Batch / acumulação | 1 / 4 |
| Learning rate | 0,0001 |
| Otimizador / scheduler | AdamW / cosine com warmup |
| Warmup / limite | 50 / 800 atualizações |
| Checkpoint / validação | A cada 50 atualizações |
| Early stopping | Após pelo menos 300 passos e cinco validações sem melhora |
| Seed | 42 |
| Hardware-alvo | RTX 4050 Laptop com 6 GB de VRAM; 32 GB de RAM |

A configuração efetiva da versão publicada será registrada a partir de `config.json`, não apenas desta tabela de intenção.

## Resultados e avaliação

**Resultados ainda não disponíveis neste rascunho.** Não são declarados loss, FID, CLIP score, tempo de treinamento ou melhoria de qualidade.

A avaliação implementada compara base e adaptador em três prompts e duas sementes fixas, com parâmetros de geração iguais. A validação mede erro quadrático de denoising em dados separados, com ruído e timestep fixos. Essa métrica não substitui análise visual e não demonstra, isoladamente, melhor alinhamento ao prompt ou generalização.

Antes da publicação dos pesos, esta seção receberá número real de atualizações, configuração final, hashes de proveniência, loss inicial e melhor loss observada, além de seis comparações visuais reais. Não haverá conversão de testes de software em métricas de qualidade do modelo.

## Como carregar os pesos após a publicação

O código abaixo pressupõe que o repositório já tenha sido publicado com o adaptador e que exista GPU CUDA compatível:

```python
import torch
from diffusers import StableDiffusionPipeline

# Os pesos-base e o adaptador são carregados separadamente.
pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipeline.load_lora_weights("ViniciusKhan/ignis-image-lora-v1")
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

# Prompts em inglês são recomendados para esta base.
imagem = pipeline(
    "A detailed oil painting of a quiet garden, soft morning light",
    generator=torch.Generator(device="cpu").manual_seed(42),
    num_inference_steps=25,
    width=512,
    height=512,
).images[0]
imagem.save("ignis-exemplo.png")
```

O filtro de segurança da base permanece habilitado. A existência de instruções de carregamento não implica que os pesos já estejam disponíveis.

## Limitações e riscos

- Dataset pequeno, heterogêneo e com legendas derivadas de catálogo; risco de sobreajuste e associação espúria entre título e conteúdo visual.
- Vieses e limitações herdados da base e da seleção museológica.
- Textos, mãos, anatomia, contagem de objetos e relações espaciais podem apresentar erros.
- Não há comprovação de geração confiável de texto em português ou de domínio geral multimodal neste adaptador.
- Imagens geradas não comprovam a existência das cenas retratadas.
- O adaptador pode aumentar a semelhança com exemplos de treinamento; inspecionar memorizações antes de ampliar a distribuição.

## Licença e distribuição

Preserve as condições da CreativeML Open RAIL-M e as atribuições aplicáveis ao modelo-base. A versão de publicação inclui a licença original e os metadados de origem recuperados do Hub. O estado de domínio público de uma obra não implica automaticamente ausência de restrições sobre marcas, pessoas retratadas ou todos os usos em todas as jurisdições.

## Reprodutibilidade e arquivos da versão

A versão final deve conter `pytorch_lora_weights.safetensors`, `training_config.json`, `training_report.json`, `data_provenance.json`, `base_model.json`, `evaluation_report.json` e amostras pareadas. Checkpoints internos, estados do otimizador, caches, tokens e imagens originais não são publicados por padrão.

## Referências e citação

- [Stable Diffusion 1.5 — model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Diffusers — treinamento LoRA](https://huggingface.co/docs/diffusers/training/lora)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Art Institute of Chicago — API](https://api.artic.edu/docs/)

Após a primeira versão, cite o repositório IGNIS, sua revisão exata e a data de acesso. Este projeto não declara artigo revisado por pares ou DOI próprio.

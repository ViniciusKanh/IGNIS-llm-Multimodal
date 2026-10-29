---
title: IGNIS Image Lab
emoji: 🔬
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 5.49.1
python_version: "3.10"
app_file: app.py
models:
  - ViniciusKhan/ignis-image-lora-v1
  - stable-diffusion-v1-5/stable-diffusion-v1-5
---

# IGNIS Image Lab

Demonstração de geração de imagens e comparação entre Stable Diffusion 1.5 e o adaptador IGNIS Image LoRA v1. A aplicação identifica qual componente gerou cada saída e utiliza a mesma seed na comparação.

**Pré-requisito:** o adaptador precisa existir no repositório indicado em `IGNIS_MODEL_ID`. A interface pode abrir antes disso, mas gerações com IGNIS informarão explicitamente a ausência dos pesos. A base pode ser testada de forma independente.

## Recursos

- Geração a partir de texto, prompt negativo, seed e seleção entre base e IGNIS.
- Comparação com parâmetros idênticos, carregamento serializado e limpeza de memória.
- Documentação do modelo publicada no Hub.
- Sem treinamento público, acesso ao sistema de arquivos do usuário ou download arbitrário de modelos pela interface.

Este Space demonstra apenas o componente de imagens. A plataforma local IGNIS Multimodal inclui visão, documentos, edição e laboratório de treinamento, que não são automaticamente disponibilizados por esta demonstração.

## Configuração

Variável `IGNIS_MODEL_ID`: repositório do adaptador. O empacotador gera também `release.json` com a revisão exata da versão do modelo. Token opcional de leitura para repositórios privados deve ser configurado como segredo `HF_TOKEN` nas configurações do Space.

O código seleciona CUDA quando disponível e FP32 no CPU. A modalidade gratuita CPU não usa a RTX 4050 do computador do autor: a execução ocorre na infraestrutura do Hugging Face e pode ser lenta. GPU e ZeroGPU dependem da disponibilidade e elegibilidade da conta; este projeto não contrata hardware automaticamente.

## Execução local

```powershell
python -m pip install -r requirements.txt
python app.py
```

## Autoria e limites

Projeto IGNIS: Vinicius de Souza Santos. Modelo-base: Stable Diffusion 1.5; especialização: adaptador LoRA IGNIS. O adaptador não representa treinamento integral da base do zero. Preservam-se os filtros e as condições de uso da base.

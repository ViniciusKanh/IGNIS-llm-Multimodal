---
title: IGNIS Image Lab
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
python_version: "3.10"
app_file: app.py
suggested_hardware: zero-a10g
models:
  - ViniciusKhan/ignis-image-lora-v1
  - stable-diffusion-v1-5/stable-diffusion-v1-5
---

# IGNIS Image Lab

Demonstração de geração de imagens e comparação entre Stable Diffusion 1.5 e o adaptador IGNIS Image LoRA v1. A aplicação identifica qual componente gerou cada saída e utiliza a mesma seed na comparação.

**Pré-requisito:** o adaptador precisa existir no repositório indicado em `IGNIS_MODEL_ID`. A interface pode abrir antes disso, mas gerações com IGNIS informarão explicitamente a ausência dos pesos. A base pode ser testada de forma independente.

## Recursos

- Estúdio com estilos, exemplos, seed fixa ou aleatória, enquadramento, intensidade LoRA, orientação, passos e amostrador.
- Comparação com parâmetros idênticos e controle deslizante base × IGNIS.
- Edição image-to-image e inpainting com máscara enviada pelo usuário.
- Downloads PNG, JSON e ZIP; receita incorporada ao PNG e importação de receitas.
- Histórico temporário de até 24 imagens por sessão.
- Documentação, registros de treinamento e seis comparações reais da versão publicada.
- Sem treinamento público, acesso ao sistema de arquivos do usuário ou download arbitrário de modelos pela interface.

Este Space demonstra apenas o componente de imagens. A plataforma local IGNIS Multimodal inclui visão, documentos, edição e laboratório de treinamento, que não são automaticamente disponibilizados por esta demonstração.

## Configuração

Variável `IGNIS_MODEL_ID`: repositório do adaptador. O empacotador gera também `release.json` com a revisão exata da versão do modelo. Token opcional de leitura para repositórios privados deve ser configurado como segredo `HF_TOKEN` nas configurações do Space.

O destino é ZeroGPU (`zero-a10g` na API), com `IGNIS_ZERO_GPU=1`. O runtime inicializa os pesos em CUDA fora das funções decoradas com `spaces.GPU`. Contas pessoais elegíveis podem hospedar ZeroGPU gratuito com quotas e filas; a disponibilidade é decidida pelo Hugging Face. Gradio em CPU Basic exige plano pago na política atual. O publicador não tenta hardware pago.

As imagens e receitas ficam temporariamente no servidor. Baixe os arquivos que desejar guardar. O tempo registrado mede a inferência, não necessariamente a espera na fila. Presets adicionam termos ao prompt; não são adaptadores adicionais. A comparação visual não comprova superioridade do modelo.

Use `scripts/deploy-space.ps1` do projeto para publicar somente esta demonstração, sem reenviar ou retreinar o modelo.

## Execução local

```powershell
python -m pip install -r requirements.txt
python app.py
```

## Autoria e limites

Projeto IGNIS: Vinicius de Souza Santos. Modelo-base: Stable Diffusion 1.5; especialização: adaptador LoRA IGNIS. O adaptador não representa treinamento integral da base do zero. Preservam-se os filtros e as condições de uso da base.

Referências de experiência: [FLUX.1 schnell](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell) e [Stable Diffusion 3.5 Large](https://huggingface.co/spaces/stabilityai/stable-diffusion-3.5-large). O design e a implementação IGNIS são próprios.

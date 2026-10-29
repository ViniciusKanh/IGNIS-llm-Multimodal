---
name: ignis-multimodal
description: Desenvolver e manter o IGNIS Multimodal local no Windows, integrando visão-linguagem, geração e edição com Diffusers, treinamento LoRA e API/interface, com otimização para RTX 4050 de 6 GB.
---

# IGNIS Multimodal

Trabalhe no projeto `C:\dev\Ignis-Multimodal` ou na cópia explicitamente escolhida pelo usuário. Preserve os projetos Ignis-30M e Cerbero OS.

## Execução

Leia `configs/train.json` e o manifesto `models/manifest.json` antes de alterar treinamento ou modelos. Use os scripts existentes: `bootstrap.ps1`, `download-models.ps1`, `train-lora.ps1`, `evaluate.ps1`, `start.ps1` e `status.ps1`. O auxiliar `scripts/run-project.ps1` desta skill encaminha essas operações ao projeto, sem duplicar a implementação.

Mantenha pesos, caches, datasets, checkpoints, logs e saídas dentro da raiz do projeto. Instale dependências no ambiente `.venv`; evite modificar o Python global. Use SmolVLM2 primeiro, Qwen3-VL opcional e Stable Diffusion 1.5 para geração e edição. Preserve origem, revisão e licença no manifesto. Publique no Hugging Face somente quando houver instrução explícita; publique o adaptador com model card identificando base, dados, licença, passos e limitações.

## Memória e inferência

Serialize acesso à GPU entre inferência e treinamento com o mesmo bloqueio de arquivo. Descarregue o modelo anterior, remova referências e limpe o cache CUDA. Use FP16 apenas em CUDA; no CPU use FP32. Prefira SDPA, offloading e uma única variante Safetensors. Para Qwen opcional, tente quantização BitsAndBytes de 4 bits somente se instalada e compatível; use fallback FP16 com offload em vez de supor suporte no Windows. Não desative o filtro de segurança do modelo-base.

## Treinamento e avaliação

Use LoRA UNet rank 8, batch 1, acumulação 4, resolução inicial 512, checkpointing e scheduler configurável. Pré-calcule latentes e embeddings em sequência para não manter VAE, encoder e UNet simultaneamente na GPU. Adapte resolução em tentativas distintas quando ocorrer falta de VRAM; não retome checkpoint com configuração ou dataset incompatível.

Verifique a licença e a proveniência das imagens. Separe dados por hash antes do treinamento, remova duplicatas e mantenha validação fora dos gradientes. Checkpoints devem conter adaptador, otimizador, scheduler, scaler, RNG e progresso. Compare base e LoRA com prompts e sementes idênticos, registre loss de validação com ruído fixo e produza amostras reais. Loss menor não demonstra sozinha melhor qualidade visual. Não chame teste rápido de treinamento concluído.

Consulte [references/workflow.md](references/workflow.md) para limites de edição, retomada e publicação. Corrija falhas locais recuperáveis; se rede, permissões ou recursos impedirem execução, registre a causa e o comando de retomada sem inventar resultados.

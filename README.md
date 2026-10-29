# IGNIS Multimodal

Plataforma local com FastAPI, React/TypeScript/Tailwind, visão-linguagem, geração e edição com Stable Diffusion 1.5 e treinamento de adaptadores LoRA. O código distingue operações implementadas de execução efetivamente validada: consulte `docs/execution-report.json`.

## Instalação

Quando esta cópia estiver fora de `C:\dev\Ignis-Multimodal`, execute `scripts/install-project.ps1` em PowerShell para copiar os fontes e instalar a skill. O instalador preserva AGENTS.md/README.md já existentes e não copia caches, modelos, resultados ou ambiente virtual. Não execute sobre uma instalação com alterações não revisadas nos arquivos de implementação.

Na raiz definitiva:

```powershell
.\scripts\bootstrap.ps1
.\scripts\download-models.ps1 -Models smolvlm,sd15,qwen
.\scripts\train-lora.ps1 -PrepareDataset
.\scripts\start.ps1
```

A interface compilada abre em `http://127.0.0.1:8000`. Para desenvolvimento, use `scripts/start.ps1 -Dev`, com interface na porta 5173. A API usa somente a interface de loopback. Não exponha este serviço diretamente à internet.

`scripts/status.ps1` mostra recursos e modelos reais. `scripts/evaluate.ps1` compara o adaptador final com o modelo-base. Downloads interrompidos são retomados pelo Hub; treinamento retoma o último checkpoint completo da mesma configuração.

## Dados e treinamento

O experimento inicial seleciona até 96 imagens. A fonte museológica usa obras com indicador individual de domínio público na API do Art Institute of Chicago. Quando ela retorna erro HTTP, inclusive 403, ou não há imagens suficientes, o coletor usa o dataset público `poloclub/diffusiondb`, com licença CC0 declarada na origem. Esse caminho baixa somente o primeiro lote, de aproximadamente 633 MB, e seleciona as imagens do experimento. Não exige instalar a biblioteca Datasets.

O DiffusionDB contém imagens sintéticas e seus prompts originais; o manifesto e a model card registram essa origem explicitamente. O recorte seleciona um exemplar por prompt normalizado, deduplica pixels e aplica exclusão textual heurística. Isso não garante segurança visual nem melhoria do modelo-base; inspecione os dados e os resultados. A separação reserva aproximadamente 15% para validação. Nenhum dado é utilizado sem legenda, licença e origem registradas.

Para escolher diretamente a fonte alternativa, execute `.venv\Scripts\python.exe scripts/prepare_dataset.py --source diffusiondb`. O modo padrão tenta o museu e muda automaticamente para a alternativa quando necessário.

Para imagens próprias, coloque os arquivos em `datasets/` e um JSONL com `file`, `caption`, `license`, `source`. Execute `.venv\Scripts\python.exe scripts/prepare_dataset.py --metadata minhas-imagens.jsonl`.

Configuração em `configs/train.json`: LoRA rank 8, resolução 512, batch 1, acumulação 4, até 800 passos; validação e checkpoints a cada 50. Early stopping após no mínimo 300 passos e cinco avaliações sem melhora. Tentativas com falta de VRAM usam 384 e 256 em execuções com nomes distintos. Passos abaixo do mínimo são identificados como teste, sem alegação de treinamento concluído.

O pipeline pré-calcula VAE e embeddings em sequência, mantém apenas UNet/LoRA durante atualizações, salva adaptadores Safetensors e estado do otimizador/scheduler/scaler/RNG. Métricas de validação usam ruído/timesteps fixos. A comparação visual usa três prompts e duas sementes; não há métricas FID/CLIP inventadas.

## Componentes e limites

- Ignis Vision: SmolVLM2 primeiro; Qwen3-VL opcional. PDF é renderizado em até quatro páginas; não constitui RAG.
- Ignis Image: geração local com SD 1.5 e adaptadores instalados.
- Ignis Edit: image-to-image e inpainting com UNet de quatro canais; não substitui um modelo especializado de inpainting.
- Ignis Core: seleção explícita de operação e roteamento simples por comandos em `/api/core`, além de uma operação para descrever uma referência e gerar outra imagem. Inclui carregamento sequencial, bloqueio de GPU, histórico SQLite, fila de operações e galeria. Não inclui agente autônomo de planejamento de solicitações complexas.

Pesos-base têm autoria pública: SmolVLM2/Qwen3-VL sob Apache 2.0; SD 1.5 sob CreativeML Open RAIL-M, com condições de uso preservadas. O código de download verifica a licença informada no Hub antes de transferir pesos. Tokens opcionais são lidos do ambiente pelo Hub; não grave tokens no repositório. Não ocorre publicação automática.

## Publicação do modelo e do Space

Há dois destinos propostos: `ViniciusKhan/ignis-image-lora-v1` para os pesos LoRA e `ViniciusKhan/IGNIS-Image-Lab` para a demonstração Gradio. O model card detalhado está em `huggingface/model/README.md`; o código e o README do Space estão em `huggingface/space/`. Esses arquivos são preparação local, não confirmação de repositórios já publicados.

Depois de concluir treinamento e comparação visual, execute `scripts/prepare-hf-release.ps1`. O empacotador exige pesos, treinamento concluído, dataset correspondente, revisão da base, licença original e amostras de avaliação existentes. Ele produz `outputs/hf-release/model/` e `outputs/hf-release/space/`, preenchendo a model card com evidências reais. Não publica datasets, caches, tokens ou estados do otimizador.

Para aplicar as correções desta cópia ao projeto instalado e executar as etapas restantes de uma vez, use `scripts/complete-publication.ps1`. O comando preserva backups dos fontes substituídos, reutiliza os modelos instalados, instala Torchvision compatível com a versão CUDA do PyTorch quando necessário, prepara o dataset, treina/retoma o adaptador, compara, compila e empacota. Sem a opção `-Publish`, a execução termina com os pacotes locais e não envia nada ao Hub.

Para publicar posteriormente, autentique-se com `hf auth login` e execute `scripts/prepare-hf-release.ps1 -Publish`. Essa opção envia os arquivos para os dois repositórios e fixa no Space a revisão exata dos pesos enviados. Não seleciona GPU paga nem altera a contratação de hardware. Adicione `-Private` para novos repositórios privados.

O Space apresenta geração, comparação e documentação do componente de imagens. Ele roda no hardware do Hugging Face; a GPU do computador local não é utilizada pelo serviço hospedado. A versão CPU é funcional, mas pode ser lenta. Os demais módulos da plataforma local não fazem parte desta demonstração inicial.

## Verificação dos componentes

```powershell
.venv\Scripts\python.exe -m pytest -q
$env:IGNIS_TEST_HTTP = '1'
.venv\Scripts\python.exe -m pytest -q
cd frontend
npm run build
```

Os testes padrão verificam handlers e esquemas da API, falhas explícitas sem respostas fictícias, bloqueio de GPU, isolamento de caminhos, divisão dos dados e seleção dos downloads. Defina IGNIS_TEST_HTTP=1 para também testar o transporte HTTP em ambiente com loopback disponível. Eles não demonstram que inferência ou treinamento funcionaram com os pesos reais.

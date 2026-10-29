# Operação e limites

- Inferência é local e usa somente pesos registrados como instalados. Downloads são explícitos.
- O inpainting usa o UNet de 4 canais do SD 1.5 com mesclagem latente da área preservada; não equivale a um checkpoint especializado de 9 canais. A interface aceita máscara branca para a região editada.
- A compreensão de documentos renderiza até quatro páginas de PDF como imagens. Não é RAG nem OCR garantido; preserve referência às páginas e incerteza de leitura.
- O primeiro dataset público é um recorte de obras em domínio público do Art Institute of Chicago. Somente registros `is_public_domain=true`, com imagem, são elegíveis; arquive metadados de origem e termos. Legendas derivam do título, tipo de obra e materiais; precisam de inspeção para especialização posterior.
- A preparação determina splits por hash de pixels para evitar vazamento por arquivos duplicados. Aumentar a coleção requer novo manifesto e nova execução de treinamento.
- Validação usa loss de denoising com ruído e timestep fixos; relatório visual usa prompts e sementes fixos. Não há FID ou CLIP sem implementação e amostra adequada.
- Antes de publicar adaptador, inclua base, licenças, manifesto do dataset, configuração, passos realmente executados e comparação visual. Não envie pesos-base, datasets ou credenciais por padrão.

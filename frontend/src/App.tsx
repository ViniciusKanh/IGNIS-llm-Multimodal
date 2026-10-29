import { useEffect, useRef, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';

type Module = 'chat' | 'studio' | 'lab' | 'models';
type Model = { key: string; repo: string; license: string; status: string; revision?: string; bytes?: number };
type Status = { cuda: boolean; gpu: string | null; torch: string; vram_allocated_mb: number; vram_free_mb: number | null; ram_free_mb: number; loaded: string[] | null; models: Model[] };
type Message = { role: string; content: string; model?: string; attachments?: string[] };
type Generation = { url: string; prompt: string; seed: number; seconds: number; adapter: string | null };
type Metric = { step: number; train_loss: number; validation_loss?: number };
type Run = { name: string; progress: {step: number; status: string; last_metrics?: Metric} | null; metrics: Metric[]; checkpoints: string[] };
type Lab = { dataset: { train: number; validation: number; fingerprint?: string; source?: {name: string} }; runs: Run[]; adapters: string[]; evaluations: {name: string; url: string; samples: string[]}[] };
type Job = { id: string; action: string; status: string; model?: string };
type Config = {name: string; rank: number; resolution: number; batch_size: number; gradient_accumulation: number; learning_rate: number; max_steps: number; warmup_steps: number; checkpoint_every: number; validation_every: number; patience: number; min_steps: number; seed: number};
const names = {chat: 'Chat Multimodal', studio: 'Estúdio de Imagens', lab: 'Laboratório', models: 'Model Manager'};
const descriptions = {chat: 'Uma imagem. Novas possibilidades de compreensão.', studio: 'Transforme ideias em imagens. Explore, edite e compare.', lab: 'Dados, adaptação e evidências em um só lugar.', models: 'Controle o que está instalado e o que ocupa sua GPU.'};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail ?? data));
  return data as T;
}
const post = (body?: unknown): RequestInit => ({method: 'POST', ...(body === undefined ? {} : {headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})});

function Icon({name, size=20}: {name: string; size?: number}) {
  const paths: Record<string, ReactNode> = {
    chat: <><path d="M4 4h16v12H9l-5 4V4Z"/><path d="M8 8h8M8 12h5"/></>,
    studio: <><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m3 17 6-6 4 4 3-3 5 5"/></>,
    lab: <><path d="M9 3h6M10 3v7l-6 9q-1 2 2 2h12q3 0 2-2l-6-9V3M8 15h8"/></>,
    models: <><path d="m12 2 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 17l9 5 9-5"/></>,
    arrow: <><path d="M5 12h14m-6-6 6 6-6 6"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] ?? paths.models}</svg>;
}
function Empty({title, children}: {title: string; children: ReactNode}) { return <div className="empty"><span className="orb"><Icon name="studio" size={30}/></span><h3>{title}</h3><p>{children}</p></div>; }
function LossChart({rows}: {rows: Metric[]}) {
  if (!rows.length) return <Empty title="Aguardando o primeiro experimento">A curva aparecerá quando houver atualizações reais do otimizador.</Empty>;
  const all = rows.flatMap(r => [r.train_loss, ...(r.validation_loss === undefined ? [] : [r.validation_loss])]);
  const low = Math.min(...all), high = Math.max(...all), span = Math.max(high-low, 0.001);
  const x = (step: number) => 48 + (step / Math.max(rows[rows.length-1].step, 1))*640;
  const y = (loss: number) => 180 - (loss-low)/span*145;
  return <svg className="loss-chart" viewBox="0 0 720 220" role="img" aria-label="Loss de treino e validação por passo">
    {[0,1,2,3].map(i=><g key={i}><path d={`M48 ${35+i*48} H690`} stroke="#e8edf2"/><text x="3" y={40+i*48} fontSize="11" fill="#728196">{(high-span*i/3).toFixed(3)}</text></g>)}
    <polyline points={rows.map(r=>`${x(r.step)},${y(r.train_loss)}`).join(' ')} fill="none" stroke="#00a8bc" strokeWidth="2"/>
    {rows.filter(r=>r.validation_loss!==undefined).map(r=><circle key={r.step} cx={x(r.step)} cy={y(r.validation_loss!)} r="4" fill="#183950"><title>{`Passo ${r.step}: validação ${r.validation_loss!.toFixed(5)}`}</title></circle>)}
    <text x="48" y="210" fontSize="11" fill="#728196">0</text><text x="630" y="210" fontSize="11" fill="#728196">{rows[rows.length-1].step} passos</text>
  </svg>;
}

export default function App() {
  const [module, setModule] = useState<Module>('chat');
  const [status, setStatus] = useState<Status | null>(null);
  const [lab, setLab] = useState<Lab | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState('');
  const [session, setSession] = useState(() => localStorage.getItem('ignis-session') || crypto.randomUUID());
  const [messages, setMessages] = useState<Message[]>([]);
  const [prompt, setPrompt] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [vision, setVision] = useState('smolvlm');
  const [imagePrompt, setImagePrompt] = useState('');
  const [negative, setNegative] = useState('');
  const [mode, setMode] = useState('text-to-image');
  const [seed, setSeed] = useState(42);
  const [size, setSize] = useState(512);
  const [steps, setSteps] = useState(25);
  const [strength, setStrength] = useState(.65);
  const [adapter, setAdapter] = useState('');
  const [initialImage, setInitialImage] = useState<File | null>(null);
  const [mask, setMask] = useState<File | null>(null);
  const [gallery, setGallery] = useState<Generation[]>([]);
  const [log, setLog] = useState('');
  const [config, setConfig] = useState<Config | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  async function refresh() {
    const [s, l, j, g] = await Promise.all([api<Status>('/status'), api<Lab>('/lab'), api<Job[]>('/jobs'), api<Generation[]>('/gallery')]);
    setStatus(s); setLab(l); setJobs(j); setGallery(g);
    if (j[0]) setLog((await api<{log: string}>(`/jobs/${j[0].id}/log`)).log);
  }
  useEffect(()=>{ let active=true; const tick=()=>refresh().catch(e=>{if(active) setError(String(e.message));}); void tick(); const timer=setInterval(tick,5000); return ()=>{active=false; clearInterval(timer);}; },[]);
  useEffect(()=>{localStorage.setItem('ignis-session', session); api<Message[]>(`/chat/${session}`).then(setMessages).catch(e=>setError(e.message));},[session]);
  useEffect(()=>{api<Config>('/config').then(setConfig).catch(e=>setError(e.message));},[]);
  useEffect(()=>{bottom.current?.scrollIntoView({behavior:'smooth', block:'nearest'});},[messages]);

  async function action(label: string, operation: ()=>Promise<unknown>) {
    setBusy(label); setError(''); setNotice('');
    try { await operation(); await refresh(); }
    catch(e) {setError(e instanceof Error ? e.message : String(e));}
    finally {setBusy('');}
  }
  async function queue(actionName: string, model='all') {
    await action('Iniciando operação', async()=>{await api('/jobs', post({action: actionName,model}));setNotice('Operação iniciada. Acompanhe o registro no Laboratório.');});
  }
  async function send(event: FormEvent) {
    event.preventDefault(); if (!prompt.trim()) return;
    await action('Analisando a solicitação', async()=>{
      const data=new FormData(); data.set('prompt',prompt);data.set('session',session);data.set('model',vision);files.forEach(f=>data.append('files',f));
      await api('/chat',{method:'POST',body:data}); setMessages(await api<Message[]>(`/chat/${session}`));setPrompt('');setFiles([]);
    });
  }
  async function generate(event: FormEvent) {
    event.preventDefault();
    await action('Gerando imagem', async()=>{
      const data=new FormData(); Object.entries({prompt:imagePrompt,negative_prompt:negative,mode,seed,width:size,height:size,steps,strength,adapter}).forEach(([k,v])=>data.set(k,String(v)));
      if(initialImage)data.set('image',initialImage);if(mask)data.set('mask',mask);
      await api('/images',{method:'POST',body:data}); setNotice('Imagem gerada e salva na galeria.');
    });
  }
  const working = Boolean(busy);
  const queued = jobs.some(j=>['queued','running'].includes(j.status));
  const latest = lab?.runs[lab.runs.length-1];
  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e=>{e.preventDefault();setModule('chat');}}><span className="brand-mark">i<span/></span><div>IGNIS<small>MULTIMODAL LAB</small></div></a>
      <div className="nav-label">ESPAÇO DE TRABALHO</div>
      <nav>{(Object.keys(names) as Module[]).map((key,i)=><button key={key} className={module===key?'active':''} onClick={()=>setModule(key)}><Icon name={key}/><span>{names[key]}</span><small>0{i+1}</small></button>)}</nav>
      <div className="sidebar-foot"><div className="local-dot"/> Ambiente local<p>Seus modelos.<br/>Seu laboratório.</p><span>IGNIS / EXPERIMENTAL 0.1</span></div>
    </aside>
    <main>
      <header className="topbar"><span>Workspace <b>/ {names[module]}</b></span><div><span className={`dot ${status?.cuda?'on':''}`}/>{status?.cuda?'CUDA disponível':'CUDA indisponível'}<span className="pill">LOCAL</span></div></header>
      <section className="page">
        <div className="page-heading"><div><div className="eyebrow">INTELIGÊNCIA EM EXPLORAÇÃO</div><h1>{names[module]}</h1><p>{descriptions[module]}</p></div><span className="outline-tag">IGNIS {module==='chat'?'VISION':module==='studio'?'IMAGE + EDIT':module==='lab'?'RESEARCH':'CORE'}</span></div>
        {error&&<div className="alert" role="alert"><strong>Operação indisponível</strong><span>{error}</span><button onClick={()=>setError('')} aria-label="Fechar erro">×</button></div>}
        {notice&&<div className="notice" role="status">{notice}</div>}
        {working&&<div className="progress-line" role="status">{busy}…</div>}

        {module==='chat'&&<div className="chat-grid">
          <section className="panel chat-panel"><div className="panel-head"><span><Icon name="chat"/> Conversa</span><button className="text-button" disabled={working} onClick={()=>{setSession(crypto.randomUUID());setFiles([]);}}>Nova conversa +</button></div>
            <div className="messages">{messages.length===0?<div className="chat-welcome"><div className="vision-emblem"><Icon name="chat" size={40}/></div><div className="eyebrow">COMECE COM UMA PERGUNTA</div><h2>Veja além da imagem.</h2><p>Envie uma imagem ou documento e explore<br/>o contexto com o Ignis Vision.</p><div className="suggestions">{['Descreva os elementos desta imagem.','Leia o documento e resuma os pontos principais.'].map(q=><button key={q} onClick={()=>setPrompt(q)}>{q}<Icon name="arrow" size={15}/></button>)}</div></div>:messages.map((m,i)=><article key={i} className={`message ${m.role}`}><span className="message-label">{m.role==='user'?'VOCÊ':'IGNIS VISION'}</span><p>{m.content}</p>{m.attachments?.map(url=><a href={url} key={url} target="_blank" rel="noreferrer"><img className="chat-attachment" src={url} alt="Imagem anexada à conversa"/></a>)}</article>)}<div ref={bottom}/></div>
            <form className="composer" onSubmit={send}><textarea aria-label="Mensagem" placeholder="Pergunte sobre a imagem ou documento…" value={prompt} onChange={e=>setPrompt(e.target.value)} required maxLength={16000}/><div className="composer-bottom"><label className="file-button">+ Anexar imagem ou PDF<input type="file" accept="image/*,application/pdf" multiple onChange={e=>setFiles(Array.from(e.target.files??[]).slice(0,4))}/></label><button className="primary" disabled={working||!prompt.trim()} type="submit">Enviar <Icon name="arrow" size={16}/></button></div>{files.length>0&&<div className="file-list">{files.map(f=><span key={f.name}>{f.name}</span>)}<button type="button" onClick={()=>setFiles([])}>Remover</button></div>}</form>
          </section>
          <aside className="context-column"><section className="panel padding"><div className="eyebrow">MOTOR DE COMPREENSÃO</div><h3>Ignis Vision</h3><label>Modelo<select value={vision} onChange={e=>setVision(e.target.value)}><option value="smolvlm">SmolVLM2 · 500M</option><option value="qwen">Qwen3-VL · 2B</option></select></label><div className="info-line"><span>Disponibilidade</span><b>{status?.models.find(m=>m.key===vision)?.status==='installed'?'Instalado':'Requer download'}</b></div><p className="muted">Compreensão visual baseada em modelos públicos. Até quatro imagens ou páginas de PDF por mensagem.</p></section><section className="note-card"><span>01 / EXPLORAR</span><h3>Do detalhe<br/>à interpretação.</h3><p>Descreva cenas, consulte documentos e investigue relações visuais.</p><div className="grid-art"><i/><i/><i/><i/></div></section></aside>
        </div>}

        {module==='studio'&&<div className="studio-grid"><form className="panel padding studio-form" onSubmit={generate}><div className="eyebrow">PARÂMETROS DE GERAÇÃO</div><h3>Nova composição</h3><label>Operação<select value={mode} onChange={e=>setMode(e.target.value)}><option value="text-to-image">Texto → imagem</option><option value="image-to-image">Imagem → imagem</option><option value="inpainting">Inpainting com máscara</option></select></label><label>Modelo / adaptador<select value={adapter} onChange={e=>setAdapter(e.target.value)}><option value="">Stable Diffusion 1.5 · base</option>{lab?.adapters.map(a=><option key={a} value={a}>SD 1.5 + {a}</option>)}</select></label><label>Prompt<textarea value={imagePrompt} onChange={e=>setImagePrompt(e.target.value)} placeholder="Descreva a cena, os materiais e a iluminação…" required maxLength={4000}/></label><label>Prompt negativo<input value={negative} onChange={e=>setNegative(e.target.value)} placeholder="Elementos que deseja evitar"/></label><div className="form-row"><label>Seed<input type="number" min={0} max={4294967295} value={seed} onChange={e=>setSeed(Number(e.target.value))}/></label><label>Resolução<select value={size} onChange={e=>setSize(Number(e.target.value))}>{[256,384,512,640,768].map(n=><option key={n} value={n}>{n} × {n}</option>)}</select></label></div><label>Passos · {steps}<input type="range" min={5} max={80} value={steps} onChange={e=>setSteps(Number(e.target.value))}/></label>{mode!=='text-to-image'&&<><label>Imagem inicial<input type="file" accept="image/*" required onChange={e=>setInitialImage(e.target.files?.[0]??null)}/></label><label>Intensidade · {strength.toFixed(2)}<input type="range" min={.1} max={.95} step={.05} value={strength} onChange={e=>setStrength(Number(e.target.value))}/></label></>}{mode==='inpainting'&&<label>Máscara · branco = editar<input type="file" accept="image/*" required onChange={e=>setMask(e.target.files?.[0]??null)}/></label>}<button className="primary full" disabled={working} type="submit">Gerar imagem <Icon name="arrow" size={17}/></button><p className="muted">O SD 1.5 responde melhor a prompts em inglês. As gerações preservam a origem e a semente.</p></form><section className="panel"><div className="panel-head"><span>Galeria de experimentos</span><span className="pill">{gallery.length} IMAGENS</span></div>{gallery.length===0?<Empty title="Sua próxima ideia começa aqui">As imagens geradas aparecerão nesta galeria, com prompt e seed.</Empty>:<div className="gallery">{gallery.map(g=><article key={g.url}><a href={g.url} target="_blank" rel="noreferrer"><img src={g.url} alt={g.prompt}/></a><div><p>{g.prompt}</p><span>Seed {g.seed} · {g.seconds}s · {g.adapter?'LoRA':'Base'}</span><a href={g.url} download>Baixar PNG ↗</a></div></article>)}</div>}</section></div>}

        {module==='lab'&&<><div className="metric-grid"><section className="panel metric"><span>IMAGENS DE TREINO</span><b>{lab?.dataset.train??0}</b><small>{lab?.dataset.source?.name??'Dataset ainda não preparado'}</small></section><section className="panel metric"><span>IMAGENS DE VALIDAÇÃO</span><b>{lab?.dataset.validation??0}</b><small>Separação por hash de conteúdo</small></section><section className="panel metric"><span>ATUALIZAÇÕES DO OTIMIZADOR</span><b>{latest?.progress?.step??'—'}</b><small>{latest?.progress?.status??'Nenhum treinamento executado'}</small></section></div><div className="lab-grid"><section className="panel padding"><div className="section-title"><div><div className="eyebrow">ACOMPANHAMENTO</div><h3>Loss por passo</h3></div><span className="muted">Ciano: treino · Azul: validação</span></div><LossChart rows={latest?.metrics??[]}/>{latest&&<p className="muted">{latest.name} · Checkpoints: {latest.checkpoints.join(', ')||'nenhum'}</p>}<div className="action-row"><button className="secondary" disabled={working||queued} onClick={()=>queue('prepare')}>Preparar dataset público</button><button className="primary" disabled={working||queued} onClick={()=>queue('train')}>Iniciar / retomar LoRA</button><button className="secondary" disabled={working||queued||!lab?.adapters.length} onClick={()=>queue('evaluate')}>Comparar base × LoRA</button><button className="text-button" disabled={!jobs.some(j=>j.action==='train'&&j.status==='running')} onClick={()=>action('Solicitando parada',()=>api('/training/stop',post()))}>Salvar e parar</button></div></section><section className="panel padding"><div className="eyebrow">EXPERIMENTO</div><h3>ignis-image-lora-v1</h3>{config&&<><label>Nome do experimento<input value={config.name} onChange={e=>setConfig({...config,name:e.target.value})}/></label><label>Passos máximos<input type="number" min={300} max={10000} value={config.max_steps} onChange={e=>setConfig({...config,max_steps:Number(e.target.value)})}/></label><label>Learning rate<input type="number" min={0.000001} max={0.001} step={0.00001} value={config.learning_rate} onChange={e=>setConfig({...config,learning_rate:Number(e.target.value)})}/></label><p className="muted">Rank {config.rank} · FP16 · {config.resolution}px<br/>Batch 1 · Acumulação {config.gradient_accumulation}</p><button className="secondary full" disabled={queued||working} onClick={()=>action('Salvando configuração',async()=>{await api('/config',post(config));setNotice('Configuração salva. Use um novo nome se alterar uma execução já iniciada.');})}>Salvar configuração</button></>}<p className="muted">A loss mede denoising. A qualidade final exige inspeção das amostras pareadas.</p></section></div><section className="panel logs"><div className="panel-head"><span>Registro de execução</span><span className="pill">{jobs[0]?.status??'SEM OPERAÇÕES'}</span></div><pre>{log||'Nenhuma operação executada. Downloads e treinamentos registram a saída aqui.'}</pre></section>{lab?.evaluations.map(e=><section className="panel padding evaluation" key={e.name}><h3>{e.name}</h3><a href={e.url} target="_blank" rel="noreferrer">Abrir relatório JSON ↗</a>{e.samples.map(s=><img src={s} key={s} alt="Comparação visual pareada entre base e LoRA"/>)}</section>)}</>}

        {module==='models'&&<><div className="resource-card"><div><div className="eyebrow">RECURSOS DO SISTEMA</div><h2>{status?.gpu??'PyTorch sem acesso à GPU'}</h2><p>{status?.torch?`PyTorch ${status.torch}`:'Conectando ao backend'} · RAM livre {status?Math.round(status.ram_free_mb/1024):'—'} GB</p></div><div className="resource-value"><b>{status?.vram_free_mb==null?'—':(status.vram_free_mb/1024).toFixed(1)}<small> GB</small></b><span>VRAM LIVRE</span></div></div><div className="section-title model-title"><h3>Modelos do laboratório</h3><button className="secondary" disabled={working||queued} onClick={()=>action('Descarregando modelo',()=>api('/models/unload',post()))}>Liberar memória</button></div><div className="model-grid">{status?.models.map(m=><section className="panel model-card" key={m.key}><div className="model-icon"><Icon name={m.key==='sd15'?'studio':'models'} size={28}/><span className={`status-tag ${m.status==='installed'?'ready':''}`}>{m.status==='installed'?'INSTALADO':m.status.toUpperCase()}</span></div><h3>{m.key==='sd15'?'Stable Diffusion 1.5':m.key==='qwen'?'Qwen3-VL · 2B':'SmolVLM2 · 500M'}</h3><p>{m.key==='sd15'?'Geração e edição de imagens':m.key==='qwen'?'Compreensão visual alternativa':'Compreensão visual leve'}</p><a href={`https://huggingface.co/${m.repo}`} target="_blank" rel="noreferrer">{m.repo} ↗</a><div className="model-details"><span>Licença <b>{m.license}</b></span><span>Revisão <b>{m.revision?.slice(0,10)??'—'}</b></span><span>Em memória <b>{status.loaded?.[0]===m.key?'Sim':'Não'}</b></span></div><div className="action-row"><button className="secondary" disabled={working||queued} onClick={()=>queue('download',m.key)}>{m.status==='installed'?'Verificar download':'Baixar modelo'}</button><button className="primary" disabled={working||queued||m.status!=='installed'} onClick={()=>action('Carregando modelo',()=>api('/models/load',post({model:m.key})))}>Carregar</button></div></section>)}</div><p className="muted attribution">Modelos-base públicos. A contribuição IGNIS está na integração, no orquestrador, na interface e nos adaptadores treinados.</p></>}
        <footer>IGNIS MULTIMODAL <span>Experimentação local · Resultados verificáveis</span></footer>
      </section>
    </main>
  </div>;
}

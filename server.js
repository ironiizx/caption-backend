import 'dotenv/config';
import express from 'express';
import fetch from 'node-fetch';
import { pipeline, env } from '@xenova/transformers';

// Forzar ONNX vía WASM
env.backends.onnx = 'wasm';
env.useBrowserCache = false;
env.allowLocalModels = true;

const PORT = process.env.PORT || 8080;
const MODEL_ID = process.env.MODEL_ID || 'Xenova/vit-gpt2-image-captioning';

// Cache local (ideal si montás un Volume en Railway)
process.env.XENOVA_USE_LOCAL_MODELS = process.env.XENOVA_USE_LOCAL_MODELS ?? '1';
process.env.TRANSFORMERS_CACHE = process.env.TRANSFORMERS_CACHE ?? './.models-cache';

const app = express();
app.use(express.json({ limit: '20mb' }));

// CORS simple
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', req.headers.origin || '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

// Raíz y health
app.get('/', (_req, res) => {
  res.type('text').send('Caption backend up. Try POST /caption or GET /health');
});

app.get('/health', (_req, res) => {
  res.json({ ok: true, model: MODEL_ID });
});

// ===================
//  ÚNICO getPipe()
// ===================
let pipePromise = null;
let ready = false;

async function getPipe() {
  if (!pipePromise) {
    console.log('🔄 Cargando modelo:', MODEL_ID);
    pipePromise = pipeline('image-to-text', MODEL_ID)
      .then(p => {
        ready = true;
        console.log('✅ Modelo listo');
        return p;
      })
      .catch(err => {
        ready = false;
        pipePromise = null;
        throw err;
      });
  }
  return pipePromise;
}

// Helper de imagen
async function getImageBytes(input) {
  if (!input) throw new Error('Falta image_url o image_base64');
  if (/^https?:\/\//i.test(input)) {
    const r = await fetch(input, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    if (!r.ok) throw new Error(`fetch ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  }
  if (input.startsWith('data:image/')) {
    return Buffer.from(input.split(',')[1], 'base64');
  }
  return Buffer.from(input, 'base64');
}

// Endpoint principal
app.post('/caption', async (req, res) => {
  try {
    const { image_url, image_base64 } = req.body || {};
    const bytes = await getImageBytes(image_url || image_base64);

    const pipe = await getPipe(); // asegura modelo cargado
    const t0 = Date.now();
    const out = await pipe(bytes, { max_new_tokens: 40 });

    res.json({
      caption: out?.[0]?.generated_text ?? '',
      model: MODEL_ID,
      latency_ms: Date.now() - t0
    });
  } catch (e) {
    console.error('Error /caption:', e);
    res.status(500).json({ error: String(e.message || e) });
  }
});

// Warmup & ready
app.get('/warmup', async (_req, res) => {
  try {
    await getPipe();
    res.json({ ready: true, model: MODEL_ID });
  } catch (e) {
    console.error('Warmup error:', e);
    res.status(500).json({ ready: false, error: String(e.message || e) });
  }
});

app.get('/ready', (_req, res) => {
  res.json({ ready, model: MODEL_ID });
});

// Calentar al arrancar (no bloquea)
getPipe().catch(() => {});

// Arrancar server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ Server escuchando en 0.0.0.0:${PORT}`);
});

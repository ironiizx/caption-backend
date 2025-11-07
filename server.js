import 'dotenv/config';
import express from 'express';
import fetch from 'node-fetch';
import { pipeline, env } from '@xenova/transformers';

// Forzar ONNX por WASM (evita binarios nativos)
env.backends.onnx = 'wasm';
env.useBrowserCache = false;
env.allowLocalModels = true;
// NO toques env.backends.onnx.wasm.numThreads acá

const PORT = process.env.PORT || 3000;
const MODEL_ID = process.env.MODEL_ID || 'Xenova/vit-gpt2-image-captioning';

// cache local opcional (mejor si usás un Volume)
process.env.XENOVA_USE_LOCAL_MODELS = process.env.XENOVA_USE_LOCAL_MODELS ?? '1';
process.env.TRANSFORMERS_CACHE = process.env.TRANSFORMERS_CACHE ?? './.models-cache';

const app = express();
app.use(express.json({ limit: '20mb' }));

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', req.headers.origin || '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

app.get('/health', (_req, res) => {
  res.json({ ok: true, model: MODEL_ID });
});

let pipePromise = null;
async function getPipe() {
  if (!pipePromise) {
    console.log('🔄 Cargando modelo:', MODEL_ID);
    pipePromise = pipeline('image-to-text', MODEL_ID);
  }
  return pipePromise;
}

async function getImageBytes(input) {
  if (!input) throw new Error('Falta image_url o image_base64');
  if (typeof input === 'string' && /^https?:\/\//i.test(input)) {
    const r = await fetch(input, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    if (!r.ok) throw new Error(`fetch ${r.status} al descargar imagen`);
    return new Uint8Array(await r.arrayBuffer());
  }
  if (typeof input === 'string' && input.startsWith('data:image/')) {
    const b64 = input.split(',')[1];
    return Buffer.from(b64, 'base64');
  }
  if (typeof input === 'string') return Buffer.from(input, 'base64');
  throw new Error('Formato de imagen no soportado');
}

app.post('/caption', async (req, res) => {
  try {
    const { image_url, image_base64 } = req.body || {};
    const bytes = await getImageBytes(image_url || image_base64);

    const pipe = await getPipe();
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

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ Server escuchando en 0.0.0.0:${PORT}`);
});

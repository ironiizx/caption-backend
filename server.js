import 'dotenv/config';
import express from 'express';
import fetch from 'node-fetch';
import { pipeline, env } from '@xenova/transformers';

// ===== FIX Railway: forzar backend ONNX a WASM (sin binarios nativos) =====
env.backends.onnx = 'wasm';            // clave: evita onnxruntime-node
env.useBrowserCache = false;           // no usa IndexedDB
env.allowLocalModels = true;           // permite cache local en disco (si lo tenés)
// opcional: menos hilos wasm si estás justo de RAM/CPU
env.backends.onnx.wasm.numThreads = 1;

// Puerto que provee Railway
const PORT = process.env.PORT || 3000;
const MODEL_ID = process.env.MODEL_ID || 'Xenova/vit-gpt2-image-captioning';

// Opcional: ruta de cache local (mejor si añadís un Volume en Railway y setear TRANSFORMERS_CACHE=/opt/models)
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

// Health MUY liviano (no carga el modelo para no tumbar el arranque)
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

// Descarga imagen (URL o dataURL/base64)
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
  if (typeof input === 'string') {
    // asume base64 crudo
    return Buffer.from(input, 'base64');
  }
  throw new Error('Formato de imagen no soportado');
}

app.post('/caption', async (req, res) => {
  try {
    const { image_url, image_base64 } = req.body || {};
    const bytes = await getImageBytes(image_url || image_base64);

    const pipe = await getPipe(); // carga perezosa (no en el arranque)
    const t0 = Date.now();
    const out = await pipe(bytes, { max_new_tokens: 40 });
    const ms = Date.now() - t0;

    res.json({ caption: out?.[0]?.generated_text ?? '', model: MODEL_ID, latency_ms: ms });
  } catch (e) {
    console.error('Error /caption:', e);
    res.status(500).json({ error: String(e.message || e) });
  }
});

// Importante: escuchar en el puerto de Railway y en 0.0.0.0
app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ Server escuchando en 0.0.0.0:${PORT}`);
  // NO precargamos el modelo en el arranque para evitar timeouts/crashes
});

// server.js
import 'dotenv/config';
import express from 'express';
import fetch from 'node-fetch';
import { pipeline, env, RawImage } from '@xenova/transformers';

// ------------ Xenova / ONNX (WASM) ------------
env.backends.onnx = 'wasm';
env.useBrowserCache = false;
env.allowLocalModels = true;

// Si tenés token de HF (opcional, para modelos que lo pidan)
env.HF_HUB_TOKEN = process.env.HF_TOKEN || undefined;

// Cache local (opcional, ideal si montás un Volume en Railway)
process.env.XENOVA_USE_LOCAL_MODELS =
  process.env.XENOVA_USE_LOCAL_MODELS ?? '1';
process.env.TRANSFORMERS_CACHE =
  process.env.TRANSFORMERS_CACHE ?? './.models-cache';

// ------------ Config ------------
const PORT = process.env.PORT || 8080;
const PRIMARY_MODEL =
  process.env.MODEL_ID || 'Xenova/vit-gpt2-image-captioning';
const FALLBACK_MODEL = process.env.FALLBACK_MODEL || 'Xenova/tiny-vit-gpt2';

// Estado global del pipeline
let pipePromise = null; // promesa del pipeline (compartida)
let ready = false;      // indica si ya está listo
let activeModel = null; // modelo realmente cargado

// ------------ App / middlewares ------------
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
  res.json({ ok: true, model: activeModel || PRIMARY_MODEL });
});

// ------------ Carga perezosa con fallback ------------
async function getPipe() {
  if (pipePromise) return pipePromise;

  const markReady = (p, modelId) => {
    activeModel = modelId;
    ready = true;
    console.log(`✅ Modelo listo: ${modelId}`);
    return p;
  };

  console.log('🔄 Cargando modelo (primary):', PRIMARY_MODEL);
  pipePromise = pipeline('image-to-text', PRIMARY_MODEL)
    .then(p => markReady(p, PRIMARY_MODEL))
    .catch(async (err) => {
      console.error(
        '⚠️  Error cargando primary, aplico fallback al tiny:',
        err?.message || err
      );
      console.log('🔁 Cargando modelo (fallback):', FALLBACK_MODEL);
      try {
        const p2 = await pipeline('image-to-text', FALLBACK_MODEL);
        return markReady(p2, FALLBACK_MODEL);
      } catch (err2) {
        pipePromise = null;
        ready = false;
        activeModel = null;
        throw err2;
      }
    });

  return pipePromise;
}

// Endpoints de warmup / ready
app.get('/warmup', async (_req, res) => {
  try {
    await getPipe(); // carga (primary o tiny)
    res.json({ ready: true, model: activeModel });
  } catch (e) {
    console.error('Warmup error:', e);
    res.status(500).json({ ready: false, error: String(e?.message || e) });
  }
});

app.get('/ready', (_req, res) => {
  res.json({ ready, model: activeModel });
});

// -------- Util: obtener bytes crudos de la imagen --------
async function getImageBytes(input) {
  if (!input) throw new Error('Falta image_url o image_base64');

  // URL http(s)
  if (typeof input === 'string' && /^https?:\/\//i.test(input)) {
    const r = await fetch(input, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    if (!r.ok) throw new Error(`fetch ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  }

  // data URL
  if (typeof input === 'string' && input.startsWith('data:image/')) {
    const b64 = input.split(',')[1];
    return new Uint8Array(Buffer.from(b64, 'base64'));
  }

  // base64 “puro”
  if (typeof input === 'string') {
    return new Uint8Array(Buffer.from(input, 'base64'));
  }

  throw new Error('Formato de imagen no soportado');
}

// ====== Endpoint principal /caption ======
app.post('/caption', async (req, res) => {
  try {
    const { image_url, image_base64, max_new_tokens } = req.body || {};

    // 1) bytes (Uint8Array)
    const bytes = await getImageBytes(image_url || image_base64);

    // 2) usar RawImage.read (compatibilidad Node)
    const img = await RawImage.read(bytes);

    // 3) asegurar modelo cargado
    const pipe = await getPipe();

    // 4) inferencia
    const t0 = Date.now();
    const out = await pipe(img, {
      max_new_tokens: typeof max_new_tokens === 'number' ? max_new_tokens : 40,
    });

    // 5) respuesta
    res.json({
      caption: out?.[0]?.generated_text ?? '',
      model: activeModel || PRIMARY_MODEL,
      latency_ms: Date.now() - t0,
    });
  } catch (e) {
    console.error('Error /caption:', e);
    res.status(500).json({ error: String(e?.message || e) });
  }
});


// ------------ Arranque (precalienta sin bloquear) ------------
getPipe().catch(() => { /* no bloquea el start; /warmup puede reintentar */ });

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ Server escuchando en 0.0.0.0:${PORT}`);
});

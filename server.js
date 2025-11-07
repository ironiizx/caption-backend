// server.js
import 'dotenv/config';
import express from 'express';
import fetch from 'node-fetch';
import { pipeline, env } from '@xenova/transformers';

// ------------ Xenova / ONNX (WASM) ------------
env.backends.onnx = 'wasm';
env.useBrowserCache = false;
env.allowLocalModels = true;

env.HF_HUB_TOKEN = process.env.HF_TOKEN || undefined;

// Cache local (opcional, ideal si montás un Volume)
process.env.XENOVA_USE_LOCAL_MODELS =
  process.env.XENOVA_USE_LOCAL_MODELS ?? '1';
process.env.TRANSFORMERS_CACHE =
  process.env.TRANSFORMERS_CACHE ?? './.models-cache';

// ------------ Config ------------
const PORT = process.env.PORT || 8080;
const PRIMARY_MODEL = process.env.MODEL_ID || 'Xenova/vit-gpt2-image-captioning';
const FALLBACK_MODEL = 'Xenova/tiny-vit-gpt2';

// Estado global del pipeline
let pipePromise = null;   // promesa del pipeline
let ready = false;        // listo para inferir
let activeModel = null;   // cuál quedó cargado (primary o tiny)

// ------------ App ------------
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

  // Func que setea estado común
  const finish = (p, modelId) => {
    activeModel = modelId;
    ready = true;
    console.log(`✅ Modelo listo: ${modelId}`);
    return p;
  };

  console.log('🔄 Cargando modelo (primary):', PRIMARY_MODEL);
  pipePromise = pipeline('image-to-text', PRIMARY_MODEL)
    .then(p => finish(p, PRIMARY_MODEL))
    .catch(async (err) => {
      // Log y fallback al tiny
      console.error('⚠️  Error cargando primary, aplico fallback al tiny:', err?.message || err);
      console.log('🔁 Cargando modelo (fallback):', FALLBACK_MODEL);
      try {
        const p2 = await pipeline('image-to-text', FALLBACK_MODEL);
        return finish(p2, FALLBACK_MODEL);
      } catch (err2) {
        // Si también falla, reseteo y propago
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
    await getPipe(); // espera a que termine de cargar (primary o tiny)
    res.json({ ready: true, model: activeModel });
  } catch (e) {
    console.error('Warmup error:', e);
    res.status(500).json({ ready: false, error: String(e?.message || e) });
  }
});

app.get('/ready', (_req, res) => {
  res.json({ ready, model: activeModel });
});

// ------------ Util para obtener bytes de imagen ------------
async function getImageBytes(input) {
  if (!input) throw new Error('Falta image_url o image_base64');

  // URL http(s)
  if (typeof input === 'string' && /^https?:\/\//i.test(input)) {
    const r = await fetch(input, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    if (!r.ok) throw new Error(`fetch ${r.status} al descargar imagen`);
    return new Uint8Array(await r.arrayBuffer());
  }
  // DataURL base64
  if (typeof input === 'string' && input.startsWith('data:image/')) {
    const b64 = input.split(',')[1];
    return Buffer.from(b64, 'base64');
  }
  // Base64 "crudo"
  if (typeof input === 'string') {
    return Buffer.from(input, 'base64');
  }

  throw new Error('Formato de imagen no soportado');
}

// ------------ Caption endpoint ------------
app.post('/caption', async (req, res) => {
  try {
    const { image_url, image_base64, max_new_tokens } = req.body || {};
    const src = image_url || image_base64;
    if (!src) {
      return res.status(400).json({ error: 'Send image_url or image_base64' });
    }

    // Descargar la imagen en bytes reales
    const response = await fetch(src);
    if (!response.ok) {
      throw new Error(`Failed to fetch image: ${response.statusText}`);
    }

    // 👇 convertir a ArrayBuffer y luego a Uint8Array
    const arrayBuffer = await response.arrayBuffer();
    const bytes = new Uint8Array(arrayBuffer);

    const pipe = await getPipe(); // modelo cargado
    const t0 = Date.now();

    // 👇 forma correcta de llamada
    const out = await pipe(
      { inputs: bytes },
      {
        max_new_tokens:
          typeof max_new_tokens === 'number' ? max_new_tokens : 40,
      }
    );

    res.json({
      caption: out?.[0]?.generated_text ?? '',
      model: typeof activeModel !== 'undefined' ? activeModel : MODEL_ID,
      latency_ms: Date.now() - t0,
    });
  } catch (e) {
    console.error('Error /caption:', e);
    res.status(500).json({ error: e?.message || String(e) });
  }
});



// ------------ Arranque (precalienta sin bloquear) ------------
getPipe().catch(() => { /* no bloquea el start; /warmup puede reintentar */ });

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ Server escuchando en 0.0.0.0:${PORT}`);
});

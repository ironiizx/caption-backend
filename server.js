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

// -------- Util para obtener bytes de imagen --------
async function getImageBytes(input) {
  if (!input) throw new Error('Falta image_url o image_base64');

  // Si es URL http(s)
  if (typeof input === 'string' && /^https?:\/\//i.test(input)) {
    const r = await fetch(input, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    if (!r.ok) throw new Error(`fetch ${r.status}`);
    // 🔹 Convertir a Uint8Array explícitamente
    return new Uint8Array(await r.arrayBuffer());
  }

  // Si es data:image/...
  if (typeof input === 'string' && input.startsWith('data:image/')) {
    const b64 = input.split(',')[1];
    const buf = Buffer.from(b64, 'base64');
    // 🔹 Convertir Buffer a Uint8Array
    return new Uint8Array(buf);
  }

  // Si es base64 puro
  if (typeof input === 'string') {
    const buf = Buffer.from(input, 'base64');
    return new Uint8Array(buf);
  }

  throw new Error('Formato de imagen no soportado');
}


// ====== Endpoint principal /caption ======
app.post('/caption', async (req, res) => {
  try {
    const { image_url, image_base64, max_new_tokens } = req.body || {};

    // Obtener bytes de la imagen (desde URL o base64)
    const bytes = await getImageBytes(image_url || image_base64);

    // Asegurarse de que el modelo esté cargado
    const pipe = await getPipe();

    const t0 = Date.now();
    // Ejecutar el modelo pasando directamente los bytes
    const out = await pipe(bytes, {
    max_new_tokens: typeof max_new_tokens === 'number' ? max_new_tokens : 40,
    });


    // Responder con resultado
    res.json({
      caption: out?.[0]?.generated_text ?? '',
      model: MODEL_ID,
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

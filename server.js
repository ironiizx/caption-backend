// server.js
import 'dotenv/config';
import express from 'express';
import fetch from 'node-fetch'; // <--- VUELVE A AÑADIRSE
import { pipeline, env } from '@xenova/transformers';
// import { RawImage } from '@xenova/transformers'; // <--- ELIMINADO
import sharp from 'sharp';

// ------------ Xenova / ONNX (WASM) ------------
env.backends.onnx = 'wasm';
env.useBrowserCache = false;
env.allowLocalModels = true;
env.HF_HUB_TOKEN = process.env.HF_TOKEN || undefined;
process.env.XENOVA_USE_LOCAL_MODELS =
  process.env.XENOVA_USE_LOCAL_MODELS ?? '1';
process.env.TRANSFORMERS_CACHE =
  process.env.TRANSFORMERS_CACHE ?? './.models-cache';

// ------------ Config ------------
const PORT = process.env.PORT || 8080;
const PRIMARY_MODEL =
  process.env.MODEL_ID || 'Xenova/vit-gpt2-image-captioning';
const FALLBACK_MODEL = process.env.FALLBACK_MODEL || 'Xenova/tiny-vit-gpt2';

let pipePromise = null;
let ready = false;
let activeModel = null;

// ------------ App / middlewares ------------
const app = express();
app.use(express.json({ limit: '20mb' }));

// ... (El código de CORS, / y /health sigue igual) ...
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', req.headers.origin || '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});
app.get('/', (_req, res) => {
  res.type('text').send('Caption backend up. Try POST /caption or GET /health');
});
app.get('/health', (_req, res) => {
  res.json({ ok: true, model: activeModel || PRIMARY_MODEL });
});


// ... (El código de getPipe(), /warmup y /ready sigue igual) ...
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
    .then((p) => markReady(p, PRIMARY_MODEL))
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
app.get('/warmup', async (_req, res) => {
  try {
    await getPipe();
    res.json({ ready: true, model: activeModel });
  } catch (e) {
    console.error('Warmup error:', e);
    res.status(500).json({ ready: false, error: String(e?.message || e) });
  }
});
app.get('/ready', (_req, res) => {
  res.json({ ready, model: activeModel });
});


// <--- AÑADIDO: Vuelve el helper para obtener un Buffer
async function getImageBuffer(input) {
  if (!input) throw new Error('Falta image_url o image_base64');

  // URL http(s)
  if (typeof input === 'string' && /^https?:\/\//i.test(input)) {
    const r = await fetch(input, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    if (!r.ok) throw new Error(`fetch ${r.status}`);
    return Buffer.from(await r.arrayBuffer());
  }
  // data URL
  if (typeof input === 'string' && input.startsWith('data:image/')) {
    const b64 = input.split(',')[1];
    return Buffer.from(b64, 'base64');
  }
  // base64 “puro”
  if (typeof input === 'string') {
    return Buffer.from(input, 'base64');
  }
  throw new Error('Formato de imagen no soportado');
}

// <--- AÑADIDO: Helper de Saturación
function getSaturation(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (l === 0 || max === min) return 0;
  return (max - min) / (1 - Math.abs(2 * l - 1));
}


// ====== Endpoint principal /caption (TOTALMENTE REESCRITO) ======
app.post('/caption', async (req, res) => {
  try {
    const { image_url, image_base64, max_new_tokens } = req.body || {};
    const t0 = Date.now(); // Inicia el timer aquí

    // 1) Obtener la imagen como un Buffer de Node.js
    const buffer = await getImageBuffer(image_url || image_base64);

    // 2) Cargar en sharp UNA SOLA VEZ
    const sharpImg = sharp(buffer);

    // 3) --- Tareas en Paralelo (más rápido) ---

    // Tarea A: Calcular Métricas
    const metricsPromise = (async () => {
      // A.1: Stats (brillo, contraste, color)
      const stats = await sharpImg.stats();

      // A.2: Densidad de Bordes (Canny)
      const edgeBuffer = await sharpImg
        .clone() // importante clonar para no afectar otras tareas
        .greyscale()
        .canny(5, 20, 10)
        .raw()
        .toBuffer();

      let edgeSum = 0;
      for (let i = 0; i < edgeBuffer.length; i++) {
        edgeSum += edgeBuffer[i];
      }
      const edgeDensity = edgeSum / edgeBuffer.length / 255;

      // A.3: Organizar métricas
      const [rStats, gStats, bStats] = stats.channels;
      const dominantColor = `rgb(${stats.dominant.r}, ${stats.dominant.g}, ${stats.dominant.b})`;
      const brightness = (rStats.mean + gStats.mean + bStats.mean) / 3 / 255;
      const contrast = (rStats.stdev + gStats.stdev + bStats.stdev) / 3 / 255;
      const saturation = getSaturation(rStats.mean, gStats.mean, bStats.mean);

      return {
        brightness, contrast, saturation, dominantColor, edgeDensity,
        textRatio: 0.0, // Sigue siendo un placeholder
      };
    })();

    // Tarea B: Preparar datos para la IA
    const aiDataPromise = (async () => {
      // Extraer píxeles crudos (RGBA) que la IA entiende
      const { data, info } = await sharpImg
        .clone()
        .ensureAlpha() // Asegura 4 canales (RGBA)
        .raw()
        .toBuffer({ resolveWithObject: true });
      
      return {
        data: data, // Buffer de píxeles
        width: info.width,
        height: info.height,
      };
    })();

    // Tarea C: Asegurar que el modelo de IA esté listo
    const pipePromise = getPipe();

    // 4) Esperar a que todo termine
    const [metrics, img, pipe] = await Promise.all([
      metricsPromise,
      aiDataPromise,
      pipePromise,
    ]);

    // 5) Inferencia (ahora 'img' es el objeto de píxeles crudos)
    const out = await pipe(img, {
      max_new_tokens: typeof max_new_tokens === 'number' ? max_new_tokens : 40,
    });

    // 6) Respuesta
    res.json({
      caption: out?.[0]?.generated_text ?? '',
      model: activeModel || PRIMARY_MODEL,
      latency_ms: Date.now() - t0,
      metrics: metrics,
    });
  } catch (e) {
    console.error('Error /caption:', e);
    res.status(500).json({ error: String(e?.message || e) });
  }
});


// ------------ Arranque (precalienta sin bloquear) ------------
getPipe().catch(() => {
  /* no bloquea el start; /warmup puede reintentar */
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ Server escuchando en 0.0.0.0:${PORT}`);
});
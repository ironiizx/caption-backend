// server.js
import 'dotenv/config';
import express from 'express';
import { pipeline, env, RawImage } from '@xenova/transformers';
import sharp from 'sharp'; // <--- AÑADIDO

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

// <--- AÑADIDO: Función helper para calcular Saturación desde RGB
function getSaturation(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (l === 0 || max === min) return 0;
  // Formula para HSL
  return (max - min) / (1 - Math.abs(2 * l - 1));
}

// ====== Endpoint principal /caption ======
app.post('/caption', async (req, res) => {
  try {
    const { image_url, image_base64, max_new_tokens } = req.body || {};

    // 1) Determinar el input para RawImage.read
    let imageInput;
    if (image_url) {
      imageInput = image_url;
    } else if (image_base64) {
      if (image_base64.startsWith('data:image/')) {
        imageInput = image_base64;
      } else {
        imageInput = `data:image/jpeg;base64,${image_base64}`;
      }
    } else {
      throw new Error('Falta image_url o image_base64 en el body');
    }

    // 2) Dejar que RawImage lea el input (URL o Data URL)
    const img = await RawImage.read(imageInput);

    // --- 3) INICIO DE MÉTRICAS CON SHARP --- // <--- AÑADIDO
    const rawPixels = {
      raw: {
        width: img.width,
        height: img.height,
        channels: 4, // RawImage nos da RGBA, sharp necesita saber esto
      },
    };

    // Tarea 1: Obtener estadísticas (brillo, contraste, color)
    const stats = await sharp(img.data, rawPixels).stats();

    // Tarea 2: Obtener densidad de bordes (Canny)
    const edgeBuffer = await sharp(img.data, rawPixels)
      .greyscale() // Canny funciona en blanco y negro
      .canny(5, 20, 10) // (radius, sigma, low_thresh, high_thresh)
      .raw()
      .toBuffer();

    // Calcular el promedio de bordes
    let edgeSum = 0;
    for (let i = 0; i < edgeBuffer.length; i++) {
      edgeSum += edgeBuffer[i]; // píxeles de borde son 255
    }
    const edgeDensity = edgeSum / edgeBuffer.length / 255; // normalizado 0-1

    // Organizar métricas
    const [rStats, gStats, bStats] = stats.channels;
    const dominantColor = `rgb(${stats.dominant.r}, ${stats.dominant.g}, ${stats.dominant.b})`;
    
    // Usamos el promedio de los promedios de canal (0-255)
    const brightness = (rStats.mean + gStats.mean + bStats.mean) / 3 / 255; // normalizado 0-1
    // Usamos el promedio de las desviaciones estándar
    const contrast = (rStats.stdev + gStats.stdev + bStats.stdev) / 3 / 255; // normalizado 0-1
    // Usamos el helper para el promedio de color
    const saturation = getSaturation(rStats.mean, gStats.mean, bStats.mean);

    const metrics = {
      brightness: brightness,
      contrast: contrast,
      saturation: saturation,
      dominantColor: dominantColor,
      edgeDensity: edgeDensity,
      textRatio: 0.0 // Dejado en 0.0 como placeholder
    };
    // --- 3) FIN DE MÉTRICAS CON SHARP ---

    // 4) asegurar modelo cargado
    const pipe = await getPipe();

    // 5) inferencia
    const t0 = Date.now();
    const out = await pipe(img, {
      max_new_tokens: typeof max_new_tokens === 'number' ? max_new_tokens : 40,
    });

    // 6) respuesta
    res.json({
      caption: out?.[0]?.generated_text ?? '',
      model: activeModel || PRIMARY_MODEL,
      latency_ms: Date.now() - t0,
      metrics: metrics, // <--- CORREGIDO (enviamos las métricas)
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
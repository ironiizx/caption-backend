// server.js
import 'dotenv/config';
import express from 'express';
import fetch from 'node-fetch';
import { pipeline, env, RawImage } from '@xenova/transformers'; 
import sharp from 'sharp';


env.backends.onnx = 'wasm';
env.useBrowserCache = false;
env.allowLocalModels = true;
env.HF_HUB_TOKEN = process.env.HF_TOKEN || undefined;
process.env.XENOVA_USE_LOCAL_MODELS =
  process.env.XENOVA_USE_LOCAL_MODELS ?? '1';
process.env.TRANSFORMERS_CACHE =
  process.env.TRANSFORMERS_CACHE ?? './.models-cache';


const PORT = process.env.PORT || 8080;
const PRIMARY_MODEL =
  process.env.MODEL_ID || 'Xenova/vit-gpt2-image-captioning';
const FALLBACK_MODEL = process.env.FALLBACK_MODEL || 'Xenova/tiny-vit-gpt2';

let pipePromise = null;
let ready = false;
let activeModel = null;


const app = express();
app.use(express.json({ limit: '20mb' }));

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

async function getImageBuffer(input) {
  if (!input) throw new Error('Falta image_url o image_base64');
  if (typeof input === 'string' && /^https?:\/\//i.test(input)) {
    const r = await fetch(input, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    if (!r.ok) throw new Error(`fetch ${r.status}`);
    return Buffer.from(await r.arrayBuffer());
  }
  if (typeof input === 'string' && input.startsWith('data:image/')) {
    const b64 = input.split(',')[1];
    return Buffer.from(b64, 'base64');
  }
  if (typeof input === 'string') {
    return Buffer.from(input, 'base64');
  }
  throw new Error('Formato de imagen no soportado');
}

function getSaturation(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (l === 0 || max === min) return 0;
  return (max - min) / (1 - Math.abs(2 * l - 1));
}


app.post('/caption', async (req, res) => {
  try {
    const { image_url, image_base64, max_new_tokens } = req.body || {};
    const t0 = Date.now();

    
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
    
    
    const buffer = await getImageBuffer(image_url || image_base64);


    const metricsPromise = (async () => {
      const stats = await sharp(buffer).stats();
      const edgeDensity = 0.0; 

      const [rStats, gStats, bStats] = stats.channels;
      const dominantColor = `rgb(${stats.dominant.r}, ${stats.dominant.g}, ${stats.dominant.b})`;
      const brightness = (rStats.mean + gStats.mean + bStats.mean) / 3 / 255;
      const contrast = (rStats.stdev + gStats.stdev + bStats.stdev) / 3 / 255;
      const saturation = getSaturation(rStats.mean, gStats.mean, bStats.mean);

      return {
        brightness, contrast, saturation, dominantColor,
        edgeDensity: edgeDensity,
        textRatio: 0.0,
      };
    })();

    
    const captionPromise = (async () => {
      const pipe = await getPipe();
      
      const img = await RawImage.read(imageInput); 
      const out = await pipe(img, {
        max_new_tokens: typeof max_new_tokens === 'number' ? max_new_tokens : 40,
      });
      return out;
    })();

    
    const [metrics, out] = await Promise.all([
      metricsPromise,
      captionPromise,
    ]);

   
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



getPipe().catch(() => {

});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ Server escuchando en 0.0.0.0:${PORT}`);
});
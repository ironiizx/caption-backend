import 'dotenv/config';
import express from 'express';
import { pipeline } from '@xenova/transformers';
import fetch from 'node-fetch';

const PORT = process.env.PORT || 3000;
// Volvemos al modelo que SÍ es 100% compatible con tu sistema
const MODEL_ID = process.env.MODEL_ID || 'Xenova/vit-gpt2-image-captioning';

const app = express();
app.use(express.json({ limit: '20mb' }));

// Middleware para CORS
app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', req.headers.origin || '*');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    if (req.method === 'OPTIONS') return res.sendStatus(204);
    next();
});

let pipePromise;
async function getPipe() {
    if (!pipePromise) {
        console.log(`Cargando modelo de IA: ${MODEL_ID}`);
        pipePromise = pipeline('image-to-text', MODEL_ID);
    }
    return await pipePromise;
}

// Endpoint simplificado que solo maneja URLs
app.post('/caption', async (req, res) => {
    try {
        // n8n nos envía un JSON, así que esperamos image_url
        const { image_url } = req.body || {};
        if (!image_url) {
            return res.status(400).json({ error: 'Falta image_url' });
        }
        
        const pipe = await getPipe();
        const input = String(image_url);

        const captionResult = await pipe(input, { max_new_tokens: 30 });
        const caption = captionResult?.[0]?.generated_text ?? '';

        // Devolvemos solo lo que el frontend (index.html) sabe mostrar
        res.json({ caption, model: MODEL_ID });

    } catch (err) {
        console.error("Error en /caption:", err);
        res.status(500).json({ error: String(err) });
    }
});

app.listen(PORT, () => {
    console.log(`✅ Servidor escuchando en http://localhost:${PORT}`);
    console.log(`   Modelo en uso: ${MODEL_ID}`);
    getPipe(); // Inicia la carga del modelo al arrancar para más rapidez
});
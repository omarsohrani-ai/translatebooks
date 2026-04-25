// api/translate.js — Vercel Serverless Function
// PDF Translation Proxy with Queue System using Google Gemini

const { GoogleGenerativeAI } = require("@google/generative-ai");

// Simple in-memory queue (persists per serverless instance)
const queue = [];
const jobResults = {};
let isProcessing = false;

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

// Language map
const LANGUAGES = {
  en: "English", ar: "Arabic", fr: "French", es: "Spanish",
  de: "German", it: "Italian", pt: "Portuguese", ru: "Russian",
  zh: "Chinese (Simplified)", ja: "Japanese", ko: "Korean",
  tr: "Turkish", fa: "Persian (Farsi)", ur: "Urdu", hi: "Hindi",
  bn: "Bengali", id: "Indonesian", ms: "Malay", nl: "Dutch",
  pl: "Polish", sv: "Swedish", no: "Norwegian", da: "Danish",
  fi: "Finnish", cs: "Czech", ro: "Romanian", hu: "Hungarian",
  el: "Greek", he: "Hebrew", th: "Thai", vi: "Vietnamese",
  uk: "Ukrainian", bg: "Bulgarian", hr: "Croatian", sk: "Slovak",
  ca: "Catalan", az: "Azerbaijani", kk: "Kazakh", uz: "Uzbek",
  sw: "Swahili", am: "Amharic", so: "Somali", ha: "Hausa"
};

async function processJob(job) {
  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

  const targetLang = LANGUAGES[job.targetLang] || "English";
  const sourceLang = job.sourceLang === "auto" ? "the source language (auto-detect)" : (LANGUAGES[job.sourceLang] || "Arabic");

  const parts = [];

  // Add page images
  for (let i = 0; i < job.images.length; i++) {
    parts.push({ text: `[Page ${job.pageNums[i]}]` });
    parts.push({
      inlineData: {
        mimeType: "image/jpeg",
        data: job.images[i]
      }
    });
  }

  parts.push({
    text: `You are a professional translator. Translate the text in these scanned document page images from ${sourceLang} into ${targetLang}.

Instructions:
- Read all visible text carefully from each scanned page image
- Provide a complete, accurate translation into ${targetLang}
- If a page is blank or has no meaningful text, write: [Blank page]
- For title pages or publication info, translate those too
- Preserve paragraph structure and formatting
- For technical/specialized terms, keep the original term in parentheses on first use
- Do NOT add commentary, footnotes, or explanations

Format your response EXACTLY as follows for each page (no extra text outside these blocks):

${job.pageNums.map(n => `===PAGE ${n}===\n[translation here]\n===END===`).join('\n\n')}

Translate every page shown above.`
  });

  const result = await model.generateContent(parts);
  const text = result.response.text();
  return text;
}

async function runQueue() {
  if (isProcessing || queue.length === 0) return;
  isProcessing = true;

  while (queue.length > 0) {
    const job = queue[0];
    jobResults[job.id] = { status: "processing", queuePos: 0 };

    try {
      const raw = await processJob(job);
      const parsed = {};
      for (const pn of job.pageNums) {
        const re = new RegExp(`===PAGE ${pn}===([\\s\\S]*?)===END===`, 'i');
        const m = raw.match(re);
        parsed[pn] = m ? m[1].trim() : `[Could not extract page ${pn}]`;
      }
      jobResults[job.id] = { status: "done", translations: parsed };
    } catch (err) {
      jobResults[job.id] = { status: "error", error: err.message };
    }

    queue.shift();

    // Update queue positions for remaining jobs
    queue.forEach((j, idx) => {
      if (jobResults[j.id]) jobResults[j.id].queuePos = idx + 1;
    });

    // Small delay between batches to respect rate limits
    await new Promise(r => setTimeout(r, 1000));
  }

  isProcessing = false;
}

// Clean up old results (older than 1 hour)
function cleanupOldResults() {
  const now = Date.now();
  for (const id in jobResults) {
    if (jobResults[id].timestamp && now - jobResults[id].timestamp > 3600000) {
      delete jobResults[id];
    }
  }
}

module.exports = async (req, res) => {
  // CORS headers
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();

  const { action } = req.query;

  // GET /api/translate?action=languages — return available languages
  if (req.method === "GET" && action === "languages") {
    return res.json({ languages: LANGUAGES });
  }

  // GET /api/translate?action=status&id=xxx — check job status
  if (req.method === "GET" && action === "status") {
    const { id } = req.query;
    if (!id || !jobResults[id]) {
      const queuePos = queue.findIndex(j => j.id === id);
      if (queuePos >= 0) {
        return res.json({ status: "queued", queuePos: queuePos + 1, queueLength: queue.length });
      }
      return res.status(404).json({ error: "Job not found" });
    }
    return res.json(jobResults[id]);
  }

  // GET /api/translate?action=queue — queue info
  if (req.method === "GET" && action === "queue") {
    return res.json({ queueLength: queue.length, isProcessing });
  }

  // POST /api/translate — submit a translation job
  if (req.method === "POST") {
    cleanupOldResults();

    if (!GEMINI_API_KEY) {
      return res.status(500).json({ error: "GEMINI_API_KEY not configured on server." });
    }

    const { images, pageNums, sourceLang, targetLang } = req.body;

    if (!images || !pageNums || images.length !== pageNums.length) {
      return res.status(400).json({ error: "Invalid request: images and pageNums required." });
    }

    if (images.length > 5) {
      return res.status(400).json({ error: "Max 5 pages per request." });
    }

    const id = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const job = { id, images, pageNums, sourceLang: sourceLang || "auto", targetLang: targetLang || "en", timestamp: Date.now() };

    queue.push(job);
    jobResults[id] = { status: "queued", queuePos: queue.length, queueLength: queue.length, timestamp: Date.now() };

    // Start processing queue (non-blocking)
    runQueue().catch(console.error);

    return res.json({ id, status: "queued", queuePos: queue.length, queueLength: queue.length });
  }

  return res.status(405).json({ error: "Method not allowed" });
};

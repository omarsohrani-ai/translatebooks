// api/translate.js — Vercel Serverless Function
// PDF Translation Proxy with Queue System using Google Gemini

const { GoogleGenerativeAI } = require("@google/generative-ai");

const queue = [];
const jobResults = {};
let isProcessing = false;

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

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

// ── Robust page extractor ──────────────────────────────────────────────────
// gemini-2.5-flash is a thinking model that can:
//   • wrap output in markdown code fences (```...```)
//   • add thinking tags (<think>...</think>)
//   • omit the final ===END=== on the last page of a batch
//   • add extra whitespace around delimiters
// This function handles all those cases gracefully.
function extractPages(raw, pageNums) {
  // 1. Strip markdown code fences
  let text = raw.replace(/^```[\w]*\s*/m, '').replace(/\s*```\s*$/m, '');

  // 2. Strip XML-style thinking tags (some model versions include these)
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
  text = text.replace(/<thinking>[\s\S]*?<\/thinking>/gi, '');

  // 3. Normalise delimiter spacing (model sometimes outputs ===PAGE 11 === etc.)
  text = text.replace(/===\s*PAGE\s+(\d+)\s*===/gi, (_, n) => `===PAGE ${n}===`);
  text = text.replace(/===\s*END\s*===/gi, '===END===');

  const parsed = {};

  for (const pn of pageNums) {
    // Match content between ===PAGE N=== and either ===END===, the next ===PAGE,
    // or end-of-string — so a missing final ===END=== never breaks extraction.
    const re = new RegExp(
      `===PAGE ${pn}===\\s*([\\s\\S]*?)(?:===END===|===PAGE \\d+===|$)`,
      'i'
    );
    const m = text.match(re);
    if (m && m[1].trim()) {
      // Remove any trailing ===END=== fragment the lazy match pulled in
      parsed[pn] = m[1].replace(/===END===/gi, '').trim();
    } else {
      parsed[pn] = `[Could not extract page ${pn}]`;
    }
  }

  return parsed;
}

// ── Translation job ────────────────────────────────────────────────────────
async function processJob(job) {
  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

  // gemini-2.5-flash-lite: best for free tier — 1,000 RPD, 15 RPM.
  // gemini-2.5-flash has higher quality but only 20 RPD free (exhausted fast).
  // To upgrade: enable billing and switch back to "gemini-2.5-flash" (~$0.20/400 pages).
  const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

  const targetLang = LANGUAGES[job.targetLang] || "English";
  const sourceLang = job.sourceLang === "auto"
    ? "the source language (auto-detect)"
    : (LANGUAGES[job.sourceLang] || "the source language");

  const parts = [];

  for (let i = 0; i < job.images.length; i++) {
    parts.push({ text: `[Page ${job.pageNums[i]}]` });
    parts.push({
      inlineData: { mimeType: "image/jpeg", data: job.images[i] }
    });
  }

  parts.push({
    text: `You are a professional literary translator. Translate all text from these scanned document pages from ${sourceLang} into ${targetLang}.

STRICT RULES — follow every one without exception:
- Translate EVERY single word completely into ${targetLang}
- Do NOT leave any word in the source language or any other language
- Do NOT use parentheses to preserve original terms — translate everything
- If the source text contains words from a third language, translate those too into ${targetLang}
- Preserve the original paragraph structure and line breaks
- If a page is blank or has no meaningful text, write exactly: [Blank page]
- Do NOT add translator notes, footnotes, commentary, or explanations
- Do NOT wrap your response in markdown code blocks or any other formatting
- Do NOT write anything outside the page blocks below

Format your response EXACTLY as shown — every page must have both delimiters:

${job.pageNums.map(n =>
  `===PAGE ${n}===\n[your ${targetLang} translation here]\n===END===`
).join('\n\n')}

Translate every page shown above.`
  });

  const result = await model.generateContent(parts);
  return result.response.text();
}

// ── Queue runner ───────────────────────────────────────────────────────────
async function runQueue() {
  if (isProcessing || queue.length === 0) return;
  isProcessing = true;

  while (queue.length > 0) {
    const job = queue[0];
    jobResults[job.id] = { status: "processing", queuePos: 0 };

    try {
      const raw = await processJob(job);
      const parsed = extractPages(raw, job.pageNums);
      jobResults[job.id] = { status: "done", translations: parsed };
    } catch (err) {
      jobResults[job.id] = { status: "error", error: err.message };
    }

    queue.shift();

    queue.forEach((j, idx) => {
      if (jobResults[j.id]) jobResults[j.id].queuePos = idx + 1;
    });

    // 4 s gap keeps us inside the 15 RPM free-tier limit for gemini-2.5-flash-lite
    await new Promise(r => setTimeout(r, 4000));
  }

  isProcessing = false;
}

// ── Cleanup ────────────────────────────────────────────────────────────────
function cleanupOldResults() {
  const now = Date.now();
  for (const id in jobResults) {
    if (jobResults[id].timestamp && now - jobResults[id].timestamp > 3600000) {
      delete jobResults[id];
    }
  }
}

// ── HTTP handler ───────────────────────────────────────────────────────────
module.exports = async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();

  const { action } = req.query;

  if (req.method === "GET" && action === "languages") {
    return res.json({ languages: LANGUAGES });
  }

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

  if (req.method === "GET" && action === "queue") {
    return res.json({ queueLength: queue.length, isProcessing });
  }

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
    const job = {
      id,
      images,
      pageNums,
      sourceLang: sourceLang || "auto",
      targetLang: targetLang || "en",
      timestamp: Date.now()
    };

    queue.push(job);
    jobResults[id] = {
      status: "queued",
      queuePos: queue.length,
      queueLength: queue.length,
      timestamp: Date.now()
    };

    runQueue().catch(console.error);

    return res.json({
      id,
      status: "queued",
      queuePos: queue.length,
      queueLength: queue.length
    });
  }

  return res.status(405).json({ error: "Method not allowed" });
};

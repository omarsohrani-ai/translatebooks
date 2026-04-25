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
// Uses a split-based approach instead of per-page regex.
// Per-page regex has a known flaw: searching for page N can partially match
// page NN (e.g. regex for page 1 can interfere with ===PAGE 11===).
// Split-based parsing avoids this entirely by slicing the text at every
// ===PAGE N=== boundary, then indexing by page number.
function extractPages(raw, pageNums) {
  // 1. Strip markdown code fences (model sometimes wraps output)
  let text = raw
    .replace(/^```[\w]*\n?/m, '')
    .replace(/\n?```\s*$/m, '');

  // 2. Strip thinking / reasoning tags
  text = text
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, '');

  // 3. Normalise delimiters — handle extra spaces, mixed case, etc.
  //    Add a newline before each ===PAGE so the split below always works
  text = text
    .replace(/===\s*PAGE\s+(\d+)\s*===/gi, (_, n) => `\n===PAGE ${n}===\n`)
    .replace(/===\s*END\s*===/gi, '\n===END===\n');

  // 4. Split on ===PAGE N=== markers — gives us one segment per page
  //    Segment format: "N\n<content>" (the split keeps the capture group)
  const segments = text.split(/\n===PAGE (\d+)===\n/);
  // segments[0]  = text before first page (discard)
  // segments[1]  = page number string "9"
  // segments[2]  = page 9 content
  // segments[3]  = page number string "10"  ... etc.

  const blocks = {};
  for (let i = 1; i < segments.length - 1; i += 2) {
    const pn  = parseInt(segments[i], 10);
    // Strip any trailing ===END=== from the content block
    const content = segments[i + 1]
      .replace(/\n?===END===\n?/gi, '')
      .trim();
    if (content) blocks[pn] = content;
  }

  // 5. Build result — fall back gracefully for any missing page
  const parsed = {};
  for (const pn of pageNums) {
    parsed[pn] = blocks[pn] || `[Could not extract page ${pn}]`;
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

CRITICAL FORMATTING RULE — this is the most important rule:
You are translating ${job.pageNums.length} separate pages: ${job.pageNums.join(', ')}.
Even if the text flows continuously from one page to the next without a break,
you MUST output a separate ===PAGE N=== block for EVERY single page number listed.
Never merge two pages into one block. Never skip a page number.
Each page image = one ===PAGE N=== block. No exceptions.

Your response must contain EXACTLY ${job.pageNums.length} blocks in this format:

${job.pageNums.map(n =>
  `===PAGE ${n}===\n[your complete ${targetLang} translation of page ${n} here]\n===END===`
).join('\n\n')}

Check your response before finishing: does it contain all ${job.pageNums.length} blocks for pages ${job.pageNums.join(', ')}?`
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

    // Skip jobs older than 10 minutes — browser session likely dead/abandoned.
    // Without this, a stuck job from a previous session blocks every new user.
    if (Date.now() - job.timestamp > 600000) {
      jobResults[job.id] = { status: 'error', error: 'Job expired — session timed out after 10 minutes' };
      queue.shift();
      continue;
    }

    jobResults[job.id] = { status: "processing", queuePos: 0 };

    try {
      const raw = await processJob(job);
      const parsed = extractPages(raw, job.pageNums);

      // ── Per-page retry ──────────────────────────────────────────────────
      // If a page failed extraction (continuous manuscript text causes the
      // model to merge pages under one delimiter), retry that page ALONE.
      // A single-page batch cannot be merged with anything — guaranteed fix.
      const failedPages = job.pageNums.filter(
        pn => parsed[pn] && parsed[pn].startsWith('[Could not extract')
      );

      for (const pn of failedPages) {
        // Find the image for this specific page
        const idx = job.pageNums.indexOf(pn);
        if (idx === -1) continue;

        const singleJob = {
          id: job.id,
          images: [job.images[idx]],
          pageNums: [pn],
          sourceLang: job.sourceLang,
          targetLang: job.targetLang,
          timestamp: job.timestamp
        };

        try {
          // Small pause before retry to avoid back-to-back requests
          await new Promise(r => setTimeout(r, 8000));
          const retryRaw = await processJob(singleJob);
          const retryParsed = extractPages(retryRaw, [pn]);
          if (retryParsed[pn] && !retryParsed[pn].startsWith('[Could not extract')) {
            parsed[pn] = retryParsed[pn];
          }
        } catch (retryErr) {
          // Keep the original failure message if retry also fails
        }
      }

      jobResults[job.id] = { status: "done", translations: parsed };
    } catch (err) {
      jobResults[job.id] = { status: "error", error: err.message };
    }

    queue.shift();

    queue.forEach((j, idx) => {
      if (jobResults[j.id]) jobResults[j.id].queuePos = idx + 1;
    });

    // Organic delay: 50–60 s random jitter between batches.
    // Keeps throughput at ~1 RPM — well inside the 15 RPM free-tier limit —
    // and avoids burst patterns that can trigger quota enforcement.
    // Trade-off: a 400-page book takes ~2 hrs but never hits rate limits.
    const delay = 50000 + Math.floor(Math.random() * 10000); // 50–60 s
    await new Promise(r => setTimeout(r, delay));
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

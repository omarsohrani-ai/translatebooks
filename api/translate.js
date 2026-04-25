// api/translate.js — Vercel Serverless Function
// PDF Translation — Gemini primary, Groq (Llama 4 Scout) automatic fallback
//
// Setup:
//   Required: GEMINI_API_KEY  — aistudio.google.com (free, 20 RPD)
//   Optional: GROQ_API_KEY    — console.groq.com    (free, 1000 RPD, no card)
//
// Behaviour:
//   • Every batch is tried with Gemini first (better OCR on difficult scripts)
//   • If Gemini daily quota is exhausted AND GROQ_API_KEY is set, the server
//     automatically switches to Groq's Llama 4 Scout for the rest of the session
//   • If neither key has quota, returns a clear QUOTA_EXHAUSTED error

const { GoogleGenerativeAI } = require("@google/generative-ai");

const queue       = [];
const jobResults  = {};
let isProcessing  = false;

// Session-level provider flag — flips to 'groq' when Gemini quota is exhausted
let activeProvider = 'gemini';

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GROQ_API_KEY   = process.env.GROQ_API_KEY;

const LANGUAGES = {
  en:"English", ar:"Arabic",  fr:"French",   es:"Spanish",
  de:"German",  it:"Italian", pt:"Portuguese",ru:"Russian",
  zh:"Chinese (Simplified)", ja:"Japanese",  ko:"Korean",
  tr:"Turkish", fa:"Persian (Farsi)", ur:"Urdu", hi:"Hindi",
  bn:"Bengali", id:"Indonesian", ms:"Malay",  nl:"Dutch",
  pl:"Polish",  sv:"Swedish",   no:"Norwegian",da:"Danish",
  fi:"Finnish", cs:"Czech",     ro:"Romanian", hu:"Hungarian",
  el:"Greek",   he:"Hebrew",    th:"Thai",     vi:"Vietnamese",
  uk:"Ukrainian",bg:"Bulgarian",hr:"Croatian", sk:"Slovak",
  ca:"Catalan", az:"Azerbaijani",kk:"Kazakh",  uz:"Uzbek",
  sw:"Swahili", am:"Amharic",   so:"Somali",   ha:"Hausa"
};

// ── Shared translation prompt ──────────────────────────────────────────────
function buildPrompt(targetLang, sourceLang, pageNums) {
  return `You are a professional literary translator. Translate all text from these scanned document pages from ${sourceLang} into ${targetLang}.

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
You are translating ${pageNums.length} separate pages: ${pageNums.join(', ')}.
Even if the text flows continuously from one page to the next without a break,
you MUST output a separate ===PAGE N=== block for EVERY single page number listed.
Never merge two pages into one block. Never skip a page number.
Each page image = one ===PAGE N=== block. No exceptions.

STOP RULE — most important: Translate ONLY the exact text visible in each image.
Once you have translated everything visible on the page, STOP immediately.
Do NOT continue, repeat, or add anything after the visible text ends.
Do NOT repeat any sentence, phrase, or paragraph more than once.

Your response must contain EXACTLY ${pageNums.length} blocks in this format:

${pageNums.map(n =>
  `===PAGE ${n}===\n[your complete ${targetLang} translation of page ${n} here]\n===END===`
).join('\n\n')}

Check your response before finishing: does it contain all ${pageNums.length} blocks for pages ${pageNums.join(', ')}?`;
}

// ── Robust split-based page extractor ─────────────────────────────────────
function extractPages(raw, pageNums) {
  let text = raw
    .replace(/^```[\w]*\n?/m, '').replace(/\n?```\s*$/m, '')
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, '');

  text = text
    .replace(/===\s*PAGE\s+(\d+)\s*===/gi, (_, n) => `\n===PAGE ${n}===\n`)
    .replace(/===\s*END\s*===/gi, '\n===END===\n');

  const segments = text.split(/\n===PAGE (\d+)===\n/);
  const blocks   = {};

  for (let i = 1; i < segments.length - 1; i += 2) {
    const pn      = parseInt(segments[i], 10);
    const content = segments[i + 1].replace(/\n?===END===\n?/gi, '').trim();
    if (content) blocks[pn] = deduplicateContent(content);
  }

  const parsed = {};
  for (const pn of pageNums) {
    parsed[pn] = blocks[pn] || `[Could not extract page ${pn}]`;
  }
  return parsed;
}

// Remove hallucinated repetitions: if a sentence appears 3+ times consecutively,
// keep only the first 2 occurrences and truncate the rest.
function deduplicateContent(text) {
  const lines = text.split('\n');
  const out   = [];
  let lastLine = '', repeatCount = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) { out.push(line); lastLine = ''; repeatCount = 0; continue; }

    if (trimmed === lastLine) {
      repeatCount++;
      if (repeatCount < 2) out.push(line); // allow one repeat at most
      // silently drop further repetitions
    } else {
      out.push(line);
      lastLine    = trimmed;
      repeatCount = 0;
    }
  }

  // Also catch paragraph-level repetition (same sentence repeated in a paragraph)
  const result = out.join('\n');
  // Find any sentence repeated 3+ times and keep only 2 occurrences
  const sentenceRepeat = /(.{30,})\n(\1\n){2,}/g;
  return result.replace(sentenceRepeat, '$1\n$1\n');
}

// ── Provider: Gemini ───────────────────────────────────────────────────────
async function processWithGemini(job) {
  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
  const model = genAI.getGenerativeModel({
    model: "gemini-2.5-flash-lite",
    generationConfig: {
      // Tight token cap per page — prevents hallucination loops on dense/repetitive text.
      // Dense Arabic manuscript page ≈ 300–400 translated words ≈ 400–550 tokens.
      // 800 per page is generous enough for real content, tight enough to cut loops.
      maxOutputTokens: job.pageNums.length * 800,
      temperature: 0.1
    }
  });

  const targetLang = LANGUAGES[job.targetLang] || "English";
  const sourceLang = job.sourceLang === "auto"
    ? "the source language (auto-detect)"
    : (LANGUAGES[job.sourceLang] || "the source language");

  const parts = [];
  for (let i = 0; i < job.images.length; i++) {
    parts.push({ text: `[Page ${job.pageNums[i]}]` });
    parts.push({ inlineData: { mimeType: "image/jpeg", data: job.images[i] } });
  }
  parts.push({ text: buildPrompt(targetLang, sourceLang, job.pageNums) });

  const result = await model.generateContent(parts);
  return result.response.text();
}

// ── Provider: Groq (Llama 4 Scout) ────────────────────────────────────────
// Llama 4 Scout is natively multimodal and lists Arabic as a supported language.
// Free tier: 1,000 RPD, 30,000 TPM — no credit card required.
// Get a key at: console.groq.com
async function processWithGroq(job) {
  const targetLang = LANGUAGES[job.targetLang] || "English";
  const sourceLang = job.sourceLang === "auto"
    ? "the source language (auto-detect)"
    : (LANGUAGES[job.sourceLang] || "the source language");

  // Build OpenAI-compatible content array with interleaved text + images
  const content = [];
  for (let i = 0; i < job.images.length; i++) {
    content.push({ type: "text", text: `[Page ${job.pageNums[i]}]` });
    content.push({
      type: "image_url",
      image_url: { url: `data:image/jpeg;base64,${job.images[i]}` }
    });
  }
  content.push({ type: "text", text: buildPrompt(targetLang, sourceLang, job.pageNums) });

  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method:  "POST",
    headers: {
      "Content-Type":  "application/json",
      "Authorization": `Bearer ${GROQ_API_KEY}`
    },
    body: JSON.stringify({
      model:      "meta-llama/llama-4-scout-17b-16e-instruct",
      messages:   [{ role: "user", content }],
      max_tokens: job.pageNums.length * 800,
      temperature: 0.1
    })
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    const msg = err?.error?.message || `Groq HTTP ${response.status}`;
    throw new Error(`[Groq] ${msg}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || "";
}

// ── Dispatcher: picks provider, handles per-page retry ────────────────────
async function processJob(job) {
  const raw = activeProvider === 'groq'
    ? await processWithGroq(job)
    : await processWithGemini(job);

  const parsed = extractPages(raw, job.pageNums);

  // Per-page retry: pages that weren't extracted are resubmitted alone
  // (single-page batches cannot be merged — guaranteed extraction)
  const failed = job.pageNums.filter(
    pn => parsed[pn]?.startsWith('[Could not extract')
  );

  for (const pn of failed) {
    const idx = job.pageNums.indexOf(pn);
    if (idx === -1) continue;
    const single = {
      ...job, images: [job.images[idx]], pageNums: [pn]
    };
    try {
      await new Promise(r => setTimeout(r, 8000));
      const retryRaw    = activeProvider === 'groq'
        ? await processWithGroq(single)
        : await processWithGemini(single);
      const retryParsed = extractPages(retryRaw, [pn]);
      if (retryParsed[pn] && !retryParsed[pn].startsWith('[Could not extract')) {
        parsed[pn] = retryParsed[pn];
      }
    } catch (_) { /* keep original failure */ }
  }

  return parsed;
}

// ── Helper: classify quota errors ─────────────────────────────────────────
function isGeminiDailyQuota(msg) {
  return msg.includes('GenerateRequestsPerDayPerProjectPerModel') ||
         (msg.includes('429') && msg.includes('limit: 20'));
}
function isGroqDailyQuota(msg) {
  return msg.toLowerCase().includes('groq') &&
         (msg.includes('429') || msg.includes('rate_limit_exceeded') ||
          msg.includes('tokens_per_day') || msg.includes('requests_per_day'));
}
// Gemini RECITATION = safety filter blocked output (common with religious/classical texts)
// Groq doesn't have this filter — automatically fall back when this happens.
function isRecitation(msg) {
  return msg.includes('RECITATION') || msg.includes('recitation') ||
         msg.includes('SAFETY') || msg.includes('finish_reason: SAFETY');
}

// ── Queue runner ───────────────────────────────────────────────────────────
async function runQueue() {
  if (isProcessing || queue.length === 0) return;
  isProcessing = true;

  while (queue.length > 0) {
    const job = queue[0];

    // Expire abandoned sessions after 3 minutes (not 10)
    // Shorter timeout means zombie jobs from failed sessions clear faster
    if (Date.now() - job.timestamp > 180000) {
      jobResults[job.id] = { status: "error", error: "Job expired — session timed out" };
      queue.shift();
      continue;
    }

    jobResults[job.id] = { status: "processing", queuePos: 0, provider: activeProvider };

    try {
      const translations = await processJob(job);
      jobResults[job.id] = {
        status: "done",
        translations,
        provider: activeProvider   // lets the client show which model was used
      };

    } catch (err) {
      const msg = err.message || '';

      // ── Gemini RECITATION / SAFETY block ─────────────────────────────
      // Gemini blocks classical religious texts (hadith, Quran, Ibn Arabi etc.)
      // as "recitation of protected content". Groq has no such filter.
      if (isRecitation(msg) && activeProvider === 'gemini' && GROQ_API_KEY) {
        console.log('[translate] Gemini RECITATION block — switching to Groq for this job');
        try {
          const savedProvider = activeProvider;
          activeProvider = 'groq';
          const translations = await processJob(job);
          activeProvider = savedProvider; // restore — only this job uses Groq
          jobResults[job.id] = { status: "done", translations, provider: 'groq' };
        } catch (groqErr) {
          jobResults[job.id] = { status: "error", error: `[Recitation block, Groq fallback failed] ${groqErr.message}` };
        }

      // ── Gemini daily quota exhausted ──────────────────────────────────
      } else if (isGeminiDailyQuota(msg)) {
        if (GROQ_API_KEY && activeProvider === 'gemini') {
          activeProvider = 'groq';
          console.log('[translate] Gemini quota exhausted — switching to Groq fallback');
          try {
            const translations = await processJob(job);
            jobResults[job.id] = { status: "done", translations, provider: 'groq' };
          } catch (groqErr) {
            const groqMsg = groqErr.message || '';
            if (isGroqDailyQuota(groqMsg)) {
              jobResults[job.id] = {
                status: "error",
                error: "QUOTA_EXHAUSTED_BOTH: Both Gemini and Groq daily limits reached. Gemini resets at midnight PT. Groq resets at midnight UTC."
              };
              drainQueue("QUOTA_EXHAUSTED_BOTH");
              isProcessing = false;
              return;
            }
            jobResults[job.id] = { status: "error", error: `[Groq fallback failed] ${groqMsg}` };
          }
        } else if (activeProvider === 'groq') {
          jobResults[job.id] = { status: "error", error: msg };
        } else {
          jobResults[job.id] = {
            status: "error",
            error: "QUOTA_EXHAUSTED: Gemini daily limit (20 RPD) reached. Add a GROQ_API_KEY in Vercel env vars for automatic fallback (free, 1000 RPD). Or wait for midnight PT reset."
          };
          drainQueue("QUOTA_EXHAUSTED");
          isProcessing = false;
          return;
        }

      // ── Groq daily quota exhausted ────────────────────────────────────
      } else if (isGroqDailyQuota(msg)) {
        jobResults[job.id] = {
          status: "error",
          error: "QUOTA_EXHAUSTED: Groq daily limit reached. Resets at midnight UTC. Try again tomorrow or enable Gemini billing at aistudio.google.com."
        };
        drainQueue("QUOTA_EXHAUSTED");
        isProcessing = false;
        return;

      // ── Unknown / transient error ─────────────────────────────────────
      } else {
        jobResults[job.id] = { status: "error", error: msg };
      }
    }

    queue.shift();
    queue.forEach((j, idx) => {
      if (jobResults[j.id]) jobResults[j.id].queuePos = idx + 1;
    });

    // 50–60 s organic delay — safe for both Gemini (15 RPM) and Groq (30 RPM)
    const delay = 50000 + Math.floor(Math.random() * 10000);
    await new Promise(r => setTimeout(r, delay));
  }

  isProcessing = false;
}

function drainQueue(reason) {
  while (queue.length > 0) {
    const j = queue.shift();
    if (!jobResults[j.id] || jobResults[j.id].status === 'queued') {
      jobResults[j.id] = { status: "error", error: reason };
    }
  }
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

  if (req.method === "GET" && action === "languages")
    return res.json({ languages: LANGUAGES });

  if (req.method === "GET" && action === "status") {
    const { id } = req.query;
    if (!id || !jobResults[id]) {
      const queuePos = queue.findIndex(j => j.id === id);
      if (queuePos >= 0)
        return res.json({ status: "queued", queuePos: queuePos + 1, queueLength: queue.length });
      return res.status(404).json({ error: "Job not found" });
    }
    return res.json(jobResults[id]);
  }

  if (req.method === "GET" && action === "queue")
    return res.json({ queueLength: queue.length, isProcessing, provider: activeProvider });

  if (req.method === "POST") {
    cleanupOldResults();

    if (!GEMINI_API_KEY && !GROQ_API_KEY)
      return res.status(500).json({ error: "No API keys configured. Set GEMINI_API_KEY or GROQ_API_KEY in Vercel environment variables." });

    // Reset to Gemini at the start of each new translation session
    // (prevents stale fallback state from a previous quota-exhausted session)
    if (GEMINI_API_KEY) activeProvider = 'gemini';
    else if (GROQ_API_KEY) activeProvider = 'groq';

    const { images, pageNums, sourceLang, targetLang } = req.body;
    if (!images || !pageNums || images.length !== pageNums.length)
      return res.status(400).json({ error: "Invalid request: images and pageNums required." });
    if (images.length > 5)
      return res.status(400).json({ error: "Max 5 pages per request." });

    const id  = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const job = {
      id, images, pageNums,
      sourceLang: sourceLang || "auto",
      targetLang: targetLang || "en",
      timestamp:  Date.now()
    };

    queue.push(job);
    jobResults[id] = {
      status: "queued",
      queuePos:    queue.length,
      queueLength: queue.length,
      timestamp:   Date.now()
    };

    runQueue().catch(console.error);
    return res.json({ id, status: "queued", queuePos: queue.length, queueLength: queue.length });
  }

  return res.status(405).json({ error: "Method not allowed" });
};

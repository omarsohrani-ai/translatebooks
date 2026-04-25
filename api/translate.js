// api/translate.js — Vercel Serverless Function
// PDF Translation — Quality-driven routing with per-page escalation
//
// Architecture (Groq-first with Gemini as quality rescue):
//
//   For explicit complex scripts (ar/fa/ur/he):
//     Tier 1: Gemini (vision)        — best OCR, no wasted calls
//     Tier 2: Llama Scout (vision)   — fallback when Gemini quota exhausts
//     Tier 3: Llama OCR + Qwen3      — last resort pipeline
//
//   For everything else (auto-detect, English, French, etc.):
//     Tier 1: Llama Scout (vision)   — saves Gemini quota for the 95% of jobs
//     Tier 2: Gemini (vision)        — PER-PAGE escalation when Llama produces garbage
//     Tier 3: Llama OCR + Qwen3      — last resort pipeline
//
//   Quality detection happens AFTER each page. Garbage = cycles, frequency
//   spam, or extraction failure. Only those specific pages get escalated.
//
// Setup:
//   Required: GEMINI_API_KEY  — aistudio.google.com (20 RPD free)
//   Optional: GROQ_API_KEY    — console.groq.com (1000 RPD free, no card)

const { GoogleGenerativeAI } = require("@google/generative-ai");

const queue       = [];
const jobResults  = {};
let isProcessing  = false;

// Session state — flips to false when Gemini quota is exhausted
let geminiAvailable = true;

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

// EXPLICITLY chosen scripts where Gemini's OCR justifies spending its quota upfront.
// 'auto' is intentionally NOT in this set — most auto users translate normal content.
const KNOWN_HARD_SCRIPTS = new Set(['ar', 'fa', 'ur', 'he', 'am', 'th']);

// ─── Prompt builders ──────────────────────────────────────────────────────
function buildTranslatePrompt(targetLang, sourceLang, pageNums, isGroqModel = false) {
  const groqExtra = isGroqModel ? `
ANTI-HALLUCINATION RULES — CRITICAL for this model:
- Output ONLY text you can physically READ in the image. Nothing else.
- If you cannot read a word clearly, write [unclear] — do NOT guess.
- If a page has only a few lines, your output must be only a few lines.
- NEVER generate filler text, religious commentary, or scholarly prose.
- NEVER repeat any sentence more than once. If you catch yourself repeating, STOP.
- If you have written more than 30 sentences for one page, STOP — something is wrong.
- The source may contain deliberate parallelism. Translate it accurately but do NOT amplify it.
` : '';

  return `You are a professional literary translator. Translate all text from these scanned document pages from ${sourceLang} into ${targetLang}.

STRICT RULES:
- Translate EVERY word completely into ${targetLang}
- Do NOT leave any word in the source language
- Do NOT use parentheses to preserve original terms — translate everything
- Preserve the original paragraph structure and line breaks
- If a page is blank, write exactly: [Blank page]
- Do NOT add translator notes, footnotes, commentary, or explanations
- Do NOT wrap your response in markdown code blocks
- Do NOT write anything outside the page blocks below
${groqExtra}
CRITICAL FORMATTING RULE:
You are translating ${pageNums.length} separate pages: ${pageNums.join(', ')}.
You MUST output a separate ===PAGE N=== block for EVERY page number listed.
Never merge two pages into one block. Never skip a page number.

STOP RULE: Translate ONLY the text visible in each image.
Once everything visible is translated, STOP immediately.
Do NOT continue, repeat, or add anything after the visible text ends.

Your response must contain EXACTLY ${pageNums.length} blocks:

${pageNums.map(n =>
  `===PAGE ${n}===\n[your complete ${targetLang} translation of page ${n} here]\n===END===`
).join('\n\n')}`;
}

function buildOCRPrompt(sourceLang, pageNums) {
  return `You are an OCR engine. Extract ALL text from these scanned document pages exactly as written.

RULES:
- Extract the EXACT text visible — every word, every line
- Preserve the ORIGINAL script (Arabic, Persian, etc.) — do NOT translate
- Preserve paragraph structure and line breaks
- If you cannot read a word, write [unclear]
- If a page is blank, write exactly: [Blank page]
- Do NOT add commentary or interpretation
- Do NOT repeat any text — extract each line exactly once then STOP

Output format — EXACTLY ${pageNums.length} blocks:

${pageNums.map(n =>
  `===PAGE ${n}===\n[exact text from page ${n} in original script]\n===END===`
).join('\n\n')}`;
}

function buildTextTranslatePrompt(targetLang, sourceLang, text) {
  return `You are a professional literary translator. Translate the following text from ${sourceLang} into ${targetLang}.

RULES:
- Translate EVERY word completely into ${targetLang}
- Preserve paragraph structure and line breaks
- Do NOT add notes, commentary, or explanations
- Do NOT repeat any sentence — if the source repeats, translate each occurrence exactly once
- If a section says [unclear] or [Blank page], keep it as-is
- STOP when the text ends

TEXT TO TRANSLATE:
${text}

TRANSLATION:`;
}

// ─── Page extractor ───────────────────────────────────────────────────────
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
    if (content) blocks[pn] = content;
  }

  // Fallback: no PAGE markers but real content → assign to first page
  if (Object.keys(blocks).length === 0 && text.trim().length > 20 && pageNums.length === 1) {
    blocks[pageNums[0]] = text.trim();
  }

  const parsed = {};
  for (const pn of pageNums) {
    parsed[pn] = blocks[pn] || `[Could not extract page ${pn}]`;
  }
  return parsed;
}

// ─── Quality detection: cycles + frequency + length sanity ────────────────
//
// Catches three failure modes:
//   1. Cyclical loops (A→B→C→A→B→C) — most common Llama failure
//   2. Frequency spam (same sentence appearing 5+ times scattered) — softer loops
//   3. Length explosion (output >600 words from one page image) — hallucination
//
function deduplicateContent(text) {
  const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length < 4) return text;

  // Pass 1: frequency cap — max 2 occurrences of any substantial sentence
  const freq  = new Map();
  const pass1 = [];

  for (const line of lines) {
    const key = line.toLowerCase().replace(/\s+/g, ' ').replace(/[.,:;!?]+$/, '').trim();

    if (key.length < 25) { pass1.push(line); continue; }

    const count = (freq.get(key) || 0) + 1;
    freq.set(key, count);
    if (count <= 2) pass1.push(line);
  }

  // Pass 2: cycle detection (A→B→C→A→B→C pattern)
  const result = detectAndRemoveCycles(pass1);

  // Pass 3: hard cap — single page rarely exceeds 500 words
  const words = result.split(/\s+/).length;
  if (words > 600) {
    const truncated = result.split(/\s+/).slice(0, 500).join(' ');
    return truncated + '\n[⚠ Output truncated — possible hallucination]';
  }

  return result;
}

function detectAndRemoveCycles(lines) {
  const output = [];
  let i = 0;

  while (i < lines.length) {
    let cycleFound = false;

    for (let cycleLen = 2; cycleLen <= 6 && !cycleFound; cycleLen++) {
      if (i + cycleLen * 3 > lines.length) continue; // need 3+ reps

      const cycle = lines.slice(i, i + cycleLen);
      let reps = 1;
      let j = i + cycleLen;

      while (j + cycleLen <= lines.length) {
        let match = true;
        for (let k = 0; k < cycleLen; k++) {
          const a = cycle[k].toLowerCase().replace(/\s+/g, ' ').trim();
          const b = lines[j + k].toLowerCase().replace(/\s+/g, ' ').trim();
          if (a !== b) { match = false; break; }
        }
        if (!match) break;
        reps++;
        j += cycleLen;
      }

      if (reps >= 3) {
        for (const line of cycle) output.push(line);
        output.push('[⚠ Repetitive content removed]');
        i = j;
        cycleFound = true;
      }
    }

    if (!cycleFound) {
      output.push(lines[i]);
      i++;
    }
  }

  return output.join('\n');
}

// Did dedup remove >60% of content? Strong signal that the page is garbage.
function isPageGarbage(originalText, dedupedText) {
  if (!originalText || originalText.startsWith('[Could not extract')) return true;
  if (dedupedText.includes('[⚠')) return true; // explicit warnings already attached

  const origWords  = originalText.split(/\s+/).length;
  const dedupWords = dedupedText.split(/\s+/).length;

  // Removed >60% AND original was substantial (not just a header)
  return origWords > 50 && dedupWords < origWords * 0.4;
}

// ─── Provider: Gemini (vision — OCR + translate in one call) ──────────────
async function callGemini(job) {
  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
  const model = genAI.getGenerativeModel({
    model: "gemini-2.5-flash-lite",
    generationConfig: {
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
  parts.push({ text: buildTranslatePrompt(targetLang, sourceLang, job.pageNums, false) });

  const result = await model.generateContent(parts);
  const text   = result.response.text();

  if (!text || text.trim().length < 5) {
    throw new Error('RECITATION: Gemini returned empty (silent safety filter)');
  }
  return text;
}

// ─── Provider: Llama Scout (vision — OCR + translate) ─────────────────────
async function callLlamaTranslate(job) {
  const targetLang = LANGUAGES[job.targetLang] || "English";
  const sourceLang = job.sourceLang === "auto"
    ? "the source language (auto-detect)"
    : (LANGUAGES[job.sourceLang] || "the source language");

  const content = [];
  for (let i = 0; i < job.images.length; i++) {
    content.push({ type: "text", text: `[Page ${job.pageNums[i]}]` });
    content.push({
      type: "image_url",
      image_url: { url: `data:image/jpeg;base64,${job.images[i]}` }
    });
  }
  content.push({ type: "text", text: buildTranslatePrompt(targetLang, sourceLang, job.pageNums, true) });

  return callGroqAPI("meta-llama/llama-4-scout-17b-16e-instruct", content, job.pageNums.length * 800);
}

// ─── Provider: Llama Scout OCR-only (vision — extract text, no translate) ─
async function callLlamaOCR(job) {
  const sourceLang = job.sourceLang === "auto"
    ? "the source language"
    : (LANGUAGES[job.sourceLang] || "the source language");

  const content = [];
  for (let i = 0; i < job.images.length; i++) {
    content.push({ type: "text", text: `[Page ${job.pageNums[i]}]` });
    content.push({
      type: "image_url",
      image_url: { url: `data:image/jpeg;base64,${job.images[i]}` }
    });
  }
  content.push({ type: "text", text: buildOCRPrompt(sourceLang, job.pageNums) });

  return callGroqAPI("meta-llama/llama-4-scout-17b-16e-instruct", content, job.pageNums.length * 600);
}

// ─── Provider: Qwen3-32B (TEXT-ONLY translator, used in rescue pipeline) ──
async function callQwen3Translate(sourceLang, targetLang, extractedText) {
  const tgt = LANGUAGES[targetLang] || "English";
  const src = sourceLang === "auto"
    ? "the source language (auto-detect)"
    : (LANGUAGES[sourceLang] || "the source language");

  const content = [{ type: "text", text: buildTextTranslatePrompt(tgt, src, extractedText) }];
  return callGroqAPI("qwen/qwen3-32b", content, 1500);
}

async function callGroqAPI(modelId, content, maxTokens) {
  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method:  "POST",
    headers: {
      "Content-Type":  "application/json",
      "Authorization": `Bearer ${GROQ_API_KEY}`
    },
    body: JSON.stringify({
      model:             modelId,
      messages:          [{ role: "user", content }],
      max_tokens:        maxTokens,
      temperature:       0.15,
      frequency_penalty: 0.3
    })
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    const msg = err?.error?.message || `Groq HTTP ${response.status}`;
    throw new Error(`[Groq/${modelId}] ${msg}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || "";
}

// ─── Per-page Gemini rescue (escalation) ──────────────────────────────────
// Re-translates SPECIFIC pages flagged as garbage, using Gemini.
async function rescueWithGemini(job, badPages) {
  console.log(`[translate] Gemini rescue for pages: ${badPages.join(',')}`);
  const rescued = {};

  for (const pn of badPages) {
    if (!geminiAvailable) break;
    const idx = job.pageNums.indexOf(pn);
    if (idx === -1) continue;

    const single = { ...job, images: [job.images[idx]], pageNums: [pn] };
    try {
      await new Promise(r => setTimeout(r, 4000));
      const raw    = await callGemini(single);
      const parsed = extractPages(raw, [pn]);
      const clean  = deduplicateContent(parsed[pn]);

      if (!parsed[pn].startsWith('[Could not extract') && !isPageGarbage(parsed[pn], clean)) {
        rescued[pn] = clean;
        console.log(`[translate] Gemini rescued page ${pn}`);
      }
    } catch (err) {
      const msg = err.message || '';
      if (isGeminiDailyQuota(msg)) {
        geminiAvailable = false;
        console.log(`[translate] Gemini quota exhausted during rescue`);
        break;
      }
      console.log(`[translate] Gemini rescue failed for page ${pn}: ${msg.slice(0, 80)}`);
    }
  }

  return rescued;
}

// ─── OCR→Qwen3 pipeline rescue (last resort) ──────────────────────────────
// Used when Gemini is unavailable AND Llama produced garbage.
async function rescueWithPipeline(job, badPages) {
  console.log(`[translate] OCR→Qwen3 pipeline rescue for pages: ${badPages.join(',')}`);
  const rescued = {};

  for (const pn of badPages) {
    const idx = job.pageNums.indexOf(pn);
    if (idx === -1) continue;

    try {
      const single = { ...job, images: [job.images[idx]], pageNums: [pn] };
      await new Promise(r => setTimeout(r, 5000));
      const ocrRaw    = await callLlamaOCR(single);
      const ocrParsed = extractPages(ocrRaw, [pn]);
      const extracted = ocrParsed[pn];

      if (!extracted || extracted.startsWith('[Could not extract') || extracted.length < 10) {
        continue;
      }

      await new Promise(r => setTimeout(r, 3000));
      const translated = await callQwen3Translate(job.sourceLang, job.targetLang, extracted);

      if (translated && translated.trim().length > 10) {
        rescued[pn] = deduplicateContent(translated.trim());
        console.log(`[translate] Pipeline rescued page ${pn}`);
      }
    } catch (err) {
      console.log(`[translate] Pipeline rescue failed page ${pn}: ${err.message?.slice(0, 80)}`);
    }
  }

  return rescued;
}

// ─── Main job dispatcher ──────────────────────────────────────────────────
async function processJob(job) {
  const lang        = (job.sourceLang || 'auto').toLowerCase();
  const isHardKnown = KNOWN_HARD_SCRIPTS.has(lang);
  const hasGemini   = !!GEMINI_API_KEY && geminiAvailable;
  const hasGroq     = !!GROQ_API_KEY;

  // ── Decide primary provider ─────────────────────────────────────────
  // Explicit hard script → Gemini (no wasted Groq call)
  // Auto-detect or other languages → Llama Scout (saves Gemini quota)
  let primaryFn, primaryName;
  if (isHardKnown && hasGemini) {
    primaryFn   = callGemini;
    primaryName = 'gemini';
  } else if (hasGroq) {
    primaryFn   = callLlamaTranslate;
    primaryName = 'llama';
  } else if (hasGemini) {
    primaryFn   = callGemini;
    primaryName = 'gemini';
  } else {
    throw new Error('NO_PROVIDERS: No API keys configured.');
  }

  let raw;
  try {
    console.log(`[translate] Primary: ${primaryName} for pages ${job.pageNums.join(',')}`);
    raw = await primaryFn(job);
  } catch (err) {
    const msg = err.message || '';

    // Primary failed entirely — try the OTHER vision provider for the whole job
    if (primaryName === 'gemini' && hasGroq) {
      if (isGeminiDailyQuota(msg)) geminiAvailable = false;
      console.log(`[translate] Gemini failed (${msg.slice(0, 60)}) — falling back to Llama for whole job`);
      try {
        raw = await callLlamaTranslate(job);
        primaryName = 'llama';
      } catch (err2) {
        return processFallbackOnly(job, err2);
      }
    } else if (primaryName === 'llama' && hasGemini) {
      if (isGroqQuota(msg)) {
        console.log(`[translate] Groq quota exhausted — falling back to Gemini for whole job`);
      } else {
        console.log(`[translate] Llama failed (${msg.slice(0, 60)}) — falling back to Gemini`);
      }
      try {
        raw = await callGemini(job);
        primaryName = 'gemini';
      } catch (err2) {
        return processFallbackOnly(job, err2);
      }
    } else {
      throw err;
    }
  }

  // ── Parse + dedup + flag bad pages ──────────────────────────────────
  const parsed  = extractPages(raw, job.pageNums);
  const deduped = {};
  const badPages = [];

  for (const pn of job.pageNums) {
    const original = parsed[pn];
    const clean    = original.startsWith('[Could not extract') ? original : deduplicateContent(original);
    deduped[pn]    = clean;
    if (isPageGarbage(original, clean)) badPages.push(pn);
  }

  let usedProvider = primaryName;

  // ── ESCALATION: bad pages get retried by the alternate vision provider ──
  if (badPages.length > 0) {
    console.log(`[translate] ${badPages.length} bad pages from ${primaryName} — escalating`);

    if (primaryName === 'llama' && hasGemini) {
      // Llama loops → Gemini rescue (quality escalation)
      const rescued = await rescueWithGemini(job, badPages);
      for (const [pn, text] of Object.entries(rescued)) {
        deduped[parseInt(pn)] = text;
        const idx = badPages.indexOf(parseInt(pn));
        if (idx !== -1) badPages.splice(idx, 1);
      }
      if (Object.keys(rescued).length > 0) usedProvider = 'llama+gemini';
    } else if (primaryName === 'gemini' && hasGroq) {
      // Gemini failed on these pages → try Llama
      const rescued = await rescueWithLlama(job, badPages);
      for (const [pn, text] of Object.entries(rescued)) {
        deduped[parseInt(pn)] = text;
        const idx = badPages.indexOf(parseInt(pn));
        if (idx !== -1) badPages.splice(idx, 1);
      }
      if (Object.keys(rescued).length > 0) usedProvider = 'gemini+llama';
    }

    // Still bad? Try the OCR→Qwen3 pipeline as last resort
    if (badPages.length > 0 && hasGroq) {
      const rescued = await rescueWithPipeline(job, badPages);
      for (const [pn, text] of Object.entries(rescued)) {
        deduped[parseInt(pn)] = text;
      }
      if (Object.keys(rescued).length > 0) usedProvider = `${usedProvider}+pipeline`;
    }
  }

  return { translations: deduped, provider: usedProvider };
}

// Per-page Llama rescue (used when Gemini was primary and failed on some pages)
async function rescueWithLlama(job, badPages) {
  const rescued = {};
  for (const pn of badPages) {
    const idx = job.pageNums.indexOf(pn);
    if (idx === -1) continue;
    const single = { ...job, images: [job.images[idx]], pageNums: [pn] };
    try {
      await new Promise(r => setTimeout(r, 4000));
      const raw    = await callLlamaTranslate(single);
      const parsed = extractPages(raw, [pn]);
      const clean  = deduplicateContent(parsed[pn]);
      if (!isPageGarbage(parsed[pn], clean)) rescued[pn] = clean;
    } catch (_) { /* skip */ }
  }
  return rescued;
}

function processFallbackOnly(job, err) {
  const msg = err.message || 'All providers failed';
  if (isGeminiDailyQuota(msg) && isGroqQuota(msg)) {
    throw new Error('QUOTA_EXHAUSTED_ALL: Both quotas exhausted. Gemini resets at midnight PT, Groq at midnight UTC.');
  }
  throw err;
}

// ─── Error classifiers ────────────────────────────────────────────────────
function isGeminiDailyQuota(msg) {
  return msg.includes('GenerateRequestsPerDayPerProjectPerModel') ||
         (msg.includes('429') && msg.includes('limit: 20'));
}
function isGroqQuota(msg) {
  return msg.includes('rate_limit_exceeded') ||
         msg.includes('tokens_per_day') ||
         msg.includes('requests_per_day');
}
function isRecitation(msg) {
  return msg.includes('RECITATION') || msg.includes('recitation') ||
         msg.includes('SAFETY') || msg.includes('finish_reason: SAFETY');
}

// ─── Queue runner ─────────────────────────────────────────────────────────
async function runQueue() {
  if (isProcessing || queue.length === 0) return;
  isProcessing = true;

  while (queue.length > 0) {
    const job = queue[0];

    if (Date.now() - job.timestamp > 180000) {
      jobResults[job.id] = { status: "error", error: "Job expired — session timed out" };
      queue.shift();
      continue;
    }

    jobResults[job.id] = { status: "processing", queuePos: 0 };

    try {
      const { translations, provider } = await processJob(job);
      jobResults[job.id] = { status: "done", translations, provider };
    } catch (err) {
      const msg = err.message || '';
      if (msg.includes('QUOTA_EXHAUSTED_ALL') || msg.includes('NO_PROVIDERS')) {
        jobResults[job.id] = { status: "error", error: msg };
        drainQueue(msg);
        isProcessing = false;
        return;
      }
      jobResults[job.id] = { status: "error", error: msg };
    }

    queue.shift();
    queue.forEach((j, idx) => {
      if (jobResults[j.id]) jobResults[j.id].queuePos = idx + 1;
    });

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

function cleanupOldResults() {
  const now = Date.now();
  for (const id in jobResults) {
    if (jobResults[id].timestamp && now - jobResults[id].timestamp > 3600000) {
      delete jobResults[id];
    }
  }
}

// ─── HTTP handler ─────────────────────────────────────────────────────────
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
    return res.json({ queueLength: queue.length, isProcessing });

  if (req.method === "POST") {
    cleanupOldResults();

    if (!GEMINI_API_KEY && !GROQ_API_KEY)
      return res.status(500).json({ error: "No API keys configured. Set GEMINI_API_KEY or GROQ_API_KEY in Vercel environment variables." });

    if (GEMINI_API_KEY) geminiAvailable = true;

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

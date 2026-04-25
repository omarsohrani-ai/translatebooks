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
  return `You are an OCR translation engine. Your task is purely mechanical: read the pixels in each image and output their meaning in ${targetLang}. Nothing else.

════════════════════════════════════════════
RULE 1 — VISUAL ANCHORING (highest priority)
════════════════════════════════════════════
Before translating, silently count the visible lines of text in the image.
Multiply that count by 10 — that is your internal word budget.
This is a private calculation: do NOT write the word count, line count,
or any meta-commentary in your output. Your output contains ONLY translated text.
Example internal budget: 20 visible lines → ~200 words max output.

This rule exists because you may recognise the document's subject matter. That recognition is a trap:
- You must NOT continue, extend, or complete any text from your training knowledge
- You must NOT add content that "fits" the genre, topic, style, or argument of the document
- You must NOT fill in what logically "comes next"

SPECIFICALLY FORBIDDEN for ALL document types — these are stock phrases your
training data contains that are NOT printed on any real page you will ever be given:
  ✗ Book introductions, prefaces, forewords, or dedications
  ✗ Translator's notes ("I have been asked to translate...", "I have endeavoured...")
  ✗ Author's acknowledgements or opening prayers of praise
  ✗ Chapter summaries or editorial commentary
  ✗ Legal boilerplate, terms-and-conditions expansions
  ✗ Medical disclaimers or standard clinical text
  ✗ Any passage that begins with "In the name of God..." unless those exact words appear in the image

This applies to ALL document types without exception:
  legal text, religious text, poetry, fiction, technical manuals,
  medical records, contracts, letters, news articles, academic papers —
  every genre carries the same risk of hallucination.

The moment you run out of visible text on the page → STOP. Do not add a single word more.
If a word is illegible → write [illegible].
If a line is partially cut off → translate only the visible portion and STOP.

LOOP SELF-CHECK: While writing, if you notice yourself repeating the same sentence
structure more than twice (e.g. "If X then Y", "If X then Y", "If X then Y"…),
STOP immediately. You have entered a hallucination loop. Delete the repetitions and end
the block. Real pages do not contain infinite variations of the same sentence.

════════════════════════════════════════════
RULE 2 — TRANSLATION QUALITY
════════════════════════════════════════════
- Translate EVERY visible word fully into ${targetLang} — no source-language words in output
- Do NOT use parentheses to preserve original terms — translate everything
- If the source contains a word from a third language, translate that too into ${targetLang}
- Preserve paragraph breaks, section headings, and typographic markers exactly as they appear
- If a page is blank or contains no meaningful text, write exactly: [Blank page]
- Do NOT add translator notes, footnotes, commentary, summaries, or explanations

════════════════════════════════════════════
RULE 3 — OUTPUT FORMAT
════════════════════════════════════════════
- Do NOT wrap output in markdown, code blocks, or any formatting tags
- Write ONLY the ===PAGE N=== blocks shown below — nothing before, nothing after

You are translating ${pageNums.length} page(s): ${pageNums.join(', ')}.

${pageNums.map(n =>
  `===PAGE ${n}===\n[translate only the exact text visible in page ${n} image — stop when the image text ends]\n===END===`
).join('\n\n')}

Final self-check before submitting:
  • Does your output contain exactly ${pageNums.length} block(s)?
  • Is the output length proportional to the text visible in the image?
  • Did you add ANYTHING not physically printed on the page? If yes → delete it.
  • Does it contain a preface, introduction, translator's note, or word count? If yes → delete it.`;
}

// ── Robust split-based page extractor ─────────────────────────────────────
function extractPages(raw, pageNums) {
  let text = raw
    .replace(/^```[\w]*\n?/m, '').replace(/\n?```\s*$/m, '')
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, '')
    // Strip meta-commentary the AI sometimes leaks into output
    .replace(/\[word count[^\]]*\]/gi, '')
    .replace(/\[line count[^\]]*\]/gi, '')
    .replace(/\[translation (note|end|complete)[^\]]*\]/gi, '')
    .replace(/\(word count[^)]*\)/gi, '');

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

  // ── Fallback: no PAGE markers found ───────────────────────────────────────
  // Happens when: (a) AI ignores format instructions, (b) Gemini silent
  // RECITATION returns a refusal sentence instead of throwing an exception,
  // (c) response is otherwise well-formed but lacks the delimiters.
  // Recovery: strip any stray markers and treat the full response as content.
  if (Object.keys(blocks).length === 0) {
    const fallback = text
      .replace(/===END===/gi, '')
      .replace(/===PAGE\s*\d+===/gi, '')
      .trim();
    if (fallback && fallback.length > 10) {
      const firstPage = pageNums[0];
      blocks[firstPage] = deduplicateContent(fallback);
      console.log(`[translate] No PAGE markers — fallback: full response assigned to page ${firstPage}`);
    }
  }

  const parsed = {};
  for (const pn of pageNums) {
    parsed[pn] = blocks[pn] || `[Could not extract page ${pn}]`;
  }
  return parsed;
}

// ── Within-page deduplication ─────────────────────────────────────────────
// Catches two classes of AI hallucination:
//   1. Consecutive repeated lines (same line printed back-to-back)
//   2. Non-consecutive repeated lines — AI loops back and re-prints content
//      that already appeared earlier on the same page
//      (this is the most common cause of duplicate aphorisms / paragraphs)
function deduplicateContent(text) {
  const lines = text.split('\n');
  const out   = [];
  let lastLine = '', repeatCount = 0;

  // Tracks substantial lines (30+ chars) already seen anywhere on this page.
  // Exact-match only — safe for texts with structurally similar but distinct sentences.
  const seenSubstantial = new Set();

  for (const line of lines) {
    const trimmed = line.trim();

    // Preserve blank lines (paragraph breaks) as-is
    if (!trimmed) { out.push(line); lastLine = ''; repeatCount = 0; continue; }

    // For substantial lines: drop if already seen anywhere earlier on this page
    if (trimmed.length >= 30) {
      if (seenSubstantial.has(trimmed)) continue; // hallucinated repeat — drop
      seenSubstantial.add(trimmed);
    }

    // For short lines: still block consecutive repeats (headers, labels, etc.)
    if (trimmed === lastLine) {
      repeatCount++;
      if (repeatCount < 2) out.push(line);
    } else {
      out.push(line);
      lastLine    = trimmed;
      repeatCount = 0;
    }
  }

  const joined = out.join('\n');
  return detectAndTruncateLoop(joined);
}

// ── Semantic loop detector ─────────────────────────────────────────────────
// Catches "template repetition" hallucinations: the AI keeps the same sentence
// structure but swaps one word each iteration (e.g. "If you are X then you will
// see Y" cycling through synonyms). These don't match as exact duplicates but
// have very high word-set overlap between consecutive windows of text.
function detectAndTruncateLoop(text) {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length < 80) return text; // too short to contain a meaningful loop

  const WINDOW = 30; // sliding window size in words
  const STEP   = Math.max(1, Math.floor(WINDOW / 2));

  for (let i = WINDOW * 2; i <= words.length - WINDOW; i += STEP) {
    // Build a set of meaningful words (3+ chars) from the previous window
    const prevSet = new Set(
      words.slice(i - WINDOW, i)
           .map(w => w.toLowerCase().replace(/[^a-z]/g, ''))
           .filter(w => w.length >= 3)
    );
    if (prevSet.size === 0) continue;

    // Count how many current-window words are already in the previous window
    const currWords = words.slice(i, i + WINDOW)
                           .map(w => w.toLowerCase().replace(/[^a-z]/g, ''))
                           .filter(w => w.length >= 3);
    if (currWords.length === 0) continue;

    const overlap = currWords.filter(w => prevSet.has(w)).length / currWords.length;

    if (overlap >= 0.62) {
      // Step back to find the exact sentence boundary before the loop starts
      const cutPoint = Math.max(0, i - STEP);
      let truncated = words.slice(0, cutPoint).join(' ');

      // Trim to the last complete sentence so output doesn't end mid-phrase
      const lastStop = Math.max(
        truncated.lastIndexOf('. '),
        truncated.lastIndexOf('! '),
        truncated.lastIndexOf('? '),
        truncated.lastIndexOf('.\n'),
      );
      if (lastStop > truncated.length * 0.4) {
        truncated = truncated.slice(0, lastStop + 1).trim();
      }

      console.log(
        `[translate] Semantic loop detected at word ${i} ` +
        `(${Math.round(overlap * 100)}% overlap) — truncating to ${cutPoint} words`
      );
      return truncated;
    }
  }
  return text;
}

// ── Cross-page / cross-batch deduplication ────────────────────────────────
// When the AI is given pages N and N+1 in separate batches, it sometimes
// re-translates the last 1–6 lines of page N at the very start of page N+1.
// This pass compares the tail of each page with the head of the next and
// strips any matching prefix from the later page.
function deduplicateAcrossPages(parsed, pageNums) {
  const sorted = [...pageNums].sort((a, b) => a - b);

  for (let i = 0; i < sorted.length - 1; i++) {
    const currPn = sorted[i];
    const nextPn = sorted[i + 1];
    if (!parsed[currPn] || !parsed[nextPn]) continue;
    // Skip error/placeholder pages
    if (parsed[currPn].startsWith('[') || parsed[nextPn].startsWith('[')) continue;

    const currLines    = parsed[currPn].split('\n').map(l => l.trim()).filter(Boolean);
    const nextLines    = parsed[nextPn].split('\n');          // keep originals for output
    const nextTrimmed  = nextLines.map(l => l.trim()).filter(Boolean);

    // Try overlaps of 1–6 lines (only match lines that are 20+ chars — ignore short headings)
    const maxCheck = Math.min(6, currLines.length, nextTrimmed.length);
    let overlapCount = 0;

    for (let t = maxCheck; t >= 1; t--) {
      const tail = currLines.slice(-t);
      const head = nextTrimmed.slice(0, t);
      const allMatch = tail.every((line, idx) => line === head[idx] && line.length >= 20);
      if (allMatch) { overlapCount = t; break; }
    }

    if (overlapCount > 0) {
      // Remove the overlapping lines from the start of the next page (preserve spacing)
      const tail = currLines.slice(-overlapCount);
      let removed = 0;
      const newNext = [];
      for (const line of nextLines) {
        if (removed < overlapCount && line.trim() === tail[removed]) {
          removed++;
          continue; // drop this duplicate line
        }
        newNext.push(line);
      }
      parsed[nextPn] = newNext.join('\n').replace(/^\n+/, '').trim();
      console.log(`[translate] Cross-page dedup: removed ${overlapCount} line(s) from start of page ${nextPn}`);
    }
  }

  return parsed;
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
      // 500 tokens/page: enough for dense real content (~350 translated words),
      // tight enough to hard-cut hallucination loops before they spiral.
      maxOutputTokens: job.pageNums.length * 500,
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
  const raw = result.response.text();

  // Gemini sometimes returns an empty string instead of throwing when its
  // safety filter (RECITATION) silently blocks classical/religious content.
  // Treat empty as RECITATION so the caller can fall back to Groq.
  if (!raw || raw.trim().length < 5) {
    throw new Error('RECITATION: Gemini returned empty response (silent safety filter)');
  }
  return raw;
}

// ── Provider: Groq (Llama 4 Maverick) ────────────────────────────────────
// Llama 4 Maverick: 17B params, 128 experts — significantly better instruction
// following than Scout (16 experts). Natively multimodal, supports Arabic.
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
      model:      "meta-llama/llama-4-maverick-17b-128e-instruct",
      messages:   [{ role: "user", content }],
      max_tokens: job.pageNums.length * 500,
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

  // Final pass: remove any cross-batch overlap between consecutive pages
  return deduplicateAcrossPages(parsed, job.pageNums);
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

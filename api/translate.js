// api/translate.js — Vercel Serverless Function
// PDF Translation — Quality-driven multi-agent routing with hallucination detection
//
// Key features:
//   1. Routing based on declared source language (saves Gemini for hard scripts)
//   2. Per-page quality detection (cycle/repetition/length anomalies)
//   3. Per-page Gemini rescue when Llama produces garbage
//   4. Hallucination signature detector — catches Llama fabricating on classical
//      texts (Islamic, Buddhist, Hindu, etc.) even when output isn't repetitive
//   5. Session-level lock: once a hard-source signature is detected, all
//      subsequent batches in this session route to Gemini regardless of routing
//   6. OCR→Qwen3 pipeline as last resort when Gemini quota exhausts
//
// Provider field returned to client:
//   'gemini'              — Gemini handled cleanly
//   'llama'               — Llama handled cleanly
//   'llama+gemini'        — Llama tried, some pages escalated to Gemini
//   'gemini+llama'        — Gemini tried, some pages fell back to Llama
//   'llama+gemini+pipeline' — Both vision providers tried, final pipeline rescue
//   'pipeline'            — Pipeline-only (Gemini exhausted, Llama looping)
//
// Setup:
//   Required: GEMINI_API_KEY  — aistudio.google.com (20 RPD free)
//   Optional: GROQ_API_KEY    — console.groq.com (1000 RPD free, no card)

const { GoogleGenerativeAI } = require("@google/generative-ai");

const queue       = [];
const jobResults  = {};
let isProcessing  = false;

// Session state
let geminiAvailable    = true;   // flips false when Gemini quota exhausts
let hardSourceDetected = false;  // flips true when classical/religious text detected

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

// Languages explicitly requiring Gemini for OCR quality
const KNOWN_HARD_SCRIPTS = new Set(['ar', 'fa', 'ur', 'he', 'am', 'th']);

// ─── Hallucination signature patterns ─────────────────────────────────────
// These appear in Llama's output when it's fabricating content from training
// data instead of reading the actual page. Detection across multiple religious
// and classical traditions, not Arabic-specific.
const HALLUCINATION_SIGNATURES = [
  // Islamic / classical Arabic markers
  /peace be upon (him|her|them)/gi,
  /\b(may|the) (god|allah) (be pleased|forgive|bless|the almighty)/gi,
  /it (was|is) narrated (by|that|from)/gi,
  /\bthe (prophet|messenger) (of god|of allah|peace)/gi,
  /\b(hadith|quran|qur'an|allah|sunnah|sahaba)\b/gi,
  /\b(sahih|tafsir|fiqh|sharia|ulema)\b/gi,
  /\bibn\s+[A-Z][a-z]+/g,  // "ibn Arabi", "ibn Sina" etc.

  // Sufi / Ibn Arabi specific patterns (the chapter naming style)
  /chapter on .{1,40}translation/gi,
  /book of (translations|biographies|exhortations|indications|sayings)/gi,
  /\b(a hint|an indication|a fine point|a subtlety|a nice saying)\s*[-:—]/gi,

  // Buddhist markers
  /\b(buddha said|the dharma|sutra|bodhisattva|sangha|nirvana)\b/gi,
  /\b(zen master|mahayana|theravada)\b/gi,

  // Hindu / Sanskrit markers
  /\b(bhagavan|krishna said|veda|upanishad|purana|shloka|verse said)\b/gi,
  /\b(arjuna|guru|swami|brahman|atman)\b/gi,

  // Jewish / Hebrew markers
  /\b(torah|talmud|midrash|rabbi|halakha|kabbalah)\b/gi,
  /\bblessed be he\b/gi,

  // Christian medieval / patristic markers
  /\b(saint augustine|aquinas|patristic|epistle of)\b/gi,
];

function detectHallucinationProneSource(text) {
  if (!text || text.length < 50) return false;

  let signalCount = 0;
  for (const pattern of HALLUCINATION_SIGNATURES) {
    const matches = text.match(pattern);
    if (matches) signalCount += matches.length;
    if (signalCount >= 3) return true; // early exit
  }
  return signalCount >= 3;
}

// ─── Prompt builders ──────────────────────────────────────────────────────
function buildTranslatePrompt(targetLang, sourceLang, pageNums, isGroqModel = false) {
  const groqExtra = isGroqModel ? `
ANTI-HALLUCINATION RULES — CRITICAL for this model:
- Output ONLY text you can physically READ in the image. Nothing else.
- If you cannot read a word clearly, write [unclear] — do NOT guess.
- If a page has only a few lines, your output must be only a few lines.
- NEVER generate filler text, religious commentary, or scholarly prose from memory.
- NEVER write phrases like "It was narrated", "Peace be upon him", "The Prophet said"
  unless those exact words are clearly visible in the image.
- NEVER repeat any sentence more than once. If you catch yourself repeating, STOP.
- If you have written more than 30 sentences for one page, STOP.
- Do NOT translate Arabic words by their phonetic similarity to other names —
  read each word carefully and translate its actual meaning.
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
- Preserve the ORIGINAL script — do NOT translate
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
    .replace(/<think>[\s\S]*?<\/think>/gi, '')       // closed think block
    .replace(/<think>[\s\S]*/gi, '')                  // unclosed think block (token cutoff mid-think)
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, '')  // closed thinking block
    .replace(/<thinking>[\s\S]*/gi, '');              // unclosed thinking block

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

  if (Object.keys(blocks).length === 0 && text.trim().length > 20 && pageNums.length === 1) {
    blocks[pageNums[0]] = text.trim();
  }

  const parsed = {};
  for (const pn of pageNums) {
    parsed[pn] = blocks[pn] || `[Could not extract page ${pn}]`;
  }
  return parsed;
}

// ─── Quality detection: cycles + frequency + length ──────────────────────
function deduplicateContent(text) {
  // Pass 0: within-line repetition (semicolon/comma separated fragments)
  // Catches: "do not be against him; do not be against him; do not be against him"
  // Strategy: split each line by [;.] and apply the same frequency cap within the line.
  text = text.split('\n').map(line => {
    const trimmed = line.trim();
    if (trimmed.length < 40) return line; // short lines — skip

    // Split on ', ' or '; ' or '. ' to catch comma-chained repetition (e.g. isnad chains)
    const parts = trimmed.split(/(?<=\w)(?:,\s+|;\s+|(?<!\w\.\w)\.\s+)/);
    if (parts.length < 3) return line;

    const seen   = new Map();
    const kept   = [];
    let removed  = 0;
    for (const part of parts) {
      const key = part.toLowerCase().replace(/\s+/g, ' ').trim();
      if (key.length < 12) { kept.push(part); continue; }
      const cnt = (seen.get(key) || 0) + 1;
      seen.set(key, cnt);
      if (cnt <= 2) kept.push(part);
      else removed++;
    }
    if (removed === 0) return line;
    return kept.join('; ') + (removed > 0 ? ' [⚠ within-line repetition removed]' : '');
  }).join('\n');

  const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length < 4) return text;

  // Pass 1: frequency cap (max 2 occurrences of any sentence)
  const freq  = new Map();
  const pass1 = [];

  for (const line of lines) {
    const key = line.toLowerCase().replace(/\s+/g, ' ').replace(/[.,:;!?]+$/, '').trim();
    if (key.length < 25) { pass1.push(line); continue; }
    const count = (freq.get(key) || 0) + 1;
    freq.set(key, count);
    if (count <= 2) pass1.push(line);
  }

  // Pass 2: cycle detection
  const result = detectAndRemoveCycles(pass1);

  // Pass 3: hard word cap
  const words = result.split(/\s+/).length;
  if (words > 600) {
    return result.split(/\s+/).slice(0, 500).join(' ') + '\n[⚠ Output truncated — possible hallucination]';
  }

  return result;
}

function detectAndRemoveCycles(lines) {
  const output = [];
  let i = 0;

  while (i < lines.length) {
    let cycleFound = false;

    for (let cycleLen = 2; cycleLen <= 6 && !cycleFound; cycleLen++) {
      if (i + cycleLen * 3 > lines.length) continue;

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

// Returns true when a non-RTL output page is still mostly in Arabic/Farsi/Urdu script —
// meaning Gemini ran out of tokens mid-batch and returned source text untranslated.
function containsUntranslatedScript(text, targetLang) {
  if (['ar','fa','ur','he','am'].includes(targetLang)) return false; // RTL target is expected
  const arabicChars = (text.match(/[\u0600-\u06FF]/g) || []).length;
  const totalChars  = text.replace(/\s/g, '').length;
  return totalChars > 30 && arabicChars / totalChars > 0.3; // >30% Arabic in non-Arabic output
}

function isPageGarbage(originalText, dedupedText, targetLang = 'en') {
  if (!originalText || originalText.startsWith('[Could not extract')) return true;
  if (dedupedText.includes('[⚠')) return true;
  if (containsUntranslatedScript(dedupedText, targetLang)) return true;
  const origWords  = originalText.split(/\s+/).length;
  const dedupWords = dedupedText.split(/\s+/).length;
  return origWords > 50 && dedupWords < origWords * 0.4;
}

// ─── Provider: Gemini ─────────────────────────────────────────────────────
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

// ─── Provider: Llama Scout (vision OCR + translate) ───────────────────────
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

// ─── Provider: Llama Scout OCR-only ───────────────────────────────────────
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

// ─── Provider: Qwen3-32B text-only ───────────────────────────────────────
async function callQwen3Translate(sourceLang, targetLang, extractedText) {
  const tgt = LANGUAGES[targetLang] || "English";
  const src = sourceLang === "auto"
    ? "the source language (auto-detect)"
    : (LANGUAGES[sourceLang] || "the source language");

  // Estimate output tokens: source word count × ~1.5 expansion, minimum 1500
  const estTokens = Math.max(1500, Math.ceil(extractedText.split(/\s+/).length * 1.5));
  const content = [{ type: "text", text: buildTextTranslatePrompt(tgt, src, extractedText) }];

  // /no_think disables Qwen3's chain-of-thought reasoning — without it the model
  // emits a large <think> block that eats into the token budget and may be
  // truncated mid-block, leaving the raw thinking text in the output.
  return callGroqAPI("qwen/qwen3-32b", content, estTokens, "/no_think");
}

async function callGroqAPI(modelId, content, maxTokens, systemPrompt = null) {
  const messages = systemPrompt
    ? [{ role: "system", content: systemPrompt }, { role: "user", content }]
    : [{ role: "user", content }];

  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method:  "POST",
    headers: {
      "Content-Type":  "application/json",
      "Authorization": `Bearer ${GROQ_API_KEY}`
    },
    body: JSON.stringify({
      model:             modelId,
      messages,
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

// ─── Per-page Gemini rescue ───────────────────────────────────────────────
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
      if (!parsed[pn].startsWith('[Could not extract') && !isPageGarbage(parsed[pn], clean, job.targetLang)) {
        rescued[pn] = clean;
      }
    } catch (err) {
      const msg = err.message || '';
      if (isGeminiDailyQuota(msg)) {
        geminiAvailable = false;
        console.log(`[translate] Gemini quota exhausted during rescue`);
        break;
      }
    }
  }

  return rescued;
}

// ─── Per-page Llama rescue ────────────────────────────────────────────────
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
      if (!isPageGarbage(parsed[pn], clean, job.targetLang)) rescued[pn] = clean;
    } catch (_) { /* skip */ }
  }
  return rescued;
}

// ─── OCR→Qwen3 pipeline rescue (last resort) ──────────────────────────────
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
      if (!extracted || extracted.startsWith('[Could not extract') || extracted.length < 10) continue;

      await new Promise(r => setTimeout(r, 3000));
      const translated = await callQwen3Translate(job.sourceLang, job.targetLang, extracted);
      if (translated && translated.trim().length > 10) {
        rescued[pn] = deduplicateContent(translated.trim());
      }
    } catch (_) { /* skip */ }
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
  // Priority order:
  //   1. Session lock: hard source detected mid-session → force Gemini
  //   2. Explicit hard script → Gemini
  //   3. Anything else → Llama (saves Gemini quota for the cases that need it)
  let primaryFn, primaryName;

  if ((hardSourceDetected || isHardKnown) && hasGemini) {
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
    console.log(`[translate] Primary: ${primaryName} for pages ${job.pageNums.join(',')}${hardSourceDetected ? ' [hard-source lock active]' : ''}`);
    raw = await primaryFn(job);
  } catch (err) {
    const msg = err.message || '';

    // Fallback to alternate provider for the entire batch
    if (primaryName === 'gemini' && hasGroq) {
      if (isGeminiDailyQuota(msg)) geminiAvailable = false;
      console.log(`[translate] Gemini failed — falling back to Llama for whole job`);
      try {
        raw = await callLlamaTranslate(job);
        primaryName = 'llama';
      } catch (err2) {
        return processFallbackOnly(job, err2);
      }
    } else if (primaryName === 'llama' && hasGemini) {
      console.log(`[translate] Llama failed — falling back to Gemini for whole job`);
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

  // ── Hallucination signature detection (only when Llama is primary) ────────
  // Fires when Llama's output contains classical/religious source markers.
  // These markers mean Llama is drawing on training data rather than reading
  // the page — output is fabricated even if it isn't repetitive.
  let hallucinationRisk = false;

  if (primaryName === 'llama' && !isHardKnown) {
    if (detectHallucinationProneSource(raw)) {
      hardSourceDetected = true;
      hallucinationRisk  = true;

      if (hasGemini) {
        // Best path: escalate entire batch to Gemini and lock session
        console.log(`[translate] Hallucination signature detected — escalating to Gemini, locking session`);
        try {
          raw = await callGemini(job);
          primaryName = 'llama→gemini';
          hallucinationRisk = false; // Gemini output is trustworthy
        } catch (err) {
          const msg = err.message || '';
          if (isGeminiDailyQuota(msg)) geminiAvailable = false;
          console.log(`[translate] Gemini escalation failed: ${msg.slice(0, 80)}`);
          primaryName = 'llama_unverified';
        }
      } else {
        // Gemini unavailable — cannot rescue. Mark output as unverified.
        // The pipeline rescue (Llama OCR → Qwen3) will run below for bad pages,
        // but even clean-looking Llama pages may be fabricated — user must be warned.
        console.log(`[translate] WARN: hallucination signature detected but Gemini unavailable — output unverified`);
        primaryName = 'llama_unverified';
      }
    }
  }

  // ── Parse + dedup + flag bad pages ──────────────────────────────────
  const parsed   = extractPages(raw, job.pageNums);
  const deduped  = {};
  const badPages = [];

  for (const pn of job.pageNums) {
    const original = parsed[pn];
    const clean    = original.startsWith('[Could not extract') ? original : deduplicateContent(original);
    deduped[pn]    = clean;
    if (isPageGarbage(original, clean, job.targetLang)) badPages.push(pn);
  }

  let usedProvider = primaryName;

  // ── Per-page escalation for bad pages ──────────────────────────────
  // "bad pages" = dedup detected garbage (cycles/truncation)
  // "unverified pages" = hallucination risk flagged but Gemini unavailable —
  //   even pages that look clean structurally may be fabricated content,
  //   so we attempt pipeline rescue on ALL pages in that case.
  const pagesNeedingRescue = (primaryName === 'llama_unverified')
    ? [...job.pageNums]   // all pages unverified → attempt pipeline on everything
    : [...badPages];      // only structurally garbage pages

  if (badPages.length > 0 || primaryName === 'llama_unverified') {
    if (badPages.length > 0)
      console.log(`[translate] ${badPages.length} bad pages from ${primaryName} — escalating`);
    if (primaryName === 'llama_unverified')
      console.log(`[translate] All pages unverified (Gemini unavailable) — attempting pipeline rescue`);

    if (primaryName.startsWith('llama') && primaryName !== 'llama_unverified' && hasGemini) {
      // Normal bad-page rescue via Gemini
      const rescued = await rescueWithGemini(job, badPages);
      for (const [pn, text] of Object.entries(rescued)) {
        deduped[parseInt(pn)] = text;
        const idx = badPages.indexOf(parseInt(pn));
        if (idx !== -1) badPages.splice(idx, 1);
      }
      if (Object.keys(rescued).length > 0) {
        usedProvider = primaryName.includes('gemini') ? primaryName : `${primaryName}+gemini`;
      }
    } else if (primaryName === 'gemini' && hasGroq) {
      const rescued = await rescueWithLlama(job, badPages);
      for (const [pn, text] of Object.entries(rescued)) {
        deduped[parseInt(pn)] = text;
        const idx = badPages.indexOf(parseInt(pn));
        if (idx !== -1) badPages.splice(idx, 1);
      }
      if (Object.keys(rescued).length > 0) usedProvider = 'gemini+llama';
    }

    // Pipeline rescue: runs for (a) pages still bad after Gemini rescue,
    // OR (b) ALL pages when output is unverified due to hallucination risk
    const pipelineTargets = primaryName === 'llama_unverified'
      ? pagesNeedingRescue   // attempt all
      : badPages;            // only remaining bad pages

    if (pipelineTargets.length > 0 && hasGroq) {
      const rescued = await rescueWithPipeline(job, pipelineTargets);
      for (const [pn, text] of Object.entries(rescued)) {
        deduped[parseInt(pn)] = text;
      }
      if (Object.keys(rescued).length > 0) {
        usedProvider = primaryName === 'llama_unverified'
          ? 'pipeline'
          : `${usedProvider}+pipeline`;
      }
    }
  }

  return { translations: deduped, provider: usedProvider, hardSourceDetected, hallucinationRisk };
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
      const result = await processJob(job);
      jobResults[job.id] = {
        status: "done",
        translations: result.translations,
        provider:          result.provider,
        hardSourceDetected: result.hardSourceDetected,
        hallucinationRisk:  result.hallucinationRisk
      };
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
    return res.json({ queueLength: queue.length, isProcessing, hardSourceDetected });

  if (req.method === "POST") {
    cleanupOldResults();

    if (!GEMINI_API_KEY && !GROQ_API_KEY)
      return res.status(500).json({ error: "No API keys configured. Set GEMINI_API_KEY or GROQ_API_KEY in Vercel environment variables." });

    // Reset session state when this is a fresh submission to a quiet queue
    // (best-effort: if no jobs are processing, treat as new session)
    if (!isProcessing && queue.length === 0) {
      if (GEMINI_API_KEY) geminiAvailable = true;
      hardSourceDetected = false;
    }

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

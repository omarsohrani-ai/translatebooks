# DocuTranslate — Free AI PDF Translator

Translate any PDF document into 40+ languages using Google Gemini AI.
Completely free to host and run.

---

## Features
- 40+ languages with auto-detection
- Queue system for multiple simultaneous users
- Handles large PDFs (400+ pages)
- Pause / Resume / Stop controls
- Downloads translated PDF automatically
- 100% free (Gemini free tier + Vercel free tier)

---

## Deployment (5 minutes)

### Step 1 — Get a free Gemini API Key
1. Go to https://aistudio.google.com
2. Sign in with your Gmail account
3. Click **"Get API Key"** → **"Create API key"**
4. Copy the key (starts with `AIza...`)

### Step 2 — Upload to GitHub
1. Go to https://github.com and create a free account
2. Click **"New repository"** → name it `docutranslate` → Create
3. Upload all these files:
   ```
   api/translate.js
   package.json
   vercel.json
   public/index.html
   .gitignore
   README.md
   ```

### Step 3 — Deploy to Vercel (free)
1. Go to https://vercel.com
2. Sign in with GitHub
3. Click **"New Project"** → Import your `docutranslate` repo
4. Click **Deploy** — wait ~1 minute

### Step 4 — Add your Gemini API Key
1. In Vercel → Your Project → **Settings** → **Environment Variables**
2. Add:
   - **Name:** `GEMINI_API_KEY`
   - **Value:** your key from Step 1
3. Click **Save** → Go to **Deployments** → **Redeploy**

### Step 5 — Use it!
1. Open your Vercel URL (e.g. `https://docutranslate.vercel.app`)
2. Enter your Vercel URL in the **Server Configuration** box
3. Upload a PDF, choose languages, click **Start Translation**

---

## Free Tier Limits

| Service | Free Limit |
|---------|-----------|
| Gemini 1.5 Flash | 15 requests/min, 1M tokens/day |
| Vercel Functions | 100GB-hours/month |
| Vercel Bandwidth | 100GB/month |

This is more than enough for personal use or a small website.

---

## Monetization Ideas
- Add a Stripe payment page for large documents
- Offer a credits system (e.g. $5 = 500 pages)
- Use Google AdSense on the page
- Offer a "priority queue" paid tier

---

## Tech Stack
- **Frontend:** Plain HTML/CSS/JS (no framework needed)
- **Backend:** Vercel Serverless Function (Node.js)
- **AI:** Google Gemini 1.5 Flash
- **PDF render:** PDF.js
- **PDF output:** jsPDF
- **Hosting:** Vercel (free) + GitHub Pages (free)

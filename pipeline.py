# ===== v4 - GitHub Actions version (run from your site, or Actions tab -> Make video -> Run workflow) =====
# ffmpeg, fonts and pip packages are installed by .github/workflows/make-video.yml

import os, re, json, time, math, asyncio, base64, subprocess, textwrap, urllib.parse, getpass, wave
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ---------- INPUTS ----------
# Keys come from GitHub repo Secrets (Settings -> Secrets and variables -> Actions). Topic/language/voice come from the Run workflow form.
def secret(name, ask, required=True):
    v = os.environ.get(name, "").strip()
    if v: return v
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v: return v.strip()
    except Exception:
        pass
    if os.environ.get("GITHUB_ACTIONS"):
        if not required: return ""
        raise RuntimeError(f"Missing secret {name}: add it in repo Settings -> Secrets and variables -> Actions")
    return getpass.getpass(ask).strip()

NVIDIA_KEY = secret("NVIDIA_KEY", "Paste your NVIDIA API key (nvapi-...), or press Enter to use only Cloudflare: ", required=False)
NVIDIA_KEY_2 = secret("NVIDIA_KEY_2", "Paste your 2nd NVIDIA API key (backup), or press Enter to skip: ", required=False)
NVIDIA_KEYS = [k for k in (NVIDIA_KEY, NVIDIA_KEY_2) if k]   # if key 1 is refused or rate-limited, key 2 is tried
CF_ID = "31fdd1f22c348a91b6da9d4a4e9c3b64"   # Cloudflare Account ID (not a secret)
CF_TOKEN = secret("CF_TOKEN", "Paste your Cloudflare API token: ")
def ask(env, question, default=""):
    v = os.environ.get(env, "").strip()
    if v or os.environ.get("GITHUB_ACTIONS"):
        return v or default
    return input(question)

TOPIC = ask("TOPIC", "Video topic (leave empty = auto topic): ").strip()
AUTO_NICHE = os.environ.get("AUTO_NICHE", "").strip() or (
    "learning videos for kids aged 2-8: alphabet ABC in small parts, numbers, colors, shapes, animals and their sounds, "
    "fruits, vegetables, good habits, and short moral stories with cute animal characters"
    + (", Hindi varnamala (swar and vyanjan)" if os.environ.get("VIDEO_LANG", "").lower().startswith("h") else ""))
CHANNEL_NAME = os.environ.get("CHANNEL_NAME", "").strip() or "Kidzivora"   # shown as a © watermark on every video
_lang = ask("VIDEO_LANG", "Language - en or hi (hindi) [en]: ", "hi")
LANG = "hi" if _lang.strip().lower().startswith("h") else "en"
_voice = ask("VOICE_GENDER", "Voice - m (male) or f (female) [f]: ", "f")
VOICE_GENDER = "m" if _voice.strip().lower().startswith("m") else "f"
_style = ask("IMAGE_STYLE", "Picture style - real (realistic humans) or cartoon [real]: ", "real")
IMAGE_STYLE = "cartoon" if _style.strip().lower().startswith("c") else "real"
_fmt = ask("VIDEO_FORMAT", "Video type - short (9:16, about 60 sec) or long (16:9 story video) [short]: ", "short")
LONG = _fmt.strip().lower().startswith("l")
STORY_GENRE, LONG_MINUTES = "moral", 5
if LONG:
    _g = ask("STORY_GENRE", "Story type - moral (kids learn a lesson), horror (kid-safe spooky), real (true story) or auto [moral]: ", "auto").strip().lower()
    if _g.startswith("a"):   # auto (daily long video): moral -> horror -> real, changing every day
        _g = ["moral", "horror", "real"][time.gmtime().tm_yday % 3]
    STORY_GENRE = "horror" if _g.startswith("h") else ("real" if _g.startswith("r") else "moral")
    if STORY_GENRE == "real" and IMAGE_STYLE == "real":
        IMAGE_STYLE = "cartoon"   # real people are shown as illustrations, never as fake photos
    try:
        LONG_MINUTES = max(2, min(10, int(ask("LONG_MINUTES", "Length in minutes (2-10) [5]: ", "5").strip() or "5")))
    except ValueError:
        LONG_MINUTES = 5

# ---------- CONSTANTS ----------
SCENES = LONG_MINUTES * 9 if LONG else 10   # ~15 words per scene: 10 scenes = about 60 seconds, 9 scenes per minute for long stories
W, H, FPS = (1920, 1080, 25) if LONG else (1080, 1920, 25)
IMG_W, IMG_H = (1344, 768) if LONG else (768, 1344)
TRANS = 0.35          # seconds of cross-transition between scenes (also the voice lead-in)
SR = 44100
OUT = Path("out"); OUT.mkdir(exist_ok=True)
VOICES = {("en","f"): "en-IN-NeerjaNeural", ("en","m"): "en-IN-PrabhatNeural",
          ("hi","f"): "hi-IN-SwaraNeural",   ("hi","m"): "hi-IN-MadhurNeural"}
VOICE_RATE = "+6%"    # "+0%" slower, "+12%" faster
STYLES = {
    "real": ("photorealistic cinematic photograph, shot on 35mm lens, natural soft lighting, shallow depth of field, "
             "realistic skin texture and faces, true-to-life colors, high detail, sharp focus, vertical composition, "
             "no text, no letters, no logos, no watermark, not cartoon, not illustration, not 3d render"),
    "cartoon": "flat vector cartoon illustration, bold colors, simple shapes, clean background, main subject centered, vertical composition",
}
STYLE = STYLES[IMAGE_STYLE] if not LONG else STYLES[IMAGE_STYLE].replace("vertical composition", "wide horizontal composition")
STYLE_SHORT = {"real": "photorealistic cinematic photograph, natural lighting, 35mm lens",
               "cartoon": "cute colorful cartoon illustration, flat vector style"}[IMAGE_STYLE]
FONTS = ["/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
         "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Bold.ttf",
         str(Path("out") / "fonts" / "NotoSansDevanagari-Bold.ttf"),
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
SKIP = set()          # image services that failed with a real problem are skipped for the rest of the video
FAILS = {}            # temporary failures (timeout, busy) per service; a service is skipped only after 3 in a row
DEAD = set()          # text models that NVIDIA has retired (404/410) - not tried again in this run
SEED = int(time.time()) % 1000000   # one seed for every picture of this video (Pollinations only; Cloudflare and NVIDIA do not accept a seed)
CF_TEXT_MODELS = ["@cf/meta/llama-3.3-70b-instruct-fp8-fast", "@cf/meta/llama-3.1-8b-instruct"]   # free backup for the script

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-1800:])
    return p.stdout

# ---------- 1. SCRIPT (NVIDIA free API) ----------
def write_script():
    base = "https://integrate.api.nvidia.com/v1"
    hdr = {"Authorization": f"Bearer {NVIDIA_KEY}"}
    language = "Hindi (Devanagari script), simple everyday spoken Hindi - not formal or shuddh Hindi" if LANG == "hi" else "simple English"
    prompt = f"""You are a top short-video storyteller. Write a 60 second vertical video script about: "{TOPIC}".
Narration language: {language}.
RULES for narration:
- Scene 1 is a strong HOOK (a surprising question or bold line), not an introduction.
- Short, punchy sentences. Each scene 14-17 words (total about 150-160 words, so the video is about 60 seconds).
- Sound like a real Indian kids' YouTuber talking to a 4-7 year old, NOT like an AI or a textbook: warm, excited, everyday spoken words kids hear at home (in Hindi, common English words like colour, ball, school, bus are fine).
- Vary every sentence: different openings, mix in questions ("बताओ तो...?" / "Can you guess...?"), sounds ("दहाड़! रोओओर!" / "Moo!"), tiny actions ("ताली बजाओ!" / "Clap with me!"), counting and one small surprise. Never start two scenes the same way.
- No filler or repeated praise: words like प्यारा/प्यारी, चमकीला, मज़ेदार, कितना, देखो (or cute, amazing, look) may appear at most ONCE in the whole script.
- Build a mini story: setup, tension or fun conflict, twist, takeaway.
- Last scene: one clear takeaway plus a call to action (follow / share / comment).
- If the topic is funny, be playful and use light humor. If it is informational, be clear and accurate with NO made-up statistics.
- Stay 100% on the topic. Every scene must be about the topic and nothing else.
- If the topic is a LEARNING LIST (letters, numbers, colors, shapes, animals, fruits, Hindi varnamala, etc.):
  stick to ONE subject only (only colours, or only animals...) - never mix colours, shapes and animals in one video;
  cover exactly the items/range named in the topic, IN ORDER, one item per scene (two items per scene only if there are more items than scenes);
  scene 1 can be a very short hook that already starts the list, the last scene is a quick recap + call to action;
  the narration says the item clearly and simply (e.g. "A for Apple! A... A... Apple!");
  the image_prompt shows EXACTLY what that scene's narration names (red colour -> a big red apple or red ball, not a random shape), big and clear in a simple bright kid-friendly setting (no letters);
  "character" must be an empty string "".
RULES for pictures:
{{PIC_RULES}}
- Use gentle words only (arguing, competing, surprised). Never use fight, battle, kill, blood, weapon.
- No text, letters or logos inside the pictures.
- "mood" is the background music style: exactly one of playful, calm, dramatic.
Return ONLY valid JSON, no markdown:
{{"title": "...", "mood": "playful", "character": "...", "scenes": [{{"narration": "...", "image_prompt": "..."}}]}}
Use exactly {SCENES} scenes."""
    if IMAGE_STYLE == "real":
        pic = ('- "character": one line describing the main real-looking people (age, gender, skin tone, hair, face, clothes) so they look the same in every scene.\n'
               '- Each "image_prompt" is in English and describes a REAL PHOTO: the action, facial emotion, camera angle (close-up, wide shot, over-the-shoulder), real location and lighting (office, street, home, golden hour). Do NOT repeat the character line, it is added automatically.')
    else:
        pic = ('- "character": one line describing the main cartoon characters\' look (colors, faces, clothes) so they look the same in every scene.\n'
               '- Each "image_prompt" is in English: describe the action, emotion, camera angle and background. Do NOT repeat the character line, it is added automatically.')
    prompt = prompt.replace("{PIC_RULES}", pic)
    models = ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-super-vl-120b-a12b"]   # old llama/nemotron models were retired by NVIDIA
    bad = ("guard", "safety", "embed", "reward", "vl", "parse", "retriev", "rerank", "clip", "nemoguard", "topic")
    try:
        r = requests.get(f"{base}/models", headers=hdr, timeout=30)
        ids = [m["id"] for m in r.json()["data"]]
        models += [m for m in ids if ("nemotron" in m.lower() or "instruct" in m.lower())
                   and not any(b in m.lower() for b in bad)]
    except Exception as e:
        print("could not list models:", e)
    if LONG:
        return write_long_script(models, pic)
    return llm_json(prompt, models, lambda d: d.get("scenes") and "narration" in d["scenes"][0] and "image_prompt" in d["scenes"][0])

# ---------- 1b. LONG STORY SCRIPT (outline first, then chapter by chapter so the story stays on track) ----------
import random
SETTINGS = ["a small village in Rajasthan", "a busy street in Mumbai", "a forest near the Himalayas", "a farm in Punjab",
            "a fishing village in Kerala", "a school in Lucknow", "a tea garden in Assam", "a riverside town in Bihar",
            "a hill station in Uttarakhand", "an old haveli in Jaipur", "a mango orchard in Uttar Pradesh", "a jungle in Madhya Pradesh"]
HEROES = ["a curious girl", "a shy boy", "a clever crow", "a little elephant", "a lazy lion cub", "a kind grandmother",
          "a naughty monkey", "a brave tortoise", "twin siblings", "a talking parrot", "a small puppy", "a young farmer"]
VALUES = ["honesty", "sharing", "courage", "kindness to animals", "hard work", "patience", "respecting elders",
          "teamwork", "saving water", "telling the truth", "not being greedy", "gratitude", "helping others", "self-belief"]
GENRE_RULES = {
    "moral": ("an ORIGINAL moral story for kids (do NOT retell Panchatantra, Akbar-Birbal or any famous story). "
              "Warm, simple, fun; one clear lesson kids learn; end with the moral said clearly."),
    "horror": ("a KID-SAFE spooky mystery story (like a mysterious sound, a friendly ghost, a dark old house). "
               "Mild suspense only: NO gore, blood, death, violence or truly scary images. The mystery gets a clever, "
               "comforting explanation and a lesson about courage. Pictures must look mysterious but cute, never frightening."),
    "real": ("a TRUE inspiring story from real history or real life (scientists, inventors, freedom fighters, sportspeople, "
             "kind ordinary heroes), told simply for kids. Use ONLY well-known documented facts; do NOT invent quotes, dates or numbers; "
             "if unsure, leave the detail out. Show real people only as illustrations, not photos."),
}

def write_long_script(models, pic):
    language = "Hindi (Devanagari script), simple everyday spoken Hindi - not formal or shuddh Hindi" if LANG == "hi" else "simple English"
    chapters = max(3, round(SCENES / 9))
    per = [SCENES // chapters + (1 if i < SCENES % chapters else 0) for i in range(chapters)]
    seed_hint = ""
    if STORY_GENRE != "real":
        seed_hint = (f"For uniqueness, use these ingredients (change them only if they clash with the topic): "
                     f"setting {random.choice(SETTINGS)}, main hero {random.choice(HEROES)}, value {random.choice(VALUES)}.")
    outline_prompt = f"""You are a top kids' story writer. Plan a {LONG_MINUTES}-minute YouTube story video.
Topic / idea: "{TOPIC}"
Story type: {GENRE_RULES[STORY_GENRE]}
{seed_hint}
Narration language later: {language}.
Make it engaging: a strong opening hook, a problem, rising tension, a twist, a satisfying ending.
Return ONLY valid JSON, no markdown:
{{"title": "catchy title in {language}", "mood": "playful or calm or dramatic", "character": "one line describing the main characters' look so they look the same in every picture (English)",
 "chapters": [{{"name": "...", "summary": "2-3 sentences of what happens"}}]}}
Use exactly {chapters} chapters."""
    outline = llm_json(outline_prompt, models, lambda d: d.get("chapters") and d.get("title"))
    chapters = max(1, min(chapters, len(outline["chapters"])))   # spread all scenes over the chapters the AI really gave
    per = [SCENES // chapters + (1 if i < SCENES % chapters else 0) for i in range(chapters)]
    print(f"story: {outline['title']} ({chapters} chapters, {SCENES} scenes)")
    scenes = []
    for ci, ch in enumerate(outline["chapters"][:chapters]):
        k = per[ci] if ci < len(per) else 9
        recent = " ".join(s["narration"] for s in scenes[-3:])
        chap_prompt = f"""You are writing chapter {ci+1} of {chapters} of this kids' story video.
Story type: {GENRE_RULES[STORY_GENRE]}
Full outline: {json.dumps(outline, ensure_ascii=False)}
This chapter: {ch.get("name","")} - {ch.get("summary","")}
Story so far ends with: "{recent or '(this is the beginning)'}"
Write exactly {k} scenes that continue smoothly from the story so far.
Narration language: {language}. Each scene 14-17 words, like a grandparent telling a bedtime story: simple, vivid, with feelings, some dialogue between characters and sound words.
- Sound like a real Indian kids' YouTuber talking to a 4-7 year old, NOT like an AI or a textbook: warm, excited, everyday spoken words kids hear at home (in Hindi, common English words like colour, ball, school, bus are fine).
- Vary every sentence: different openings, mix in questions ("बताओ तो...?" / "Can you guess...?"), sounds ("दहाड़! रोओओर!" / "Moo!"), tiny actions ("ताली बजाओ!" / "Clap with me!"), counting and one small surprise. Never start two scenes the same way.
- No filler or repeated praise: words like प्यारा/प्यारी, चमकीला, मज़ेदार, कितना, देखो (or cute, amazing, look) may appear at most ONCE in the whole script.
{"The last scene must end the story and say the lesson clearly, then ask kids to like and subscribe." if ci == chapters - 1 else "Do not end the story yet."}
RULES for pictures:
{pic}
- Use gentle words only. Never use fight, battle, kill, blood, weapon.
- No text, letters or logos inside the pictures.
Return ONLY valid JSON, no markdown: {{"scenes": [{{"narration": "...", "image_prompt": "..."}}]}}"""
        part = llm_json(chap_prompt, models, lambda d: d.get("scenes") and "narration" in d["scenes"][0] and "image_prompt" in d["scenes"][0])
        scenes += part["scenes"][:k]
        print(f"  chapter {ci+1}/{chapters}: {len(part['scenes'][:k])} scenes")
    return {"title": outline["title"], "mood": outline.get("mood", "calm"), "character": outline.get("character", ""), "scenes": scenes}

def _parse_json(text):
    if isinstance(text, dict):
        return text
    text = re.sub(r"^```(?:json)?|```$", "", str(text).strip(), flags=re.M).strip()
    return json.loads(text[text.find("{"): text.rfind("}") + 1])

def llm_json(prompt, models, ok):
    # 1) NVIDIA models, 2) Cloudflare Workers AI backup (free, same CF_TOKEN as the pictures)
    last = None
    for kn, key in enumerate(NVIDIA_KEYS, 1):
      hdr = {"Authorization": f"Bearer {key}"}
      denied = 0
      for model in dict.fromkeys(models):
        if denied >= 3: break
        if model in DEAD: continue
        try:
            r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=hdr, timeout=120,
                              json={"model": model, "temperature": 0.7, "max_tokens": 8000,   # reasoning models think first, so give room
                                    "messages": [{"role": "user", "content": prompt}]})
            if r.status_code in (404, 410):
                DEAD.add(model)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            data = _parse_json(msg.get("content") or "")
            if not ok(data):
                raise ValueError("reply had no proper scenes")
            print(f"text made with {model} (NVIDIA key {kn})")
            return data
        except Exception as e:
            last = e
            print(f"  {model} failed, trying next ({str(e)[:80]})")
            if "401" in str(e) or "403" in str(e) or "429" in str(e):   # one model can be blocked; give up on this key after 3 refusals
                denied += 1
                if denied >= 3: print(f"  NVIDIA key {kn} refused 3 times -> trying next key / Cloudflare backup")
    for model in CF_TEXT_MODELS:
        try:
            r = requests.post(f"https://api.cloudflare.com/client/v4/accounts/{CF_ID}/ai/run/{model}",
                              headers={"Authorization": f"Bearer {CF_TOKEN}"}, timeout=180,
                              json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 2500, "temperature": 0.7})
            r.raise_for_status()
            data = _parse_json(r.json()["result"]["response"])
            if not ok(data):
                raise ValueError("reply had no proper scenes")
            print("text made with Cloudflare", model)
            return data
        except Exception as e:
            last = e
            print(f"  cloudflare {model} failed ({str(e)[:80]})")
    raise RuntimeError(f"No model worked: {last}")

# ---------- 0. AUTO TOPIC (daily scheduled runs, or empty topic) ----------
def previous_titles():
    repo, tok = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GH_TOKEN")
    if not repo: return []
    try:
        h = {"Authorization": f"Bearer {tok}"} if tok else {}
        r = requests.get(f"https://api.github.com/repos/{repo}/releases?per_page=60", headers=h, timeout=30)
        return [x.get("name", "") for x in r.json() if x.get("tag_name", "").startswith("video-")]
    except Exception:
        return []

def auto_topic():
    lang = "Hindi written in English letters (Hinglish)" if LANG == "hi" else "English"
    done = previous_titles()
    prompt = f"""Suggest ONE fresh, catchy topic for a 60 second YouTube Short in this niche: {AUTO_NICHE}.
It must be evergreen and positive, about ordinary people (no real famous people, no news, no politics, no religion).
It must be about ONE single subject (e.g. only fruits, only animal sounds, only numbers 1-10, or one small story) - never combine colours, shapes and animals in one video.
Write the topic in {lang}, max 12 words.
Do NOT repeat any of these earlier topics: {json.dumps(done[:40], ensure_ascii=False)}
Return ONLY valid JSON: {{"topic": "..."}}"""
    data = llm_json(prompt, ["nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-3-ultra-550b-a55b"],
                    lambda d: bool(str(d.get("topic", "")).strip()))
    return data["topic"].strip()

# ---------- 2. IMAGES (Cloudflare FLUX -> NVIDIA -> Pollinations -> colour placeholder) ----------
def placeholder(path, idx):
    cols = [(255,190,80),(120,200,255),(160,230,160),(255,150,170),(200,170,255)]
    Image.new("RGB", (1024, 1024), cols[idx % 5]).save(path)

def _save_b64(b64, path):
    path.write_bytes(base64.b64decode(b64))
    Image.open(path).verify()

def make_image(prompt, path, idx):
    full = f"{prompt}, {STYLE}"
    if CF_ID and CF_TOKEN and 'cf' not in SKIP:
        tries = [full,
                 f"{prompt[:180]}, {STYLE_SHORT}",
                 f"{TOPIC}, {STYLE_SHORT}"]
        for n, pr in enumerate(tries):
            try:
                r = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{CF_ID}/ai/run/@cf/black-forest-labs/flux-1-schnell",
                    headers={"Authorization": f"Bearer {CF_TOKEN}"}, json={"prompt": pr, "steps": 4}, timeout=120)
                if r.status_code in (401, 403):
                    print("  cloudflare token/permission problem:", r.text[:150]); SKIP.add("cf"); break
                if r.status_code == 429:
                    if n == 2:
                        print("  cloudflare daily free limit reached -> skipping Cloudflare for this video"); SKIP.add("cf")
                    time.sleep(5); continue
                if r.status_code != 200:
                    print(f"  cloudflare {r.status_code} (try {n+1}/3): {r.text[:150]}"); continue
                _save_b64(r.json()["result"]["image"], path); return
            except Exception as e:
                print(f"  cloudflare error (try {n+1}/3): {str(e)[:80]}")
    for kn, key in enumerate(NVIDIA_KEYS, 1):
        if f"nv{kn}" in SKIP: continue
        for attempt in range(2):   # one quick retry for timeouts / busy server
          try:
            body = {"prompt": full, "width": IMG_W, "height": IMG_H, "steps": 4}   # NVIDIA does not accept "seed"
            r = requests.post("https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell",
                              headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                              json=body, timeout=60)
            if r.status_code in (401, 403):
                print(f"  nvidia image key {kn} refused -> not using it again ({r.text[:80]})"); SKIP.add(f"nv{kn}"); break
            if r.status_code in (400, 422):
                print("  nvidia image said:", r.text[:200]); break
            if r.status_code == 429:
                time.sleep(8)
            r.raise_for_status()
            d = r.json()
            _save_b64(d.get("artifacts", [{}])[0].get("base64") or d.get("image"), path)
            FAILS[f"nv{kn}"] = 0; return
          except Exception as e:
            FAILS[f"nv{kn}"] = FAILS.get(f"nv{kn}", 0) + 1
            print(f"  nvidia image (key {kn}) try {attempt+1}/2 failed ({str(e)[:60]})")
            if FAILS[f"nv{kn}"] >= 6:
                print(f"  nvidia key {kn} keeps failing -> skipping it for this video"); SKIP.add(f"nv{kn}"); break
            time.sleep(3)
    if 'pl' not in SKIP:
        try:
            time.sleep(6)
            url = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(full)
            r = requests.get(url, params={"width": IMG_W, "height": IMG_H, "nologo": "true", "seed": SEED}, timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content); Image.open(path).verify(); FAILS["pl"] = 0; return
        except Exception as e:
            FAILS["pl"] = FAILS.get("pl", 0) + 1
            print(f"  pollinations failed ({str(e)[:50]})")
            if FAILS["pl"] >= 3:
                print("  pollinations keeps failing -> skipping it for this video"); SKIP.add("pl")
    print(f"  image {idx}: using colour placeholder")
    placeholder(path, idx)

# ---------- 3. VOICE (edge-tts, free) ----------
async def _tts(text, out):
    import edge_tts
    try:
        com = edge_tts.Communicate(text, VOICES[(LANG, VOICE_GENDER)], rate=VOICE_RATE, boundary="WordBoundary")
    except TypeError:   # older edge-tts gives word timings by default
        com = edge_tts.Communicate(text, VOICES[(LANG, VOICE_GENDER)], rate=VOICE_RATE)
    words = []
    with open(out, "wb") as f:
        async for ch in com.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                st = ch["offset"] / 1e7
                words.append((st, st + ch["duration"] / 1e7, ch["text"]))
    return words

def make_voice(text, out):
    # returns [(start_sec, end_sec, word), ...] for word-by-word captions (empty list = fall back to old captions)
    with ThreadPoolExecutor() as ex:   # thread, because Colab already has a running event loop
        return ex.submit(asyncio.run, _tts(text, out)).result()

def audio_len(p):
    return float(run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",str(p)]).strip())

# ---------- 4. SCENE LAYERS (background, moving picture, timed captions) ----------
HINDI_FONT_URL = ("https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts/"
                  "NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Bold.ttf")

def ensure_hindi_font():
    # Hindi needs a Devanagari font, otherwise captions show empty boxes. Download it once if the system has none.
    if any(Path(fp).exists() and "Devanagari" in fp for fp in FONTS):
        return
    dst = OUT / "fonts" / "NotoSansDevanagari-Bold.ttf"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(HINDI_FONT_URL, timeout=60)
        r.raise_for_status()
        dst.write_bytes(r.content)
        ImageFont.truetype(str(dst), 20)
        print("downloaded Hindi font")
    except Exception as e:
        print("could not get Hindi font, Hindi captions may show boxes:", str(e)[:80])

def get_font(size, text=""):
    # Hindi text -> Devanagari font; English text -> DejaVu/Liberation (the Hindi font has no English letters)
    try:
        engine = ImageFont.Layout.RAQM   # joins Hindi letters (matras) correctly
    except Exception:
        engine = None
    hindi = any("\u0900" <= ch <= "\u097f" for ch in text)
    order = [fp for fp in FONTS if ("Devanagari" in fp) == hindi] + [fp for fp in FONTS if ("Devanagari" in fp) != hindi]
    for fp in order:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size, layout_engine=engine)
            except Exception:
                return ImageFont.truetype(fp, size)
    return ImageFont.load_default()

def split_chunks(text, n=4):
    words = text.split()
    return [" ".join(words[i:i+n]) for i in range(0, len(words), n)] or [text]

def caption_png(text, y_start, dst):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = get_font(78, text)
    lines = textwrap.wrap(text, width=17)[:2]
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=font)
        d.text(((W - tw) / 2, y_start + i * 100), ln, font=font, fill=(255, 224, 40, 255),
               stroke_width=9, stroke_fill=(0, 0, 0, 255))
    layer.save(dst)

def word_caption_png(words, hi, y_start, dst):
    # one group of words, the word being spoken is yellow and bigger feel, others white
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = get_font(82, " ".join(words))
    space = d.textlength(" ", font=font)
    lines, cur, cur_w = [], [], 0.0
    for i, w in enumerate(words):
        ww = d.textlength(w, font=get_font(82, w))
        if cur and cur_w + space + ww > W - 120:
            lines.append(cur); cur, cur_w = [], 0.0
        cur_w += (space if cur else 0) + ww; cur.append((i, w, ww))
    if cur: lines.append(cur)
    for li, ln in enumerate(lines[:2]):
        total = sum(x[2] for x in ln) + space * (len(ln) - 1)
        x = (W - total) / 2
        for i, w, ww in ln:
            col = (255, 224, 40, 255) if i == hi else (255, 255, 255, 255)
            d.text((x, y_start + li * 108), w, font=get_font(82, w), fill=col, stroke_width=9, stroke_fill=(0, 0, 0, 255))
            x += ww + space
    layer.save(dst)

def prepare_scene(raw, idx, text, voice_dur, words=None):
    img = Image.open(raw).convert("RGB")
    fx = 0
    if LONG:     # 16:9 frame: wide picture fills it; square picture centred with a blurred copy at the sides
        s = max(W / img.width, H / img.height)
        bg = img.resize((int(img.width * s), int(img.height * s)))
        l, t = (bg.width - W) // 2, (bg.height - H) // 2
        bg = bg.crop((l, t, l + W, t + H))
        if img.width / img.height > 1.3:
            fg = bg; bg = Image.new("RGB", (W, H), (0, 0, 0))
        else:
            bg = ImageEnhance.Brightness(bg.filter(ImageFilter.GaussianBlur(45))).enhance(0.65)
            fg = img.resize((H, H)); fx = (W - H) // 2
        fy = 0; text_y = H - 250
    elif (img.width / img.height) > 0.8:   # whole picture visible, blurred copy fills the rest of the tall frame
        s = max(W / img.width, H / img.height)
        bg = img.resize((int(img.width * s), int(img.height * s)))
        l, t = (bg.width - W) // 2, (bg.height - H) // 2
        bg = bg.crop((l, t, l + W, t + H)).filter(ImageFilter.GaussianBlur(45))
        bg = ImageEnhance.Brightness(bg).enhance(0.65)
        fg = img.resize((W, W)); fy = 300; text_y = 300 + W + 70
    else:        # tall picture: fill the frame
        s = max(W / img.width, H / img.height)
        fg = img.resize((int(img.width * s), int(img.height * s)))
        l, t = (fg.width - W) // 2, (fg.height - H) // 2
        fg = fg.crop((l, t, l + W, t + H)); bg = Image.new("RGB", (W, H), (0, 0, 0)); fy = 0; text_y = 1420
    bg_p, fg_p = OUT / f"s{idx}_bg.png", OUT / f"s{idx}_fg.png"
    bg.save(bg_p); fg.save(fg_p)
    caps = []
    if words:   # word-by-word captions: groups of 3 words, the spoken word highlighted
        for g in range(0, len(words), 3):
            grp = words[g:g + 3]
            for k, (st, en, w) in enumerate(grp):
                nxt = words[g + k + 1][0] if g + k + 1 < len(words) else max(en, voice_dur)
                cp = OUT / f"s{idx}_w{g + k}.png"
                word_caption_png([x[2] for x in grp], k, text_y, cp)
                caps.append((cp, TRANS + st, TRANS + nxt))
        return dict(bg=bg_p, fg=fg_p, fw=fg.width, fh=fg.height, fx=fx, fy=fy, caps=caps)
    chunks = split_chunks(text)
    total_chars = sum(len(c) for c in chunks)
    t0 = TRANS
    for j, c in enumerate(chunks):
        span = voice_dur * len(c) / total_chars
        cp = OUT / f"s{idx}_cap{j}.png"
        caption_png(c, text_y, cp)
        caps.append((cp, t0, t0 + span)); t0 += span
    return dict(bg=bg_p, fg=fg_p, fw=fg.width, fh=fg.height, fx=fx, fy=fy, caps=caps)

MOTIONS = [  # gentle camera moves (small zoom so characters near the edges are never cut off)
    "z='1+0.10*on/{F}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2'",          # slow zoom in
    "z='1.08':x='(iw-iw/zoom)*on/{F}':y='(ih-ih/zoom)/2'",              # pan right
    "z='1.10-0.10*on/{F}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2'",       # slow zoom out
    "z='1.08':x='(iw-iw/zoom)*(1-on/{F})':y='(ih-ih/zoom)/2'",          # pan left
    "z='1+0.09*on/{F}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*0.25'",       # zoom toward top
]

def scene_clip(sc, audio, out, idx):
    al = audio_len(audio)
    dur = TRANS + al + TRANS          # voice starts after the transition and ends before the next one
    F = int(dur * FPS)
    k = 2 if sc["fw"] == W and sc["fh"] == W else 1.5
    motion = MOTIONS[idx % len(MOTIONS)].replace("{F}", str(F))
    inputs = ["-loop", "1", "-framerate", str(FPS), "-i", str(sc["bg"]),
              "-loop", "1", "-framerate", str(FPS), "-i", str(sc["fg"])]
    for cp, _, _ in sc["caps"]:
        inputs += ["-loop", "1", "-framerate", str(FPS), "-i", str(cp)]
    inputs += ["-i", str(audio)]
    a_idx = 2 + len(sc["caps"])
    fg_chain = (f"[1:v]scale={int(sc['fw']*k)}:{int(sc['fh']*k)},"
                f"zoompan={motion}:d={F}:s={sc['fw']}x{sc['fh']}:fps={FPS}[fg];"
                f"[0:v][fg]overlay={sc.get('fx', 0)}:{sc['fy']}:shortest=1[b0];")
    chain, prev = "", "b0"
    for j, (_, t0, t1) in enumerate(sc["caps"]):
        chain += f"[{prev}][{2+j}:v]overlay=0:0:enable='between(t,{t0:.2f},{t1:.2f})'[b{j+1}];"
        prev = f"b{j+1}"
    fc = fg_chain + chain + f"[{prev}]setsar=1,format=yuv420p[v];" \
         f"[{a_idx}:a]adelay={int(TRANS*1000)}|{int(TRANS*1000)},apad,aformat=sample_rates={SR}:channel_layouts=stereo[a]"
    run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
         "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-c:a", "aac", "-ar", str(SR), "-ac", "2", str(out)])
    return dur

# ---------- 5. CINEMATIC JOIN: transitions + colour grade + vignette + film grain + © watermark ----------
def watermark_png(dst):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    text = f"© {CHANNEL_NAME}"
    font = get_font(44, text)
    tw = d.textlength(text, font=font)
    d.text((W - tw - 40, 60), text, font=font, fill=(255, 255, 255, 190), stroke_width=3, stroke_fill=(0, 0, 0, 140))
    layer.save(dst)

def join_clips(clips, durs, out):
    n = len(clips)
    names = ["fade", "smoothleft", "fade", "smoothright"]
    inputs = []
    for c in clips: inputs += ["-i", str(c)]
    fc, vprev, aprev, offsets, acc = "", "0:v", "0:a", [], 0.0
    for i in range(1, n):
        acc += durs[i-1]
        off = acc - i * TRANS
        offsets.append(off)
        tr = names[(i-1) % len(names)]
        fc += f"[{vprev}][{i}:v]xfade=transition={tr}:duration={TRANS}:offset={off:.3f}[xv{i}];"
        fc += f"[{aprev}][{i}:a]acrossfade=d={TRANS}[xa{i}];"
        vprev, aprev = f"xv{i}", f"xa{i}"
    grade = "eq=contrast=1.07:saturation=1.12,vignette=PI/4.5,noise=alls=3:allf=t"
    wm = OUT / "watermark.png"; watermark_png(wm)
    inputs += ["-loop", "1", "-framerate", str(FPS), "-i", str(wm)]
    fc += f"[{vprev}]{grade}[g];[g][{n}:v]overlay=0:0:shortest=1,format=yuv420p[vout]"
    run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[vout]", "-map", f"[{aprev}]",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-maxrate", "6M", "-bufsize", "12M",
         "-c:a", "aac", "-ar", str(SR), str(out)])
    return offsets, sum(durs) - (n - 1) * TRANS

def join_many(clips, durs, out, size=10):
    # long videos: join 10 scenes at a time (keeps ffmpeg memory low), then glue the parts together
    if len(clips) <= size + 2:
        return join_clips(clips, durs, out)
    parts, total = [], 0.0
    for i in range(0, len(clips), size):
        pp = OUT / f"part{i // size}.mp4"
        _, t = join_clips(clips[i:i + size], durs[i:i + size], pp)
        parts.append(pp); total += t
        print(f"  joined part {i // size + 1}/{(len(clips) + size - 1) // size}")
    lst = OUT / "parts.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    return [], total

# ---------- 6. BACKGROUND MUSIC (soft royalty-free tracks, auto-lowered under the voice) ----------
# Tracks by Kevin MacLeod (incompetech.com), free under CC BY 4.0 -> put the credit line (saved in out/music_credit.txt) in your video description.
MUSIC_BASES = ["https://incompetech.com/music/royalty-free/mp3-royaltyfree/",
               "https://aux.incompetech.com/music/royalty-free/mp3-royaltyfree/"]
SOFT = ["Wallpaper.mp3", "Dreamer.mp3", "Carefree.mp3", "Local Forecast - Elevator.mp3"]
MUSIC = {   # soft music for every mood; first track that downloads is used
    "playful":  ["Carefree.mp3"] + SOFT,
    "calm":     SOFT,
    "dramatic": ["Dreamer.mp3"] + SOFT,
}
MUSIC_VOL = 0.18      # music level when nobody speaks (0.12 softer, 0.25 louder)

def get_music(mood):
    own = Path("music.mp3")      # if you upload your own music.mp3 to Colab, it is used instead
    if own.exists():
        print("using your music.mp3")
        return own, None
    for name in MUSIC.get(mood, MUSIC["playful"]):
        dst = OUT / ("track_" + name.replace(" ", "_"))
        for b in MUSIC_BASES:
            try:
                r = requests.get(b + urllib.parse.quote(name), timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200 or len(r.content) < 100000:
                    continue
                dst.write_bytes(r.content)
                audio_len(dst)   # check it is real audio
                title = name[:-4]
                credit = (f'"{title}" Kevin MacLeod (incompetech.com)\n'
                          "Licensed under Creative Commons: By Attribution 4.0 License\n"
                          "http://creativecommons.org/licenses/by/4.0/")
                (OUT / "music_credit.txt").write_text(credit, encoding="utf-8")
                print(f"using music: {title}")
                return dst, credit
            except Exception as e:
                print(f"  music {name} failed ({str(e)[:60]})")
    return None, None


def write_wav(path, mono):
    pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
    st = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR); f.writeframes(st.tobytes())

def add_sound(video, total, offsets, mood, final):
    music_wav = OUT / "music.wav"
    track, credit = get_music(mood)
    if track:   # loop to video length, soft fade in and fade out
        run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(track), "-t", f"{total:.2f}",
             "-af", f"afade=t=in:st=0:d=1.5,afade=t=out:st={max(0, total - 2.5):.2f}:d=2.5",
             "-ar", str(SR), "-ac", "2", str(music_wav)])
    else:
        print("no music could be downloaded, video will have voice + effects only")
        write_wav(music_wav, np.zeros(int(total * SR)))
    base = ("[0:a]asplit=2[v1][v2];"
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={MUSIC_VOL}[m];"
            "[m][v2]sidechaincompress=threshold=0.02:ratio=12:attack=10:release=600[md];"   # music dips while the voice speaks
            "[v1][md]amix=inputs=2:duration=first:dropout_transition=0{NORM}[a]")
    last = None
    for norm, boost in ((":normalize=0", ""), ("", ",volume=2.2")):
        try:
            fc = base.replace("{NORM}", norm).replace("[a]", f"{boost}[a]") if boost else base.replace("{NORM}", norm)
            run(["ffmpeg", "-y", "-i", str(video), "-i", str(music_wav),
                 "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(final)])
            if credit:
                print("\nPUT THIS IN YOUR VIDEO DESCRIPTION (music credit):\n" + credit + "\n")
            return
        except RuntimeError as e:
            last = e
    raise last

# ===== RUN =====
def build_video(script):
    mood = script.get("mood", "playful")
    if mood not in MUSIC: mood = "playful"
    clips, durs = [], []
    for i, sc in enumerate(script["scenes"]):
        print(f"scene {i+1}/{len(script['scenes'])}: picture, voice, camera move")
        raw, mp3, clip = OUT / f"s{i}_raw.png", OUT / f"s{i}.mp3", OUT / f"s{i}.mp4"
        ch = str(script.get("character", "") or "").strip()
        make_image(f"{ch}. {sc['image_prompt']}" if ch else sc["image_prompt"], raw, i)
        words = make_voice(sc["narration"], mp3)
        layers = prepare_scene(raw, i, sc["narration"], audio_len(mp3), words)
        durs.append(scene_clip(layers, mp3, clip, i)); clips.append(clip)
    print("joining scenes with transitions + cinematic look...")
    joined = OUT / "joined.mp4"
    offsets, total = join_many(clips, durs, joined)
    print(f"adding music ({mood}) and sound effects...")
    add_sound(joined, total, offsets, mood, OUT / "final.mp4")
    return OUT / "final.mp4"

# ---------- 7. YOUTUBE UPLOAD (free YouTube Data API; needs YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN secrets) ----------
def upload_youtube(video, title, description, privacy):
    cid, csec, rtok = (os.environ.get(k, "").strip() for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"))
    if not (cid and csec and rtok):
        print("YouTube secrets missing -> skipping YouTube upload"); return None
    tok = requests.post("https://oauth2.googleapis.com/token", timeout=30, data={
        "client_id": cid, "client_secret": csec, "refresh_token": rtok, "grant_type": "refresh_token"})
    tok.raise_for_status()
    access = tok.json()["access_token"]
    meta = {"snippet": {"title": title[:95], "description": description[:4900], "categoryId": "27",
                        "tags": ["shorts", "motivation", "education"]},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}}
    size = Path(video).stat().st_size
    r = requests.post("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                      headers={"Authorization": f"Bearer {access}", "Content-Type": "application/json; charset=UTF-8",
                               "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)},
                      data=json.dumps(meta), timeout=60)
    r.raise_for_status()
    with open(video, "rb") as f:
        up = requests.put(r.headers["Location"], data=f, timeout=900,
                          headers={"Authorization": f"Bearer {access}", "Content-Type": "video/mp4"})
    up.raise_for_status()
    vid = up.json()["id"]
    print(f"uploaded to YouTube ({privacy}): https://youtube.com/shorts/{vid}")
    return vid

ensure_hindi_font()
if not TOPIC and LONG:
    TOPIC = f"any original {STORY_GENRE} story idea of your choice"
if not TOPIC:
    print("0/3 no topic given -> picking one automatically...")
    TOPIC = auto_topic()
    print("auto topic:", TOPIC)
print("1/3 writing script...")
script = write_script()
(OUT / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
print("2/3 building scenes...")
final = build_video(script)
print("3/3 DONE ->", final)

# title + description for the GitHub release (your site) and YouTube
_title = (script.get("title") or TOPIC).strip()
_credit = (OUT / "music_credit.txt").read_text(encoding="utf-8") if (OUT / "music_credit.txt").exists() else ""
_desc = "\n".join(s["narration"] for s in script["scenes"]) + ("\n\nMusic:\n" + _credit if _credit else "") + ("" if LONG else "\n\n#shorts")
(OUT / "title.txt").write_text(_title, encoding="utf-8")
(OUT / "release_notes.txt").write_text(f"Topic: {TOPIC}\nLanguage: {LANG} | Voice: {VOICE_GENDER} | Style: {IMAGE_STYLE}\n\n{_desc}", encoding="utf-8")
if os.environ.get("YT_UPLOAD", "").lower() in ("yes", "true", "1"):
    try:
        upload_youtube(final, _title if LONG else f"{_title} #shorts", _desc, os.environ.get("YT_PRIVACY", "private") or "private")
    except Exception as e:
        print("YouTube upload failed (video is still saved):", str(e)[:300])

try:   # in Colab: show and download; on GitHub: the workflow uploads out/final.mp4 for you
    from IPython.display import Video, display
    from google.colab import files
    display(Video(str(final), embed=True, width=300))
    files.download(str(final))
except Exception:
    pass

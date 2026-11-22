# ===== v3 REALISTIC - GitHub Actions version (run from the "Actions" tab -> Make video -> Run workflow) =====
# ffmpeg, fonts and pip packages are installed by .github/workflows/make-video.yml

import os, re, json, time, math, asyncio, base64, subprocess, textwrap, urllib.parse, getpass, wave
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ---------- INPUTS ----------
# Keys come from GitHub repo Secrets (Settings -> Secrets and variables -> Actions). Topic/language/voice come from the Run workflow form.
def secret(name, ask):
    v = os.environ.get(name, "").strip()
    if v: return v
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v: return v.strip()
    except Exception:
        pass
    if os.environ.get("GITHUB_ACTIONS"):
        raise RuntimeError(f"Missing secret {name}: add it in repo Settings -> Secrets and variables -> Actions")
    return getpass.getpass(ask).strip()

NVIDIA_KEY = secret("NVIDIA_KEY", "Paste your NVIDIA API key (nvapi-...): ")
CF_ID = "31fdd1f22c348a91b6da9d4a4e9c3b64"   # Cloudflare Account ID (not a secret)
CF_TOKEN = secret("CF_TOKEN", "Paste your Cloudflare API token: ")
TOPIC = (os.environ.get("TOPIC") or input("Video topic: ")).strip()
_lang = os.environ.get("VIDEO_LANG") or input("Language - en or hi (hindi) [en]: ")
LANG = "hi" if _lang.strip().lower().startswith("h") else "en"
_voice = os.environ.get("VOICE_GENDER") or input("Voice - m (male) or f (female) [f]: ")
VOICE_GENDER = "m" if _voice.strip().lower().startswith("m") else "f"
_style = os.environ.get("IMAGE_STYLE") or input("Picture style - real (realistic humans) or cartoon [real]: ")
IMAGE_STYLE = "cartoon" if _style.strip().lower().startswith("c") else "real"

# ---------- CONSTANTS ----------
SCENES = 8
W, H, FPS = 1080, 1920, 25
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
STYLE = STYLES[IMAGE_STYLE]
STYLE_SHORT = {"real": "photorealistic cinematic photograph, natural lighting, 35mm lens",
               "cartoon": "cute colorful cartoon illustration, flat vector style"}[IMAGE_STYLE]
FONTS = ["/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
         "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Bold.ttf",
         str(Path("out") / "fonts" / "NotoSansDevanagari-Bold.ttf"),
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
SKIP = set()          # image services that failed with a real problem are skipped for the rest of the video

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-1800:])
    return p.stdout

# ---------- 1. SCRIPT (NVIDIA free API) ----------
def write_script():
    base = "https://integrate.api.nvidia.com/v1"
    hdr = {"Authorization": f"Bearer {NVIDIA_KEY}"}
    language = "Hindi (Devanagari script)" if LANG == "hi" else "simple English"
    prompt = f"""You are a top short-video storyteller. Write a 45-60 second vertical video script about: "{TOPIC}".
Narration language: {language}.
RULES for narration:
- Scene 1 is a strong HOOK (a surprising question or bold line), not an introduction.
- Talk like a friend, not a textbook. Short, punchy sentences. Max 16 words per scene.
- Build a mini story: setup, tension or fun conflict, twist, takeaway.
- Last scene: one clear takeaway plus a call to action (follow / share / comment).
- If the topic is funny, be playful and use light humor. If it is informational, be clear and accurate with NO made-up statistics.
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
    models = ["nvidia/nemotron-3-ultra-550b-a55b", "meta/llama-3.3-70b-instruct",
              "nvidia/llama-3.3-nemotron-super-49b-v1", "nvidia/llama-3.1-nemotron-70b-instruct",
              "meta/llama-3.1-70b-instruct", "nvidia/nemotron-3-super-120b"]
    bad = ("guard", "safety", "embed", "reward", "vl", "parse", "retriev", "rerank", "clip", "nemoguard", "topic")
    try:
        r = requests.get(f"{base}/models", headers=hdr, timeout=30)
        ids = [m["id"] for m in r.json()["data"]]
        models += [m for m in ids if ("nemotron" in m.lower() or "instruct" in m.lower())
                   and not any(b in m.lower() for b in bad)]
    except Exception as e:
        print("could not list models:", e)
    last = None
    for model in dict.fromkeys(models):
        try:
            r = requests.post(f"{base}/chat/completions", headers=hdr, timeout=120,
                              json={"model": model, "temperature": 0.7, "max_tokens": 2500,
                                    "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            if not data.get("scenes") or "narration" not in data["scenes"][0] or "image_prompt" not in data["scenes"][0]:
                raise ValueError("reply had no proper scenes")
            print("script made with", model)
            return data
        except Exception as e:
            last = e
            print(f"  {model} failed, trying next ({str(e)[:80]})")
    raise RuntimeError(f"No model worked: {last}")

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
                    time.sleep(10); continue
                if r.status_code != 200:
                    print(f"  cloudflare {r.status_code} (try {n+1}/3): {r.text[:150]}"); continue
                _save_b64(r.json()["result"]["image"], path); return
            except Exception as e:
                print(f"  cloudflare error (try {n+1}/3): {str(e)[:80]}")
    if 'nv' not in SKIP:
        try:
            r = requests.post("https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell",
                              headers={"Authorization": f"Bearer {NVIDIA_KEY}", "Accept": "application/json"},
                              json={"prompt": full, "width": 768, "height": 1344, "steps": 4}, timeout=30)
            r.raise_for_status()
            d = r.json()
            _save_b64(d.get("artifacts", [{}])[0].get("base64") or d.get("image"), path); return
        except Exception as e:
            print(f"  nvidia image failed, skipping it from now on ({str(e)[:50]})"); SKIP.add("nv")
    if 'pl' not in SKIP:
        try:
            time.sleep(6)
            url = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(full)
            r = requests.get(url, params={"width": 768, "height": 1344, "nologo": "true", "seed": idx}, timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content); Image.open(path).verify(); return
        except Exception as e:
            print(f"  pollinations failed, skipping it from now on ({str(e)[:50]})"); SKIP.add("pl")
    print(f"  image {idx}: using colour placeholder")
    placeholder(path, idx)

# ---------- 3. VOICE (edge-tts, free) ----------
async def _tts(text, out):
    import edge_tts
    await edge_tts.Communicate(text, VOICES[(LANG, VOICE_GENDER)], rate=VOICE_RATE).save(str(out))

def make_voice(text, out):
    with ThreadPoolExecutor() as ex:   # thread, because Colab already has a running event loop
        ex.submit(asyncio.run, _tts(text, out)).result()

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

def get_font(size):
    try:
        engine = ImageFont.Layout.RAQM   # joins Hindi letters (matras) correctly
    except Exception:
        engine = None
    for fp in FONTS:
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
    font = get_font(78)
    lines = textwrap.wrap(text, width=17)[:2]
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=font)
        d.text(((W - tw) / 2, y_start + i * 100), ln, font=font, fill=(255, 224, 40, 255),
               stroke_width=9, stroke_fill=(0, 0, 0, 255))
    layer.save(dst)

def prepare_scene(raw, idx, text, voice_dur):
    img = Image.open(raw).convert("RGB")
    square = (img.width / img.height) > 0.8
    if square:   # whole picture visible, blurred copy fills the rest of the tall frame
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
    chunks = split_chunks(text)
    total_chars = sum(len(c) for c in chunks)
    caps, t0 = [], TRANS
    for j, c in enumerate(chunks):
        span = voice_dur * len(c) / total_chars
        cp = OUT / f"s{idx}_cap{j}.png"
        caption_png(c, text_y, cp)
        caps.append((cp, t0, t0 + span)); t0 += span
    return dict(bg=bg_p, fg=fg_p, fw=fg.width, fh=fg.height, fy=fy, caps=caps)

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
                f"[0:v][fg]overlay=0:{sc['fy']}:shortest=1[b0];")
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

# ---------- 5. CINEMATIC JOIN: transitions + colour grade + vignette + film grain ----------
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
    grade = "eq=contrast=1.07:saturation=1.12,vignette=PI/4.5,noise=alls=3:allf=t,format=yuv420p"
    fc += f"[{vprev}]{grade}[vout]"
    run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[vout]", "-map", f"[{aprev}]",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-maxrate", "6M", "-bufsize", "12M",
         "-c:a", "aac", "-ar", str(SR), str(out)])
    return offsets, sum(durs) - (n - 1) * TRANS

# ---------- 6. BACKGROUND MUSIC (real royalty-free tracks, auto-lowered under the voice) + whoosh sound effects ----------
# Tracks by Kevin MacLeod (incompetech.com), free under CC BY 4.0 -> put the credit line (saved in out/music_credit.txt) in your video description.
MUSIC_BASES = ["https://incompetech.com/music/royalty-free/mp3-royaltyfree/",
               "https://aux.incompetech.com/music/royalty-free/mp3-royaltyfree/"]
MUSIC = {   # first track that downloads is used
    "playful":  ["Carefree.mp3", "Fluffing a Duck.mp3", "Monkeys Spinning Monkeys.mp3", "Sneaky Snitch.mp3"],
    "calm":     ["Wallpaper.mp3", "Local Forecast - Elevator.mp3", "Dreamer.mp3", "Carefree.mp3"],
    "dramatic": ["Investigations.mp3", "Darkest Child.mp3", "Impact Moderato.mp3", "Wallpaper.mp3"],
}
MUSIC_VOL = 0.22      # music level when nobody speaks (0.15 softer, 0.30 louder)

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


def synth_whooshes(total, times):
    out = np.zeros(int(total * SR))
    rng = np.random.default_rng(3)
    ln = int(0.7 * SR); x = np.linspace(0, 1, ln)
    noise = rng.standard_normal(ln)
    hp = noise - np.convolve(noise, np.ones(40) / 40, mode="same")   # keep the airy high part
    sweep = np.sin(2*np.pi*np.cumsum(180 + 900 * x) / SR) * 0.25
    w = (hp * 0.5 + sweep) * np.sin(np.pi * x) ** 2
    w = w / (np.max(np.abs(w)) + 1e-9) * 0.5
    for tm in times:
        s = int(max(0, tm - 0.35) * SR)
        e = min(len(out), s + ln)
        out[s:e] += w[: e - s]
    return out

def write_wav(path, mono):
    pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
    st = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR); f.writeframes(st.tobytes())

def add_sound(video, total, offsets, mood, final):
    music_wav, sfx_wav = OUT / "music.wav", OUT / "sfx.wav"
    track, credit = get_music(mood)
    if track:   # loop to video length, soft fade in and fade out
        run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(track), "-t", f"{total:.2f}",
             "-af", f"afade=t=in:st=0:d=1.5,afade=t=out:st={max(0, total - 2.5):.2f}:d=2.5",
             "-ar", str(SR), "-ac", "2", str(music_wav)])
    else:
        print("no music could be downloaded, video will have voice + effects only")
        write_wav(music_wav, np.zeros(int(total * SR)))
    write_wav(sfx_wav, synth_whooshes(total, [o + TRANS / 2 for o in offsets]))
    base = ("[0:a]asplit=2[v1][v2];"
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={MUSIC_VOL}[m];"
            "[m][v2]sidechaincompress=threshold=0.02:ratio=12:attack=10:release=600[md];"   # music dips while the voice speaks
            "[2:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=0.7[sx];"
            "[v1][md][sx]amix=inputs=3:duration=first:dropout_transition=0{NORM}[a]")
    last = None
    for norm, boost in ((":normalize=0", ""), ("", ",volume=2.2")):
        try:
            fc = base.replace("{NORM}", norm).replace("[a]", f"{boost}[a]") if boost else base.replace("{NORM}", norm)
            run(["ffmpeg", "-y", "-i", str(video), "-i", str(music_wav), "-i", str(sfx_wav),
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
        make_image(f"{script.get('character', '')}. {sc['image_prompt']}".strip(), raw, i)
        make_voice(sc["narration"], mp3)
        layers = prepare_scene(raw, i, sc["narration"], audio_len(mp3))
        durs.append(scene_clip(layers, mp3, clip, i)); clips.append(clip)
    print("joining scenes with transitions + cinematic look...")
    joined = OUT / "joined.mp4"
    offsets, total = join_clips(clips, durs, joined)
    print(f"adding music ({mood}) and sound effects...")
    add_sound(joined, total, offsets, mood, OUT / "final.mp4")
    return OUT / "final.mp4"

ensure_hindi_font()
print("1/3 writing script...")
script = write_script()
(OUT / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
print("2/3 building scenes...")
final = build_video(script)
print("3/3 DONE ->", final)

try:   # in Colab: show and download; on GitHub: the workflow uploads out/final.mp4 for you
    from IPython.display import Video, display
    from google.colab import files
    display(Video(str(final), embed=True, width=300))
    files.download(str(final))
except Exception:
    pass

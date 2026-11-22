// ===== Cloudflare Worker: lets anyone use your AI Shorts site WITHOUT a GitHub token =====
// Deploy: Cloudflare dashboard -> Workers & Pages -> Create -> Worker -> paste this -> Deploy
// Then Settings -> Variables and Secrets:
//   GH_TOKEN      (Secret)  your GitHub classic token (repo + workflow)
//   GH_OWNER      (Text)    saurabh-maker
//   GH_REPO       (Text)    ai-video
//   GH_BRANCH     (Text)    main            (optional)
//   APP_PASSWORD  (Secret)  any password    (optional: if set, the site asks for it before making a video)
//   ALLOW_ORIGIN  (Text)    https://saurabh-maker.github.io   (optional: only your site may call this worker)

const WORKFLOW = "make-video.yml";

export default {
  async fetch(request, env) {
    const cors = {
      "Access-Control-Allow-Origin": env.ALLOW_ORIGIN || "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };
    if (request.method === "OPTIONS") return new Response(null, { headers: cors });

    const json = (data, status = 200) =>
      new Response(JSON.stringify(data), { status, headers: { ...cors, "Content-Type": "application/json" } });

    const gh = async (path, opts = {}) => {
      const r = await fetch(`https://api.github.com/repos/${env.GH_OWNER}/${env.GH_REPO}${path}`, {
        ...opts,
        headers: {
          "Authorization": `Bearer ${env.GH_TOKEN}`,
          "Accept": "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": "ai-shorts-worker",
          ...(opts.body ? { "Content-Type": "application/json" } : {}),
        },
      });
      if (r.status === 204) return null;
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(`GitHub said ${r.status}: ${data.message || r.statusText}`);
      return data;
    };

    const slimRun = x => ({ id: x.id, status: x.status, conclusion: x.conclusion, html_url: x.html_url, created_at: x.created_at });

    try {
      const url = new URL(request.url);

      // site settings (does it need a password?)
      if (url.pathname === "/config") return json({ needsPassword: !!env.APP_PASSWORD });

      // last 5 video jobs
      if (url.pathname === "/runs") {
        const d = await gh(`/actions/workflows/${WORKFLOW}/runs?per_page=5`);
        return json({ workflow_runs: (d.workflow_runs || []).map(slimRun) });
      }

      // one video job
      if (url.pathname === "/run") {
        const id = (url.searchParams.get("id") || "").replace(/\D/g, "");
        if (!id) return json({ message: "missing id" }, 400);
        return json(slimRun(await gh(`/actions/runs/${id}`)));
      }

      // finished videos (GitHub Releases named video-<run id>)
      if (url.pathname === "/videos") {
        const d = await gh(`/releases?per_page=30`);
        return json(d.filter(x => (x.tag_name || "").startsWith("video-")).map(x => ({
          tag: x.tag_name, name: x.name, date: x.published_at, body: x.body,
          url: ((x.assets || []).find(a => a.name.endsWith(".mp4")) || {}).browser_download_url || null,
        })));
      }

      // start a new video
      if (url.pathname === "/make" && request.method === "POST") {
        const b = await request.json().catch(() => ({}));
        if (env.APP_PASSWORD && b.password !== env.APP_PASSWORD) return json({ message: "Wrong password" }, 401);
        const pick = (v, ok, def) => (ok.includes(v) ? v : def);
        const inputs = {
          topic: String(b.topic || "").slice(0, 200),
          lang: pick(b.lang, ["hi", "en"], "hi"),
          voice: pick(b.voice, ["f", "m"], "f"),
          style: pick(b.style, ["real", "cartoon"], "real"),
          youtube: pick(b.youtube, ["no", "yes"], "no"),
          yt_privacy: pick(b.yt_privacy, ["private", "unlisted", "public"], "private"),
        };
        await gh(`/actions/workflows/${WORKFLOW}/dispatches`, {
          method: "POST", body: JSON.stringify({ ref: env.GH_BRANCH || "main", inputs }),
        });
        return json({ ok: true });
      }

      return json({ message: "not found" }, 404);
    } catch (e) {
      return json({ message: e.message }, 500);
    }
  },
};

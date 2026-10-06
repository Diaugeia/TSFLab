"""Proxy between Codex and vLLM's Responses API.

vLLM 0.20.2 validates `input` items against a narrower schema than Codex sends, so
the proxy rewrites each request before forwarding it and streams the response back
unchanged. With --log DIR it also stores each request body and any error response.

    python responses_proxy.py <listen-port> <upstream-url> [--log DIR]
"""

import json
import sys
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

UP = sys.argv[2].rstrip("/")
LOG = Path(sys.argv[sys.argv.index("--log") + 1]) if "--log" in sys.argv else None
if LOG:
    LOG.mkdir(parents=True, exist_ok=True)


def rewrite(body: dict) -> dict:
    """Make a Codex request acceptable to vLLM 0.20.2.

    - `reasoning` items (the model's earlier reasoning, echoed back by Codex) fail
      vLLM's input validation; they are dropped, as a stateless API would.
    - The `developer` role is mapped to `system` (Qwen's chat template knows only system).
    """
    inp = body.get("input")
    if isinstance(inp, list):
        out = []
        for it in inp:
            if not isinstance(it, dict):
                out.append(it)
                continue
            if it.get("type") == "reasoning":
                continue
            if it.get("role") == "developer":
                it = dict(it, role="system")
            out.append(it)
        body["input"] = out
    return body


async def handle(req: web.Request) -> web.StreamResponse:
    data = await req.read()
    stamp = f"{time.time():.3f}"
    if req.method == "POST" and req.path.endswith("/responses") and data:
        try:
            body = json.loads(data)
            if LOG:
                (LOG / f"{stamp}.in.json").write_text(json.dumps(body, indent=1))
            body = rewrite(body)
            data = json.dumps(body).encode()
        except ValueError:
            pass
    headers = {k: v for k, v in req.headers.items() if k.lower() not in ("host", "content-length")}
    async with ClientSession(timeout=ClientTimeout(total=None)) as s:
        async with s.request(req.method, UP + req.path_qs, data=data, headers=headers) as r:
            skip = ("content-length", "transfer-encoding", "content-encoding")
            resp = web.StreamResponse(status=r.status, headers={k: v for k, v in r.headers.items() if k.lower() not in skip})
            await resp.prepare(req)
            err = b""
            async for chunk in r.content.iter_any():
                if r.status >= 400:
                    err += chunk
                await resp.write(chunk)
            if LOG and err:
                (LOG / f"{stamp}.err.txt").write_bytes(err)
            await resp.write_eof()
            return resp


app = web.Application(client_max_size=256 * 1024 * 1024)
app.router.add_route("*", "/{tail:.*}", handle)
web.run_app(app, host="127.0.0.1", port=int(sys.argv[1]))

"""Proxy in front of vLLM's Anthropic /v1/messages endpoint.

vLLM 0.20 accepts only user and assistant roles in `messages`, while Claude Code also
sends system-role messages; the proxy moves them into the top-level `system` field and
streams responses through unchanged.

    python anthropic_proxy.py <listen-port> <upstream-url>
"""

import json
import sys

from aiohttp import ClientSession, ClientTimeout, web

UP = sys.argv[2].rstrip("/")


def fix(body: dict) -> dict:
    msgs = body.get("messages") or []
    extra = [m for m in msgs if m.get("role") == "system"]
    if extra:
        body["messages"] = [m for m in msgs if m.get("role") != "system"]
        blocks = body.get("system") or []
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        for m in extra:
            c = m.get("content")
            if isinstance(c, str):
                blocks.append({"type": "text", "text": c})
            else:
                blocks += [b for b in c or [] if b.get("type") == "text"]
        body["system"] = blocks
    return body


async def handle(req: web.Request) -> web.StreamResponse:
    data = await req.read()
    if req.method == "POST" and req.path.startswith("/v1/messages") and data:
        try:
            data = json.dumps(fix(json.loads(data))).encode()
        except ValueError:
            pass
    headers = {k: v for k, v in req.headers.items() if k.lower() not in ("host", "content-length")}
    async with ClientSession(timeout=ClientTimeout(total=None)) as s:
        async with s.request(req.method, UP + req.path_qs, data=data, headers=headers) as r:
            skip = ("content-length", "transfer-encoding", "content-encoding")
            resp = web.StreamResponse(status=r.status, headers={k: v for k, v in r.headers.items() if k.lower() not in skip})
            await resp.prepare(req)
            async for chunk in r.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp


app = web.Application(client_max_size=64 * 1024 * 1024)
app.router.add_route("*", "/{tail:.*}", handle)
web.run_app(app, host="127.0.0.1", port=int(sys.argv[1]))

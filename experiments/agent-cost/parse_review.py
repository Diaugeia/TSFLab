"""Parse a reviewer's raw reply into JSON: parse_review.py <review.X.raw> <review.X.json>.

Tolerates text around the JSON and a malformed tail (for example an extra closing
brace): the first decodable object is used, and `valid` is recovered by pattern if
the object lost it.
"""

import json
import os
import re
import sys

raw = open(sys.argv[1]).read() if os.path.exists(sys.argv[1]) else ""
out = None
i = raw.find("{")
if i >= 0:
    try:
        out, _ = json.JSONDecoder().raw_decode(raw[i:])
    except ValueError:
        out = None
if not isinstance(out, dict):
    out = {"error": "no JSON", "raw": raw[-2000:]}
if "valid" not in out:
    m = re.findall(r'"valid"\s*:\s*(true|false)', raw)
    if m:
        out["valid"] = m[-1] == "true"
        out["note"] = "valid recovered from malformed JSON"
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(sys.argv[2], "valid=", out.get("valid"))

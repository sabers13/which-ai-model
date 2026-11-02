# Record and replay

Every live model response is stored in `recordings/` the moment it
arrives, keyed so a later run can serve the identical bytes with no
network. Replay is the default for everything run through `make`: the
Makefile exports `WAM_MODE=replay`, and only the single live benchmark
target overrides it. So `make all` rebuilds results byte-identically on a
machine with no key and no network, and every page number traces back to
committed files rather than to a call nobody can reproduce. The rest of
this page shows how the key is derived, then the three rules for when to
record, when to replay, and when a miss means re-recording. The key idea
fits in one sentence: the key names the exact request, so identical keys
mean identical requests, and anything else is a new recording.

## How the recording key is built

The key binds the response to exactly the request that produced it. First
a request digest is computed as the sha256 of the canonical JSON of the
request messages — sorted keys, no whitespace — so any prompt or input
change alters the digest. Then the recording key is the sha256 of the
model id, the prompt version, and that digest joined with newlines. One
JSONL row is stored per call: key, model id, prompt version, digest, a
response object with content, tool calls, and scrubbed provider usage, and
the UTC timestamp. Usage may carry the gateway's measured cost under a
provider key; it never carries headers, keys, or credential URLs:

```python
from which_ai_model.core import Recorder, recording_key, request_digest

digest = request_digest(messages)
key = recording_key("deepseek-v4.1-flash", "route-v1", digest)
Recorder().record(key, "deepseek-v4.1-flash", "route-v1", digest, response)
found = Recorder().lookup(key)  # the response dict, or None on a miss
```

## When to record, when to replay, and when to re-record

Record during the one live run and during explicitly live debugging;
replay everywhere else, including all tests, where recorded fixtures stand
in for the network. A replay miss is a signal, not an error to paper over:
it means the prompt, model, or input changed without a matching recording,
and the fix is to re-record deliberately — never to fall back to a live
call inside a replay-mode run. Because the key includes the prompt
version, bumping a prompt automatically invalidates old recordings for it,
which is exactly what you want before a gate comparison.

Parity checks for the migrated tools run entirely in replay with a fake
model returning the same raw content as the legacy fixture, proving the
new tool behaves like the old one for free. The regression gate compares
two replay runs, so its verdict is deterministic: same recordings, same
scores, same decision every time, on every machine that checks out the
same commit.

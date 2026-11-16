# cancho-studio: editing a picture given by URL (design, 2026-11-16)

> **Status: design; nothing is built.** The owner chose (2026-11-16) that the **server downloads** the picture, with protections, rather than the browser fetching it and uploading it. Claims are tagged **[read]** (source or document, not run), **[measured]** (run, with the command) or **[estimate]**. Cancho native: no Python, no Rust, no OpenSSL through FFI.

## 1. Goal and scope

`POST /edit` (`studio/upload.cho`) takes a picture as a multipart part `image`. It would also take a part `url`: an `http://` or `https://` address of a JPEG, PNG or WebP picture. The server downloads it, keeps it as `<data>/upload-N.<ext>` exactly as an uploaded one (the extension from the first bytes, never from the URL or the `Content-Type`), and makes the job as today (`jobs.create`). Exactly one of `image` and `url` must be there.

**Not in scope:** a video by URL (no pipeline takes a video as input: `media edit` takes an image, `media video` only text; image-to-video is #174), a URL for the *results* (they are served only on the loopback, behind the token), redirects (section 4, decision R1), IPv6.

## 2. What the server is today, and what changes

**[measured]** `cancho authority studio/studio.cho studio/files.cho studio/jobs.cho studio/results.cho studio/auth.cho studio/range.cho studio/upload.cho .../http-server/server.cho --std` (2026-11-16) says `UNBOUNDED: this program holds the network without narrowing it`, with `net_in("")` among the rows: the server's inbound authority is **every port**, not `127.0.0.1`. **Correction in place:** the comment in `studio/studio.cho` said "the authority row names the host"; it does not. The bind to `127.0.0.1` is the code's (`tcp_listen_on(nn, "127.0.0.1", ...)`, checked with `lsof` and by the Host/Origin/token checks of docs/studio.md), not the report's. The report does not show it because the host is a run-time argument of the call and `narrow` takes a literal (`docs/net-bounds.md`).

**What this feature adds, stated plainly:** an outbound connection whose destination is data (a URL a person typed), so the report gains `net_out("")`, **every host and port**, plus `file_read` of the trust store. No narrowing can express "any public address on port 80 or 443"; the bound is therefore the code's, and section 4 is that code's specification. The report will say "unbounded" about the network, as it already does about the inbound side, and the document must not pretend otherwise.

## 3. What cancho has for it

* **[read]** `packages/http-client` (`docs/http-client.md`, built): a non-blocking HTTP/1.1 client that performs no I/O of its own (the caller owns sockets, TLS engine and poller); chunked bodies, status and headers parsed from the head, tickets per request.
* **[read]** `packages/tls` and `packages/x509` (`docs/tls-core.md`, `tls-parity.md`, `x509-verify.md`, built): a TLS 1.3 and 1.2 client in cancho, certificate chains verified against trust anchors. **Its own documents say nothing built on it may be called production-ready before the independent review of #209.** `cancho-hooks` uses it for `https` delivery (`docs/tls-hooks.md` section 9: 97 checks on both backends; a handshake costs 4 to 7 times the CPU of OpenSSL's).
* **[read]** `examples/tls_nb/dns.cho` (a DNS A-record query builder and a total answer parser, fuzzed with a million damaged answers), `rtcp.cho` (a resolver on the poller over TCP, no thread, no libc; **no `/etc/hosts`, no search list, one name server**) and `pin.cho` (`is_public`: the ranges that are not for the public internet; `choose`: connect to the address that was checked, as a literal, so there is no second lookup for a changed answer to win).
* **[read]** `std.conns` (`tcp_connect_start`: no lookup for an IP literal).
* **[measured]** The system's trust store is `/etc/ssl/cert.pem` on this machine: 128 certificates (`grep -c "BEGIN CERTIFICATE"`). cancho has no environment access (`tls-hooks.md` section 9), so the path is an argument.
* **Not yet checked:** that these pieces build into one program with `http.server` and run on one poller in this checkout's pinned compiler (a49ea779); that `/etc/resolv.conf` on macOS names a resolver reachable over TCP.

## 4. The protections (each one a check with a test that can fail)

| # | Threat | Protection |
|---|---|---|
| P1 | Other schemes (`file:`, `ftp:`, `gopher:`, `data:`), userinfo (`http://user:pass@host`), fragments of confusion | Only `http://` and `https://`; a `@` before the first `/` is refused; the URL is 1 to 2,000 bytes of printable ASCII, no spaces, no control characters; anything else is refused with a reason before any connection |
| P2 | Hosts spelled so a parser reads one address and a library another (`http://2130706433/`, `http://0x7f.1/`, `http://127.1/`, `http://[::1]/`) | **One accepted form for an address: four decimal octets, each 0 to 255 with no leading zero.** A host made only of digits and dots that is not that form is refused; a `[` is refused (no IPv6). Any other host must be a valid name (labels of letters, digits and hyphens, 253 bytes at most) |
| P3 | A name that points at the inside (`localhost`, a name resolving to 10.0.0.5, cloud metadata `169.254.169.254`) | The name is resolved by the program (`rtcp`); **every** A record is judged by `is_public`; **one** non-public address refuses the whole answer; `localhost` and names with no dot are refused before resolving |
| P4 | DNS rebinding (the name answers a public address to the check and a private one to the connection) | The connection is made to the **address that was checked, as a literal**; there is no second lookup; the name is used afterwards only for the `Host` header, SNI and the certificate check |
| P5 | Redirects to the inside | **R1: a 3xx is not followed**: the fetch fails with the status and the `Location` host only. (A later slice may follow up to 3 redirects, each through P1 to P4 again and never from `https` to `http`; it is not part of the first version.) |
| P6 | Port scanning and other services on a public host | Ports **80 and 443 only**, and the port in the URL, if any, must be the scheme's own |
| P7 | A huge or endless body | `Content-Length` over 15 MiB is refused before reading a byte; the bytes counted while reading cannot pass 15 MiB (a body with no length is cut and refused); `Accept-Encoding: identity` is sent and a response with a `Content-Encoding` is refused (no decompressor to bomb); the whole fetch has a deadline (connect 10 s, total 30 s) |
| P8 | Slow or many fetches holding the server | One loop, one poller: a fetch is a state machine advanced on events, never a blocking call; **at most 2 fetches at once**, a third is a 503 with `Retry-After`; the held request's ticket is answered on every ending |
| P9 | A server that lies about the content | The bytes are judged by their first bytes (`kind_of`: JPEG, PNG, WebP) exactly as an upload is; the URL's extension and the `Content-Type` are not read for that |
| P10 | Leaking what the studio knows | The request carries no cookie, no `Authorization`, nothing of the token, no `Referer`, no prompt; `User-Agent: cancho-studio`; the log line has the host and the outcome, **not the path or the query** (they can hold a secret) |
| P11 | TLS that does not verify | `https` verifies the chain against the trust store and the host name; a failure is refused with the stage told apart (as `tls_nb` does); there is **no option to skip verification** |
| P12 | Using the feature against the author's own tests | A flag `--allow-private` (off by default, and said in the first line the server prints) lets the checks use a stub on the loopback; it is the only way to reach a non-public address |

What the table does **not** claim: that a public address is a safe one (a public server can still be hostile: the bytes are only ever read, size-bounded and kept as a file the edit pipeline opens, as an upload is), that the pure TLS has been independently reviewed (it has not; see section 3), or that the checks make the report bounded (it will say `net_out("")`).

## 5. How it fits the single-threaded server

`studio.cho`'s loop answers a request in the turn it arrives. A fetch takes many turns, so the request is **held** (`server.hold`) and answered later with `server.answer`, as the file streams are (docs/studio.md, "Big files and `Range`"). A fetch is a slot in a small table (as `streams` is): its ticket, its phase (resolving, connecting, handshaking, sending, reading, done), its deadline, the bytes so far in a buffer capped at 15 MiB. On every turn each live slot is advanced by the events the poller reported for its tokens. When the body is complete and judged, `upload.cho`'s job creation runs with the bytes (the same function `handle` uses after it has the picture) and the held request is answered `302 /jobs/N`; on any failure it is answered with the status and one line, as `/edit` does.

## 6. Slices, each with a gate that can fail

| slice | what | held to |
|---|---|---|
| U0 | this document, and the comment in `studio/studio.cho` corrected | review |
| U1 | `studio/fetch_policy.cho`: P1, P2, P6 and the range test (`is_public`, copied with its source named, as `pin.cho` does) as pure functions, and `studio/fetch_check.sh`'s table of adversarial URLs | a corpus of at least 60 URLs (accepted and refused, each with the reason) written before the code; mutants of every rule |
| U2 | the fetch state machine over plain `http` to an address literal, against a stub on the loopback (`--allow-private`), with P7, P8, P5(R1), P9, P10 | the stub serves a short body, a body over the cap, a lying `Content-Length`, a 3xx, a server that never answers, one that sends bytes forever; each refused as designed |
| U3 | names: the resolver and the pin (P3, P4) | a stub name server: private answer, mixed answer, answer that changes between two lookups, NXDOMAIN, a damaged answer |
| U4 | `https` with the trust store (P11) | a TLS stub with a good chain, an expired one, a wrong name, a self-signed one; each refused with its stage |
| U5 | the form field, `/edit`'s `url` part, the OpenAPI document, `docs/studio.md` | the whole flow in a browser once, as the upload was |

## 7. Not decided

* **R1 (no redirects).** Many image URLs redirect (a CDN, an `http` to `https` upgrade). Refusing them is the safe first version; following up to 3 with the same checks is the likely second.
* **Which name server.** `rtcp` takes one. The proposal is to read the first `nameserver` of `/etc/resolv.conf` (a file read: an argument names it, since cancho has no environment); whether macOS's file is reachable over TCP is **to be measured in U3**.
* **Whether to accept the pure TLS before #209's review.** The alternative the owner has already declined is OpenSSL through FFI. Proposed: build it on the pure TLS, say in this document and in the server's first line that the TLS has not been independently reviewed, and keep `https` off unless the server is started with a trust-store path.

# Recording the Godspeed demo (asciinema → GIF)

The README hero showcases a ~45-60 s animated demo of a realistic
Godspeed task. This doc captures the instructions for re-recording
it so the demo can stay fresh across releases.

## Tools

```bash
# Ubuntu / WSL (recommended: recordings are cleaner on a real terminal)
sudo apt install asciinema
cargo install --git https://github.com/asciinema/agg    # converter to GIF

# macOS
brew install asciinema
brew install --HEAD asciinema-agg                       # or cargo install --git ...
```

## Record

```bash
# From the repo root:
cd ~/Documents/Project\ Portfolio/godspeed

asciinema rec docs/demo.cast \
    --title "Godspeed v3.3 — 60-second demo" \
    --idle-time-limit 2 \
    --command "bash -c 'cd /tmp && mkdir -p godspeed_demo && cd godspeed_demo && godspeed'"
```

Inside the recording session, type a realistic task. Suggested prompt
(mirrors benchmark T2 which finishes in ~50 s cleanly):

```
Create utils.py with uppercase and count_vowels functions, plus a test
for each in tests/test_utils.py. Use type hints.
```

Let the agent complete. Press Ctrl+D or type `/quit` to exit. The
recording saves to `docs/demo.cast`.

## Convert to GIF

```bash
agg --theme monokai --font-size 14 --cols 120 --rows 30 \
    docs/demo.cast docs/demo.gif
```

Target size: ≤ 1.5 MB (README embed stays responsive). If the GIF is
larger, trim:

- Re-record with `--idle-time-limit 1`
- Pre-trim with `asciinema-edit cut docs/demo.cast --from 0 --to 60`
  (requires the `asciinema-edit` tool)

## Embed in README

```markdown
<p align="center">
  <img src="docs/demo.gif" alt="Godspeed v3.3 demo — add a function, add tests, ~50 seconds." width="780" />
</p>
```

## Alternative: asciinema player embed

If the GIF gets too large, embed the interactive player instead:

```markdown
<a href="https://asciinema.org/a/<ID>" target="_blank">
  <img src="https://asciinema.org/a/<ID>.svg" alt="Godspeed v3.3 demo" />
</a>
```

Upload via `asciinema upload docs/demo.cast` → you get a shareable URL.

## Checklist before committing the GIF

- [ ] Recording is 45-75 s (too short feels demoey; too long loses
      attention)
- [ ] Shows ONE realistic task, not a kitchen-sink demo
- [ ] Task actually completes successfully (not a demo of errors)
- [ ] No secrets / personal paths visible (`$HOME`, API keys in env)
- [ ] Terminal dimensions are readable at embedded size
      (cols=120, font-size=14 is a good default)
- [ ] File size ≤ 1.5 MB
- [ ] README hero block updated to reference the new GIF path

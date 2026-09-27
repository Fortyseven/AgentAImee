# Example workspace

This is the Aimee example workspace. A few ground rules:

- `secondary/` is a second workspace root (multi-root demo). It holds
  `secret-note.txt` — read it when asked.
- Safety: never run destructive shell commands (anything containing `rm `).
  The client in `custom_tool.py` also enforces this via an approval hook.
- Be concise.

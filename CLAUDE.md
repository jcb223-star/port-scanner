# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This git repository is rooted at the user's home directory (`/home/kali`). The `.gitignore` excludes everything (`*`) except explicitly allow-listed files (`!port_scanner.py`, `!.gitignore`), so only those two files are tracked — the rest of the home directory (Desktop, Documents, other project folders, dotfiles, etc.) is local clutter, not part of this project. When making changes, only touch files that are actually tracked or explicitly relevant to the task at hand; do not assume other files/directories in the home directory are part of this codebase.

If you add new source files to this project, remember they must be explicitly un-ignored in `.gitignore` (e.g. `!newfile.py`) or `git add` will refuse to stage them.

## Code architecture

The repository currently consists of a single script:

- `port_scanner.py` — a standalone, multithreaded TCP port scanner (stdlib only, no third-party dependencies). It resolves the target host, parses a port spec (comma-separated ports and/or ranges, e.g. `22,80,1-1024`), then uses a `ThreadPoolExecutor` to attempt a connection to each port concurrently, reporting open ports and their best-guess service name (via `socket.getservbyport`).

This tool is for authorized security testing only (scanning hosts you own or have explicit permission to test), per the docstring at the top of the file.

## Commands

Run the scanner directly with Python 3 (no build step, no install required):

```bash
python3 port_scanner.py <target> [-p PORTS] [-t THREADS] [-w TIMEOUT]
```

Examples:
```bash
python3 port_scanner.py 192.168.1.10                 # scan default ports 1-1024
python3 port_scanner.py example.com -p 22,80,443      # scan specific ports
python3 port_scanner.py 10.0.0.5 -p 1-65535 -t 500 -w 0.5   # full scan, more threads, shorter timeout
```

Run tests with:

```bash
python3 -m pytest test_port_scanner.py -v
```

`scan_port` tests mock `socket.socket`/`socket.getservbyport` so they run without real network access. There is no linter or build configuration in this repository.

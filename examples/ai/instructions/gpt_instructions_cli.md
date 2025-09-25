# MainSequence CLI — Usage Guide

> Run every command via the module entrypoint:  
> `python -m mainsequence …`

This CLI helps you sign in, view and set up projects locally (with SSH keys generated & added automatically), open mapped folders, and optionally build/run containers from a project.

---

## Quick start

```bash
# 1) Sign in (prompts for password)
python -m mainsequence login you@example.com

# 2) See your current settings
python -m mainsequence settings

# 3) List projects you have access to
python -m mainsequence project list

# 4) Clone & link a project locally (creates ~/.ssh/<repo-key>, starts ssh-agent, writes .env)
python -m mainsequence project set-up-locally 123

# 5) Open the mapped folder in your file manager
python -m mainsequence project open 123

# 6) Open a terminal pre-wired with ssh-agent & the project’s key
python -m mainsequence project open-signed-terminal 123
```

---

## Prerequisites

- **Python** and **git** available on PATH.
- **OpenSSH**/**ssh-agent** available (the CLI will try to start it and add keys).
- For clipboard copy of the public key: `pbcopy` (macOS) or `wl-copy`/`xclip` (Linux); optional.
- For `build_and_run`:  
  - **uv** (`pip install uv`)  
  - **Docker** (if you use the Docker part).

---

## Configuration & files

- **Config directory**
  - **Windows:** `%APPDATA%\MainSequenceCLI`
  - **macOS:** `~/Library/Application Support/MainSequenceCLI`
  - **Linux:** `~/.config/mainsequence`
- **Files**
  - `config.json` — stores `backend_url` and `mainsequence_path`
  - `token.json` — stores `{ username, access, refresh, ts }`
  - `project-links.json` — maps project IDs to absolute local paths

**Defaults**
- Backend: `https://main-sequence.app/` (override with `MAIN_SEQUENCE_BACKEND_URL`)
- Projects base: `~/mainsequence`

---

## Environment variables

- `MAIN_SEQUENCE_BACKEND_URL` – set before running the CLI to point at a different backend.
- `MAIN_SEQUENCE_USER_TOKEN` – access token; `login --export` prints a shell-ready `export …` line.

---

## Top-level commands

### `login`
Obtain and store tokens, set `MAIN_SEQUENCE_USER_TOKEN` for the current process, and (by default) print a projects table.

```bash
python -m mainsequence login <email>
# options:
#   --export      Print: export MAIN_SEQUENCE_USER_TOKEN="…"
#   --no-status   Do not print the projects table after login
```

**Examples**
```bash
# POSIX shells: set token env var for the current shell
eval "$(python -m mainsequence login you@example.com --export)"
```

> On Windows PowerShell, run login normally (it prints a POSIX `export …` line). Then set the token manually:  
> `$env:MAIN_SEQUENCE_USER_TOKEN = "<paste token>"`

---

### `settings`
Show or change local settings.

```bash
# `settings` with no subcommand == `settings show`
python -m mainsequence settings
python -m mainsequence settings show

# Change the projects base folder (created if missing)
python -m mainsequence settings set-base "D:\Work\mainsequence"         # Windows
python -m mainsequence settings set-base ~/work/mainsequence             # macOS/Linux
```

Output of `show` is JSON, e.g.:
```json
{
  "backend_url": "https://main-sequence.app/",
  "mainsequence_path": "/home/alex/mainsequence"
}
```

---

### `project` (requires login)

#### `project list`
List projects with local mapping status and guessed path.

```bash
python -m mainsequence project list
```

Output columns:
- **ID, Project, Data Source, Class, Status, Local, Path**  
`Local` is **Local** if a mapping exists or a default folder is present; otherwise `—`.

#### `project set-up-locally <project_id>`
End-to-end local setup:
1. Figures out the repository SSH URL (from `git_ssh_url` or nested data source hints).
2. Creates a dedicated SSH key in `~/.ssh/<repo-name>` (if missing).
3. Tries to register the public key as a **deploy key** on the server.
4. Starts `ssh-agent` and adds that key.
5. Clones into:  
   `<base>/<org-slug>/projects/<safe-project-name>`
6. Fetches environment text from the backend and writes `.env` (ensuring `VFB_PROJECT_PATH=<local-path>`).

```bash
python -m mainsequence project set-up-locally 123
# optional: override base dir for this one command
python -m mainsequence project set-up-locally 123 --base-dir ~/my-projects
```

**Exit codes**
- `1` – project not found/visible, or repo URL missing
- `2` – target folder already exists
- `3` – `git clone` failed

#### `project open <project_id>`
Opens the mapped/guessed folder in your OS file manager.

```bash
python -m mainsequence project open 123
```

If no mapping exists yet, the CLI tries the default path:
`<base>/<org-slug>/projects/<safe-project-name>`.

#### `project delete-local <project_id> [--permanent]`
Unlink the local mapping; optionally **delete** the folder.

```bash
# just unlink the mapping (keeps folder)
python -m mainsequence project delete-local 123

# unlink and remove the folder (dangerous)
python -m mainsequence project delete-local 123 --permanent
```

#### `project open-signed-terminal <project_id>`
Opens a new terminal window **in the project directory** with `ssh-agent` started and the repo key added.  
- **Windows:** PowerShell window; tries to enable the `ssh-agent` service.  
- **macOS:** opens **Terminal.app** with a ready shell.  
- **Linux:** opens the first available emulator (e.g. `gnome-terminal`, `konsole`, etc.).

```bash
python -m mainsequence project open-signed-terminal 123
```

---

### `build_and_run [DOCKERFILE]`
Project packaging helper:

1) `uv lock`  
2) `uv export --format requirements --no-dev --hashes > requirements.txt`  
3) If `DOCKERFILE` provided: `docker build …` then `docker run -it` the image.

```bash
# Only lock & export requirements.txt (no Docker)
python -m mainsequence build_and_run

# Build & run with a specific Dockerfile
python -m mainsequence build_and_run ./Dockerfile
```

**Requirements & behavior**
- Fails early if `uv` is missing or `pyproject.toml` isn’t present.
- Image name defaults to `<cwd-name>-img` (override with `IMAGE_NAME`).
- Tag defaults to short `git` SHA, else a timestamp (override with `TAG`).

---

## Typical workflow

```bash
# sign in
python -m mainsequence login you@example.com

# pick a project to set up
python -m mainsequence project list

# set it up locally (creates SSH key, clones, writes .env, links the folder)
python -m mainsequence project set-up-locally 456

# open the folder to browse files
python -m mainsequence project open 456

# open a ready-to-use terminal for git operations over SSH
python -m mainsequence project open-signed-terminal 456
```

---

## Troubleshooting

- **“Not logged in.”**  
  Run `python -m mainsequence login <email>` again. Tokens live in `token.json`.

- **Token refresh / 401**  
  The CLI will auto-refresh once. If it still fails, re-login.

- **No project repo URL found**  
  The CLI looks at `git_ssh_url` then digs into `data_source.related_resource.extra_arguments`.  
  If your project truly lacks a repo, ask an admin to attach one.

- **`git clone` failed**  
  Likely SSH access. The CLI generates `~/.ssh/<repo-name>` and tries to add it as a deploy key.  
  You may need to add that **public key** to your Git host manually. Re-run after access is granted.

- **Clipboard didn’t copy the public key**  
  Install `pbcopy` (macOS) or `wl-copy`/`xclip` (Linux), or open `~/.ssh/<repo-name>.pub` and copy manually.

- **Linux: “No terminal emulator found”** when opening a signed terminal  
  Install one of: `x-terminal-emulator`, `gnome-terminal`, `konsole`, `xfce4-terminal`, `tilix`, `mate-terminal`, `alacritty`, `kitty`, or `xterm`.

- **Change backend URL**  
  Run with `MAIN_SEQUENCE_BACKEND_URL` set, e.g.  
  `export MAIN_SEQUENCE_BACKEND_URL="https://staging.main-sequence.app"`  
  (or edit `config.json` directly).

---

## Help

```bash
python -m mainsequence --help
python -m mainsequence login --help
python -m mainsequence settings --help
python -m mainsequence project --help
python -m mainsequence build_and_run --help
```

# mainsequence/cli/cli.py
from __future__ import annotations

import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
from typing import Optional

import typer

from . import config as cfg
from .api import (
    ApiError,
    add_deploy_key,
    deep_find_repo_url,
    fetch_project_env_text,
    get_current_user_profile,
    get_projects,
    login as api_login,
    repo_name_from_git_url,
    safe_slug,
)
from .ssh_utils import (
    ensure_key_for_repo,
    open_folder,
    open_signed_terminal,
    start_agent_and_add_key,
)
import time

app = typer.Typer(help="MainSequence CLI (login + project operations)")
project = typer.Typer(help="Project commands")
settings = typer.Typer(help="Settings (base folder, backend, etc.)")
app.add_typer(project, name="project")
app.add_typer(settings, name="settings")

# ---------- helpers ----------

def _projects_root(base_dir: str, org_slug: str) -> pathlib.Path:
    p = pathlib.Path(base_dir).expanduser()
    return p / org_slug / "projects"

def _determine_repo_url(p: dict) -> str:
    repo = (p.get("git_ssh_url") or "").strip()
    if repo.lower() == "none":
        repo = ""
    if not repo:
        extra = (p.get("data_source") or {}).get("related_resource", {}) or {}
        extra = extra.get("extra_arguments") or (p.get("data_source") or {}).get("extra_arguments") or {}
        repo = deep_find_repo_url(extra) or ""
    return repo

def _copy_clipboard(txt: str) -> bool:
    try:
        if sys.platform == "darwin":
            p = subprocess.run(["pbcopy"], input=txt, text=True)
            return p.returncode == 0
        elif shutil.which("wl-copy"):
            p = subprocess.run(["wl-copy"], input=txt, text=True)
            return p.returncode == 0
        elif shutil.which("xclip"):
            p = subprocess.run(["xclip", "-selection", "clipboard"], input=txt, text=True)
            return p.returncode == 0
    except Exception:
        pass
    return False

def _render_projects_table(items: list[dict], links: dict) -> str:
    """Return an aligned table with Local status + path."""
    def ds(obj, path, default=""):
        try:
            for k in path.split("."):
                obj = obj.get(k, {})
            return obj or default
        except Exception:
            return default

    rows = []
    for p in items:
        pid   = str(p.get("id", ""))
        name  = p.get("project_name") or "(unnamed)"
        dname = ds(p, "data_source.related_resource.display_name", "")
        klass = ds(p, "data_source.related_resource.class_type",
                   ds(p, "data_source.related_resource_class_type", ""))
        status = ds(p, "data_source.related_resource.status", "")
        local_path = links.get(pid, "")
        local_ok = bool(local_path and pathlib.Path(local_path).exists())
        local = "Local" if local_ok else "—"
        path_col = local_path if local_ok else "—"
        rows.append((pid, name, dname, klass, status, local, path_col))

    header = ["ID","Project","Data Source","Class","Status","Local","Path"]
    if not rows:
        return "No projects."

    # compute widths
    colw = [max(len(r[i]) for r in rows + [tuple(header)]) for i in range(len(header))]
    fmt = "  ".join("{:<" + str(colw[i]) + "}" for i in range(len(header)))
    out = [fmt.format(*header), fmt.format(*["-"*len(h) for h in header])]
    for r in rows:
        out.append(fmt.format(*r))
    return "\n".join(out)

# ---------- commands ----------

@app.command()
def login(
    email: str = typer.Argument(..., help="Email/username (server expects 'email' field)"),
    password: Optional[str] = typer.Option(None, prompt=True, hide_input=True, help="Password"),
    export: bool = typer.Option(False, "--export", help='Print `export MAIN_SEQUENCE_USER_TOKEN=...` so you can eval it'),
    no_status: bool = typer.Option(False, "--no-status", help="Do not print projects table after login")
):
    """
    Obtain tokens (same as Electron), store them locally, and set MAIN_SEQUENCE_USER_TOKEN
    in the current process. On success, shows the base folder and a Local/Path status
    table like the Electron app.
    """
    try:
        res = api_login(email, password)
    except ApiError as e:
        # <— no traceback, just the server message
        typer.secho(f"Login failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception:
        typer.secho("Login failed.", fg=typer.colors.RED)
        raise typer.Exit(1)

    # success output
    cfg_obj = cfg.get_config()
    base = cfg_obj["mainsequence_path"]
    typer.secho(f"Signed in as {res['username']} (Backend: {res['backend']})", fg=typer.colors.GREEN)
    typer.echo(f"Projects base folder: {base}")

    # optional export line
    tok = cfg.get_tokens().get("access", os.environ.get("MAIN_SEQUENCE_USER_TOKEN", ""))
    if export and tok:
        print(f'export MAIN_SEQUENCE_USER_TOKEN="{tok}"')

    # summary table like the Electron view
    if not no_status:
        try:
            items = get_projects()
        except Exception:
            items = []
        links = cfg.get_links()
        typer.echo("\nProjects:")
        typer.echo(_render_projects_table(items, links))

@settings.command("show")
def settings_show():
    """Show settings (backend + base folder)."""
    c = cfg.get_config()
    typer.echo(json.dumps({
        "backend_url": c.get("backend_url"),
        "mainsequence_path": c.get("mainsequence_path")
    }, indent=2))

@settings.command("set-base")
def settings_set_base(path: str = typer.Argument(..., help="New projects base folder")):
    """Change the projects base folder (like the Electron 'Change…' button)."""
    out = cfg.set_config({"mainsequence_path": path})
    typer.secho(f"Projects base folder set to: {out['mainsequence_path']}", fg=typer.colors.GREEN)

@project.command("list")
def project_list():
    """List projects with Local status and path."""
    items = get_projects()
    links = cfg.get_links()
    typer.echo(_render_projects_table(items, links))

@project.command("open")
def project_open(project_id: int):
    """Open the local folder in the OS file manager."""
    links = cfg.get_links()
    path = links.get(str(project_id))
    if not path or not pathlib.Path(path).exists():
        typer.secho("No local folder mapped for this project. Run `set-up-locally` first.", fg=typer.colors.RED)
        raise typer.Exit(1)
    open_folder(path)
    typer.echo(f"Opened: {path}")

@project.command("delete-local")
def project_delete_local(
    project_id: int,
    permanent: bool = typer.Option(False, "--permanent", help="Also remove the folder (dangerous)")
):
    """Unlink the mapped folder, optionally delete it."""
    mapped = cfg.remove_link(project_id)
    if not mapped:
        typer.echo("No mapping found.")
        return
    p = pathlib.Path(mapped)
    if p.exists():
        if permanent:
            import shutil
            shutil.rmtree(mapped, ignore_errors=True)
            typer.secho(f"Deleted: {mapped}", fg=typer.colors.YELLOW)
        else:
            typer.secho(f"Unlinked mapping (kept folder): {mapped}", fg=typer.colors.GREEN)
    else:
        typer.echo("Mapping removed; folder already absent.")

@project.command("open-signed-terminal")
def project_open_signed_terminal(project_id: int):
    """Open a terminal window in the project directory with ssh-agent started and the repo's key added."""
    links = cfg.get_links()
    dir_ = links.get(str(project_id))
    if not dir_ or not pathlib.Path(dir_).exists():
        typer.secho("No local folder mapped for this project. Run `set-up-locally` first.", fg=typer.colors.RED)
        raise typer.Exit(1)

    proc = subprocess.run(["git", "-C", dir_, "remote", "get-url", "origin"], text=True, capture_output=True)
    origin = (proc.stdout or "").strip().splitlines()[-1] if proc.returncode == 0 else ""
    name = repo_name_from_git_url(origin) or pathlib.Path(dir_).name
    key_path = pathlib.Path.home() / ".ssh" / name
    open_signed_terminal(dir_, key_path, name)

@project.command("set-up-locally")
def project_set_up_locally(
    project_id: int,
    base_dir: Optional[str] = typer.Option(None, "--base-dir", help="Override base dir (default from settings)")
):
    """
    Idempotently set up a project locally:
      - derive repo URL
      - create per-repo SSH key (~/.ssh/<repoName>)
      - start/prime ssh-agent; upload deploy key (best-effort)
      - clone into <base>/<org>/projects/<slug>
      - fetch .env and write VFB_PROJECT_PATH
      - remember mapping
    """
    cfg_obj = cfg.get_config()
    base = base_dir or cfg_obj["mainsequence_path"]

    prof = get_current_user_profile()
    org_name = prof.get("organization") or "default"
    org_slug = re.sub(r"[^a-z0-9-_]+", "-", org_name.lower()).strip("-") or "default"

    items = get_projects()
    p = next((x for x in items if int(x.get("id", -1)) == project_id), None)
    if not p:
        typer.secho("Project not found/visible.", fg=typer.colors.RED)
        raise typer.Exit(1)

    repo = _determine_repo_url(p)
    if not repo:
        typer.secho("No repository URL found for this project.", fg=typer.colors.RED)
        raise typer.Exit(1)

    name = safe_slug(p.get("project_name") or f"project-{project_id}")
    projects_root = _projects_root(base, org_slug)
    target_dir = projects_root / name
    projects_root.mkdir(parents=True, exist_ok=True)

    # key & clipboard
    key_path, pub_path, pub = ensure_key_for_repo(repo)
    copied = _copy_clipboard(pub)

    # deploy key (best-effort)
    try:
        host = platform.node()
        add_deploy_key(project_id, host, pub)
    except Exception:
        pass

    # agent + add key
    agent_env = start_agent_and_add_key(key_path)

    # clone
    if target_dir.exists():
        typer.secho(f"Target already exists: {target_dir}", fg=typer.colors.RED)
        raise typer.Exit(2)

    env = os.environ.copy() | agent_env
    env["GIT_SSH_COMMAND"] = f'ssh -i "{str(key_path)}" -o IdentitiesOnly=yes'
    rc = subprocess.call(["git", "clone", repo, str(target_dir)], env=env, cwd=str(projects_root))
    if rc != 0:
        try:
            if target_dir.exists():
                import shutil
                shutil.rmtree(target_dir, ignore_errors=True)
        except Exception:
            pass
        typer.secho("git clone failed", fg=typer.colors.RED)
        raise typer.Exit(3)

    # .env
    env_text = ""
    try:
        env_text = fetch_project_env_text(project_id)
    except Exception:
        env_text = ""

    env_text = (env_text or "").replace("\r", "")
    if any(line.startswith("VFB_PROJECT_PATH=") for line in env_text.splitlines()):
        lines = []
        for line in env_text.splitlines():
            if line.startswith("VFB_PROJECT_PATH="):
                lines.append(f"VFB_PROJECT_PATH={str(target_dir)}")
            else:
                lines.append(line)
        env_text = "\n".join(lines)
    else:
        if env_text and not env_text.endswith("\n"):
            env_text += "\n"
        env_text += f"VFB_PROJECT_PATH={str(target_dir)}\n"
    (target_dir / ".env").write_text(env_text, encoding="utf-8")

    # remember mapping
    cfg.set_link(project_id, str(target_dir))

    typer.secho(f"Local folder: {target_dir}", fg=typer.colors.GREEN)
    typer.echo(f"Repo URL: {repo}")
    if copied:
        typer.echo("Public key copied to clipboard.")

@app.command("build_and_run")
def build_and_run(dockerfile: Optional[str] = typer.Argument(
    None,
    help="Path to Dockerfile to build & run. If omitted, only lock & export requirements."
)):
    """
    - uv lock
    - uv export --format requirements --no-dev --hashes > requirements.txt
    - If DOCKERFILE argument is given: docker build -f DOCKERFILE . && docker run IMAGE
    """

    # ----- sanity checks for uv + project files -----
    if shutil.which("uv") is None:
        typer.secho("uv is not installed. Install it with: pip install uv", fg=typer.colors.RED)
        raise typer.Exit(1)

    if not pathlib.Path("pyproject.toml").exists():
        typer.secho(f"pyproject.toml not found in {pathlib.Path.cwd()}", fg=typer.colors.RED)
        raise typer.Exit(1)

    # ----- 1) solve and lock -----
    typer.secho("Running: uv lock", fg=typer.colors.BLUE)
    p = subprocess.run(["uv", "lock"])
    if p.returncode != 0:
        typer.secho("uv lock failed.", fg=typer.colors.RED)
        raise typer.Exit(p.returncode)

    # ----- 2) export pinned, hashed requirements -----
    typer.secho("Exporting hashed requirements to requirements.txt", fg=typer.colors.BLUE)
    p = subprocess.run(
        ["uv", "export", "--format", "requirements", "--no-dev", "--hashes"],
        capture_output=True, text=True
    )
    if p.returncode != 0:
        typer.secho("uv export failed:", fg=typer.colors.RED)
        if p.stderr:
            typer.echo(p.stderr.strip())
        raise typer.Exit(p.returncode)

    pathlib.Path("requirements.txt").write_text(p.stdout, encoding="utf-8")
    typer.secho("requirements.txt written.", fg=typer.colors.GREEN)

    # ----- 3) optional Docker build + run -----
    if dockerfile is None:
        typer.secho("No Dockerfile provided; skipping Docker build/run.", fg=typer.colors.BLUE)
        return

    df_path = pathlib.Path(dockerfile)
    if not df_path.exists():
        typer.secho(f"Dockerfile not found: {dockerfile}", fg=typer.colors.RED)
        raise typer.Exit(1)

    if shutil.which("docker") is None:
        typer.secho("Docker CLI is not installed or not on PATH.", fg=typer.colors.RED)
        raise typer.Exit(1)

    # Image name: directory-name + '-img' (overridable via env IMAGE_NAME)
    cwd_name = pathlib.Path.cwd().name
    safe_name = re.sub(r"[^a-z0-9_.-]+", "-", cwd_name.lower())
    image_name = os.environ.get("IMAGE_NAME", f"{safe_name}-img")

    # Tag: short git sha if available, else timestamp (overridable via env TAG)
    tag = os.environ.get("TAG")
    if not tag:
        try:
            tag = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            tag = time.strftime("%Y%m%d%H%M%S")

    image_ref = f"{image_name}:{tag}"

    typer.secho(f"Building Docker image: {image_ref}", fg=typer.colors.BLUE)
    build = subprocess.run(["docker", "build", "-f", str(df_path), "-t", image_ref, "."])
    if build.returncode != 0:
        typer.secho("docker build failed.", fg=typer.colors.RED)
        raise typer.Exit(build.returncode)

    typer.secho(f"Running container: {image_ref}", fg=typer.colors.BLUE)
    try:
        # interactive by default; relies on your ENTRYPOINT
        subprocess.check_call(["docker", "run", "--rm", "-it", image_ref])
    except subprocess.CalledProcessError as e:
        typer.secho(f"docker run failed (exit {e.returncode}).", fg=typer.colors.RED)
        raise typer.Exit(e.returncode)


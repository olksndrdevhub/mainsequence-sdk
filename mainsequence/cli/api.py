from __future__ import annotations
import os, re, subprocess, json, platform, pathlib, shlex, sys
import requests
from .config import backend_url, get_tokens, save_tokens, set_env_access

AUTH_PATHS = {
    "obtain": "/auth/jwt-token/token/",
    "refresh": "/auth/jwt-token/token/refresh/",
    "ping": "/auth/rest-auth/user/",
}

S = requests.Session()
S.headers.update({"Content-Type": "application/json"})

class ApiError(RuntimeError): ...

def _full(path: str) -> str:
    p = "/" + path.lstrip("/")
    return backend_url() + p

def _normalize_api_path(p: str) -> str:
    p = "/" + (p or "").lstrip("/")
    if not re.match(r"^/(api|auth|pods|orm|user)(/|$)", p):
        raise ApiError("Only /api/*, /auth/*, /pods/*, /orm/*, /user/* allowed")
    return p

def _access_token() -> str | None:
    # 1) environment override
    t = os.environ.get("MAIN_SEQUENCE_USER_TOKEN")
    if t:
        return t
    # 2) token file
    tok = get_tokens()
    return tok.get("access")

def _refresh_token() -> str | None:
    tok = get_tokens()
    return tok.get("refresh")

def login(email: str, password: str) -> dict:
    """Obtain & store tokens; set MAIN_SEQUENCE_USER_TOKEN in the current process."""
    url = _full(AUTH_PATHS["obtain"])
    payload = {"email": email, "password": password}  # Electron forces 'email' field
    r = S.post(url, data=json.dumps(payload))
    try:
        data = r.json()
    except Exception:
        data = {}
    if not r.ok:
        msg = data.get("detail") or data.get("message") or r.text
        raise ApiError(f"Login failed: {msg}")
    access = data.get("access") or data.get("token") or data.get("jwt") or data.get("access_token")
    refresh = data.get("refresh") or data.get("refresh_token")
    if not access or not refresh:
        raise ApiError("Server did not return expected tokens.")
    save_tokens(email, access, refresh)
    set_env_access(access)
    return {"username": email, "backend": backend_url()}

def refresh_access() -> str:
    refresh = _refresh_token()
    if not refresh:
        raise ApiError("Missing refresh token; run `mainsequence login`.")
    r = S.post(_full(AUTH_PATHS["refresh"]), data=json.dumps({"refresh": refresh}))
    data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
    if not r.ok:
        raise ApiError(data.get("detail") or f"Refresh failed ({r.status_code})")
    access = data.get("access")
    if not access:
        raise ApiError("Refresh succeeded but no access token returned.")
    # persist alongside existing refresh
    tokens = get_tokens()
    save_tokens(tokens.get("username") or "", access, refresh)
    set_env_access(access)
    return access

def authed(method: str, api_path: str, body: dict | None = None) -> requests.Response:
    api_path = _normalize_api_path(api_path)
    access = _access_token()
    if not access:
        # try refresh implicitly
        access = refresh_access()
    headers = {"Authorization": f"Bearer {access}"}
    r = S.request(method.upper(), _full(api_path), headers=headers,
                  data=None if method.upper() in {"GET","HEAD"} else json.dumps(body or {}))
    if r.status_code == 401:
        # one retry after refresh
        access = refresh_access()
        headers = {"Authorization": f"Bearer {access}"}
        r = S.request(method.upper(), _full(api_path), headers=headers,
                      data=None if method.upper() in {"GET","HEAD"} else json.dumps(body or {}))
    return r

# ---------- Helpers matched to Electron code ----------

def safe_slug(s: str) -> str:
    x = re.sub(r"[^a-z0-9-_]+", "-", (s or "project").lower()).strip("-")
    return x[:64] or "project"

def repo_name_from_git_url(url: str | None) -> str | None:
    if not url: return None
    s = re.sub(r"[?#].*$", "", url.strip())
    last = s.split("/")[-1] if "/" in s else s
    if last.lower().endswith(".git"): last = last[:-4]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", last)

def deep_find_repo_url(extra) -> str | None:
    if not isinstance(extra, dict): return None
    cand = ["ssh_url","git_ssh_url","repo_ssh_url","git_url","repo_url","repository","url"]
    for k in cand:
        v = extra.get(k)
        if isinstance(v, str) and (v.startswith("git@") or re.search(r"\.git($|\?)", v)):
            return v
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, str) and (vv.startswith("git@") or re.search(r"\.git($|\?)", vv)):
                    return vv
    for v in extra.values():
        if isinstance(v, dict):
            found = deep_find_repo_url(v)
            if found: return found
    return None

def get_current_user_profile() -> dict:
    who = authed("GET", AUTH_PATHS["ping"])
    d = who.json() if who.ok else {}
    uid = d.get("id") or d.get("pk") or (d.get("user") or {}).get("id") or d.get("user_id")
    if not uid:
        return {}
    full = authed("GET", f"/user/api/user/{uid}/")
    u = full.json() if full.ok else {}
    org_name = (u.get("organization") or {}).get("name") or u.get("organization_name") or ""
    return {"username": u.get("username") or "", "organization": org_name}

def get_projects() -> list[dict]:
    r = authed("GET", "/orm/api/pods/projects/")
    data = r.json() if r.ok else {}
    if isinstance(data, list):
        return data
    return data.get("results") or []

def fetch_project_env_text(project_id: int | str) -> str:
    r = authed("GET", f"/orm/api/pods/projects/{project_id}/get_environment/")
    raw = r.json() if r.headers.get("content-type","").startswith("application/json") else r.text
    if isinstance(raw, dict):
        raw = raw.get("environment") or raw.get("env") or raw.get("content") or raw.get("text") or ""
    return (raw or "")

def add_deploy_key(project_id: int | str, key_title: str, public_key: str) -> None:
    try:
        authed("POST", f"/orm/api/pods/projects/{project_id}/add_deploy_key/",
               {"key_title": key_title, "public_key": public_key})
    except Exception:
        pass

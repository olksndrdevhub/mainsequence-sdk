from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import importlib
import pkgutil

@dataclass(frozen=True)
class Page:
    slug: str
    title: str
    render: Callable          # signature: (ctx) -> None
    visible: bool = True
    has_sidebar: bool = False # <— lets the scaffold hide sidebar automatically
    order: int = 0            # optional ordering hook

_PAGES: Dict[str, Page] = {}

def register_page(slug: str, title: str, *, visible: bool = True, has_sidebar: bool = False, order: int = 0):
    """Decorator to register a page function."""
    def _wrap(fn: Callable) -> Callable:
        if slug in _PAGES:
            raise ValueError(f"Duplicate page slug '{slug}'")
        _PAGES[slug] = Page(slug=slug, title=title, render=fn, visible=visible, has_sidebar=has_sidebar, order=order)
        return fn
    return _wrap

def list_pages(*, visible_only: bool = False) -> List[Page]:
    items = list(_PAGES.values())
    items.sort(key=lambda p: (p.order, p.title.lower()))
    return [p for p in items if p.visible] if visible_only else items

def get_page(slug: str) -> Optional[Page]:
    return _PAGES.get(slug)

def autodiscover(package: str) -> None:
    """
    Import all modules in `package`, so that any @register_page side-effects run.
    Example: autodiscover("examples.alm_app.views")
    """
    pkg = importlib.import_module(package)
    if not getattr(pkg, "__path__", None):
        return
    for mod in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(mod.name)

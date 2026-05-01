"""Template endpoints for the staff dashboard's onboarding flow.

Two handlers — both auth-gated via ``authenticate_query`` — that back the
Phase 0 onboarding "template picker" surface (spec §5.7, plan Task 24):

  * ``GET /kg/templates`` — list every loaded template as
    ``{id, description, domain}``. The library is GLOBAL (lives on disk
    under ``services/kg/templates/<domain>/``), so this list does NOT
    vary by ``customer_id``. Auth is still required because the staff
    dashboard is the only audience and we want the kg surface to have
    consistent auth behaviour (no public unauth endpoints).
  * ``POST /kg/templates/{template_id}/apply`` — load the named template
    by id and PUT it as a class for the authenticated tenant. Reuses
    the same upsert + advisory-lock + kg_check path as
    ``PUT /classes/{class_id}`` (via ``services.kg.api.write.upsert_class``).

Important kg_check semantics for apply (spec §5.3, §7.2):

The kg_check universe used during apply is the TENANT'S existing class
universe, not the global template universe. So a template that
references another template via ``often_confused_with`` (e.g.
``auth-401-jwt-refresh`` → ``auth-403-rbac``) will fail with 422 if
the referenced template hasn't already been applied for this tenant.
This is correct: customers who want both classes need to apply both;
the template library being internally consistent is a property of the
library, not a license to short-circuit per-tenant validation.

``domain`` is the first dash-segment of the id (e.g.
``auth-401-jwt-refresh`` → ``"auth"``). Slug parsing uses
``id.split("-", 1)[0]`` rather than a hand-rolled regex — the
frontmatter id pattern (``^[a-z][a-z0-9-]{2,63}$``) already guarantees
a leading ``[a-z]`` segment.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.kg.advisory_lock import tenant_xact_lock
from services.kg.api.write import upsert_class
from services.kg.schema import BugClass
from services.kg.templates._loader import load_all_templates
from services.retrieval.auth import authenticate_query
from shared.db import with_tenant

router = APIRouter()


class TemplateListItem(BaseModel):
    """One row in the ``GET /kg/templates`` response."""

    id: str
    description: str
    domain: str


class TemplateListResponse(BaseModel):
    """Envelope for ``GET /kg/templates``. Wrapping in ``{items: [...]}``
    leaves room for future pagination / metadata without a breaking change."""

    items: list[TemplateListItem]


def _domain_for(template_id: str) -> str:
    """Return the leading dash-segment as the template's domain.

    ``auth-401-jwt-refresh`` → ``"auth"``. The frontmatter id pattern
    enforces a leading ``[a-z]`` segment so ``split("-", 1)[0]`` is
    safe — no regex needed.
    """
    return template_id.split("-", 1)[0]


def _load_template_index() -> dict[str, BugClass]:
    """Build an ``id → BugClass`` index for the loaded library.

    Templates ship as files on disk and are immutable at runtime, so a
    fresh load per request is cheap (24 small JSON files) and avoids
    the failure modes of a cached singleton (stale state across reloads
    in tests, lock contention on first init). If the library grows by
    an order of magnitude, lift this into a module-level lru_cache.
    """
    return {t.frontmatter.id: t for t in load_all_templates()}


@router.get("/templates", response_model=TemplateListResponse)
async def list_templates(
    customer_id: str = Depends(authenticate_query),
) -> TemplateListResponse:
    """Return every loaded template as ``{id, description, domain}``,
    ordered by ``id`` ascending. Cross-tenant — the template library is
    global. ``customer_id`` is dependency-resolved purely to enforce
    auth; the response does not vary by tenant.
    """
    items = [
        TemplateListItem(
            id=t.frontmatter.id,
            description=t.frontmatter.description,
            domain=_domain_for(t.frontmatter.id),
        )
        for t in load_all_templates()
    ]
    items.sort(key=lambda i: i.id)
    return TemplateListResponse(items=items)


@router.post("/templates/{template_id}/apply")
async def apply_template(
    template_id: str,
    customer_id: str = Depends(authenticate_query),
) -> JSONResponse:
    """Apply a named template as a class for the authenticated tenant.

    Loads the template by id, then runs the same upsert path as
    ``PUT /classes/{class_id}`` — advisory-lock + kg_check + upsert.
    Returns ``{"status": "ok", "class_id": <applied id>}``.

    Status codes:
      * 201 if newly applied for this tenant.
      * 200 if the same class_id was already PUT for this tenant
        (idempotent re-apply).
      * 404 if ``template_id`` doesn't exist in the library.
      * 422 if kg_check fails. This is most commonly a template that
        references another template (e.g. via ``often_confused_with``)
        which hasn't been applied yet for this tenant — the kg_check
        universe is the TENANT's existing class set, not the global
        template universe. Customers who want both classes need to
        apply both. See module docstring for the rationale.
    """
    index = _load_template_index()
    template = index.get(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="template not found")

    async with with_tenant(customer_id) as conn:
        async with tenant_xact_lock(conn, customer_id=customer_id):
            status_code = await upsert_class(
                conn,
                customer_id=customer_id,
                class_id=template.frontmatter.id,
                payload=template,
            )

    return JSONResponse(
        content={"status": "ok", "class_id": template.frontmatter.id},
        status_code=status_code,
    )

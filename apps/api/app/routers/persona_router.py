from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.security import require_admin
from app.services.persona import activate_persona_draft, generate_persona_draft

router = APIRouter(prefix="/api/persona", tags=["persona"], dependencies=[Depends(require_admin)])


class PersonaDraftRequest(BaseModel):
    person_id: str
    display_name: str = ""
    activate: bool = False


class PersonaDraftResponse(BaseModel):
    status: str
    persona: dict


@router.post("/draft", response_model=PersonaDraftResponse)
def build_persona_draft(req: PersonaDraftRequest) -> PersonaDraftResponse:
    """Generate a persona draft from ingested materials. Optionally activate it as the live profile."""
    draft = (
        activate_persona_draft(req.person_id, req.display_name)
        if req.activate
        else generate_persona_draft(req.person_id)
    )
    if "error" in draft:
        return PersonaDraftResponse(status="error", persona=draft)
    return PersonaDraftResponse(status="ok", persona=draft)

"""API endpoints for managing memories."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import require_admin
from app.services.memory import save_memory

router = APIRouter(prefix="/api/memory", tags=["memory"], dependencies=[Depends(require_admin)])


class AddMemoryRequest(BaseModel):
    person_id: str
    text: str = Field(min_length=1, max_length=5000)
    contributor: str = ""


class AddMemoryResponse(BaseModel):
    id: str
    status: str
    text: str


@router.post("/add", response_model=AddMemoryResponse)
def add_memory(req: AddMemoryRequest) -> AddMemoryResponse:
    """Manually add a memory/story/fact about the person."""
    result = save_memory(
        person_id=req.person_id,
        text=req.text,
        source="manual",
        contributor=req.contributor,
    )
    return AddMemoryResponse(**result)

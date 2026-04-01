"""Pydantic schemas for VLM responses."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ObjectDescription(BaseModel):
    """Detailed description of a single object in a scene."""

    id: Optional[str] = Field(default=None, description="Unique ID if available")
    name: str = Field(
        description="Common name (chair, table, door, wall, floor, sofa, etc.)"
    )
    confidence: float = Field(description="0-1 confidence score")
    attributes: List[str] = Field(
        default_factory=list,
        description="Color, material, state (open/closed), etc.",
    )
    description: str = Field(default="", description="Detailed description")
    affordances: List[str] = Field(
        default_factory=list,
        description="Possible actions (e.g., 'can sit on')",
    )
    state: str = Field(
        default="", description="Current state (open, closed, locked, etc.)"
    )
    location_description: str = Field(
        default="",
        alias="location description",
        description="Location relative to other objects",
    )


class ObjectCropDescription(BaseModel):
    """Description of a single object produced from a full frame with a highlighted bbox."""

    name: str = Field(
        description="Refined common name for the object (correct if the given label is wrong)"
    )
    confidence: float = Field(
        default=1.0, description="0-1 confidence in the identification"
    )
    attributes: List[str] = Field(
        default_factory=list,
        description="Color, material, size, state (open/closed/on/off), texture, shape, etc.",
    )
    description: str = Field(
        default="",
        description="One to three sentence factual description of the object's appearance and purpose",
    )
    affordances: List[str] = Field(
        default_factory=list,
        description="Possible interactions the object affords (e.g., 'can sit on', 'can open')",
    )
    state: str = Field(
        default="",
        description="Current operational state (open, closed, on, off, idle, etc.)",
    )
    location_description: str = Field(
        default="",
        alias="location description",
        description="Spatial location relative to the room or nearby objects",
    )
    spatial_relations: List[str] = Field(
        default_factory=list,
        description="Explicit spatial relations to named nearby objects (e.g., 'to the left of the sink')",
    )

    model_config = {"populate_by_name": True}


class ImageDescription(BaseModel):
    """Full scene description from a single image."""

    caption: str = Field(default="", description="One-sentence overview")
    room_type_guess: Optional[str] = Field(default=None, description="Room type guess")
    description: Optional[str] = Field(default=None, description="Detailed description")
    scene_layout: Optional[str] = Field(
        default=None, description="Spatial arrangement description"
    )
    objects: List[ObjectDescription] = Field(
        default_factory=list, description="Objects in the scene"
    )


class ObjectTag(BaseModel):
    """List of object category tags."""

    tags: List[str] = Field(
        default_factory=list,
        description="Unique, lowercase singular nouns for object categories",
    )


class FunctionalTag(BaseModel):
    """List of functional/interactive element tags."""

    functional_tags: List[str] = Field(
        default_factory=list,
        description="Unique, lowercase singular nouns for functional elements",
    )


class SceneObjectSummary(BaseModel):
    """Summary of an object instance in a room."""

    id: Optional[str] = Field(default=None, description="Unique ID if available")
    name: str = Field(default="")
    type: str = Field(default="")
    quantity: int = Field(default=1)


class SceneSummary(BaseModel):
    """Room-level scene summary from multiple observations."""

    room_summary: Optional[str] = Field(
        default=None, description="Dense summary of room layout and relationships"
    )
    room_type: Optional[str] = Field(default=None, description="Type of room")
    layout: Optional[str] = Field(default=None, description="Room layout description")
    objects: List[SceneObjectSummary] = Field(
        default_factory=list, description="Distinct object instances"
    )


class RoomBrief(BaseModel):
    """Brief summary of a single room."""

    id: Optional[str] = Field(default=None, description="Room ID if known")
    room_type: Optional[str] = Field(default=None, description="Semantic room name")
    caption: str = Field(default="", description="Short one-line caption")


class FloorSummaryOutput(BaseModel):
    """Floor-level summary from multiple room summaries."""

    floor_caption: str = Field(default="", description="Short floor-level caption")
    rooms: List[RoomBrief] = Field(
        default_factory=list, description="List of room briefs"
    )

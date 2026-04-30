"""Pydantic schemas for VLM responses."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ObjectDescription(BaseModel):
    """Detailed description of a single object in a scene."""

    id: Optional[str] = Field(description="Unique ID if available")
    name: str = Field(
        description="Common name (chair, table, door, wall, floor, sofa, bed, window, shelf, plant, monitor, cabinet, counter, sink, etc.)"
    )
    confidence: float = Field(description="0-1 confidence score")
    attributes: List[str] = Field(
        description="Color, material, state (open/closed), supports-standing/sitting, etc.",
    )
    description: str = Field(description="Detailed description of the object")
    affordances: List[str] = Field(
        description="Possible actions (e.g., 'can sit on', 'can place items in')",
    )
    state: str = Field(
        description="Current state of the object (e.g., 'open', 'closed', 'locked', 'unlocked', 'dirty')"
    )
    location_description: str = Field(
        alias="location description",
        description="Description of the object's location in the scene relative to other objects",
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

    caption: str = Field(description="One-sentence overview of the scene")
    room_type_guess: Optional[str] = Field(description="Guess for the room type")
    description: Optional[str] = Field(description="Detailed description of the scene")
    scene_layout: Optional[str] = Field(
        description="Description of the overall layout of the scene to describe the spatial arrangement of objects relative to each other"
    )
    objects: List[ObjectDescription] = Field(description="List of objects in the scene")


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

    id: Optional[str] = Field(description="Unique ID if available")
    name: str
    type: str
    quantity: int


class SceneSummary(BaseModel):
    """Room-level scene summary from multiple observations."""

    room_summary: Optional[str] = Field(
        description="Dense summary of the room layout, semantic attributes, and relationships"
    )
    room_type: Optional[str] = Field(description="The type of the room")
    layout: Optional[str] = Field(description="Description of the room layout")
    objects: List[SceneObjectSummary] = Field(description="Distinct object instances in the room")


class RoomBrief(BaseModel):
    """Brief summary of a single room."""

    id: Optional[str] = Field(default=None, description="Room ID if known")
    room_type: Optional[str] = Field(default=None, description="Semantic room name")
    caption: str = Field(default="", description="Short one-line caption")


class FloorSummaryOutput(BaseModel):
    """Floor-level summary from multiple room summaries."""

    floor_caption: str = Field(description="Short one-line caption for the floor")
    rooms: List[RoomBrief] = Field(description="List of rooms with their number/name and a short caption")

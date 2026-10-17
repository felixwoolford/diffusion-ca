"""System/user prompt construction for the rule LLM (SPEC §2.3).

The system prompt encodes the CA's character — the main place dynamics are
tuned. It must contain: the pixel-art frame, the neighborhood semantics
(which tile is self vs N/E/S/W and how the cell should relate), the
arm-specific prompt style, and the strict output contract.
"""
from __future__ import annotations

from . import RuleRequest

PROMPT_MAX = 500

RELATIONS = {
    "evolve": (
        "The cell must evolve *together with* its neighbors: where a motif, "
        "border or texture crosses a tile edge, continue it in the new state "
        "so the mosaic stays coherent; but the new tile must remain visually "
        "distinct from all four neighbors — never a copy or near-copy of one."
    ),
    "contrast": (
        "The cell must stay visually *distinct* from its four neighbors: "
        "choose shapes, colors or a motif that contrasts with them, while "
        "keeping the overall pixel-art style consistent."
    ),
}


def build_system_prompt(art_size: int, palette_n: int, arm: str,
                        relation: str = "evolve",
                        palette_hex: list[str] | None = None) -> str:
    frame = (
        f"You are the update rule of a cellular automaton whose cells are "
        f"{art_size}x{art_size} pixel-art tiles. You are describing a "
        f"{art_size}x{art_size} pixel-art tile: chunky shapes on a coarse grid, "
        f"limited {palette_n}-color palette, flat colors, no gradients, no fine detail."
    )
    if palette_hex:
        frame += " Exact palette (hex): " + " ".join(palette_hex) + "."

    semantics = (
        "You are shown one image: a cross layout of five tiles on a toroidal grid "
        "(borders wrap). Labels are drawn in the top-left corner of each tile: "
        "C is the CENTER tile — the cell whose next state you are writing; "
        "N/E/S/W are its north/east/south/west neighbors. Read the positions from "
        "the labels, never from their screen position."
    )
    relation_text = RELATIONS[relation]

    if arm == "t2i":
        style = (
            "Output a full descriptive prompt of the new center tile as a standalone "
            "image (text-to-image; no reference images are used, so the description "
            "must carry the whole state, including any continuation of neighbor motifs)."
        )
    elif arm == "edit_self":
        style = (
            "Output an EDIT INSTRUCTION for modifying the center tile (it is image 1, "
            "nearest-upscaled pixel art). State what changes and what is kept."
        )
    else:  # edit_neighborhood
        style = (
            "Output an EDIT INSTRUCTION. Image 1 is the center tile (modify it); "
            "images 2-5 are the N/E/S/W neighbors (references only — take from them, "
            "never modify them). State what changes and what is kept."
        )

    contract = (
        f"Output EXACTLY ONE prompt string, at most {PROMPT_MAX} characters, "
        "no quotes, no preamble, no labels, no reasoning."
    )
    return "\n\n".join([frame, semantics, relation_text, style, contract])


def frame_text(art_size: int, palette_n: int, palette_hex: list[str] | None = None) -> str:
    """The pixel-art framing paragraph alone (injected into RuleRequest.frame)."""
    t = (
        f"You are the update rule of a cellular automaton whose cells are "
        f"{art_size}x{art_size} pixel-art tiles. You are describing a "
        f"{art_size}x{art_size} pixel-art tile: chunky shapes on a coarse grid, "
        f"limited {palette_n}-color palette, flat colors, no gradients, no fine detail."
    )
    if palette_hex:
        t += " Exact palette (hex): " + " ".join(palette_hex) + "."
    return t


def build_user_text(req: RuleRequest) -> str:
    r, c = req.cell
    parts = [f"Step {req.step}, cell ({r},{c})."]
    if req.instruction:
        parts.append(f"Run instruction: {req.instruction}")
    parts.append("Output the prompt now.")
    return " ".join(parts)

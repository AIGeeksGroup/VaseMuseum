from __future__ import annotations

from typing import Any, Literal

from .caption import structured_caption_beazley_desc, structured_caption_limc
from .types import Doc


CaptionMode = Literal["structured", "nl"]


def _clean(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    return s


def doc_from_limc_row(
    row: dict[str, Any],
    *,
    caption_mode: CaptionMode,
    nl_caption_text: str | None = None,
) -> Doc:
    rid = _clean(row.get("ID")) or _clean(row.get("Resource IRI")) or _clean(row.get("ARK URL"))
    uri = _clean(row.get("Resource IRI")) or _clean(row.get("ARK URL"))
    title = _clean(row.get("Label")) or _clean(row.get("Object")) or f"LIMC {rid}"
    structured = structured_caption_limc(row)
    text = structured if caption_mode == "structured" else (_clean(nl_caption_text) or structured)

    meta = {
        "ark_url": _clean(row.get("ARK URL")),
        "resource_iri": _clean(row.get("Resource IRI")),
        "label": _clean(row.get("Label")),
        "id": _clean(row.get("ID")),
        "scene": _clean(row.get("Scene")),
        "scene_iri": _clean(row.get("Scene IRI")),
        "object": _clean(row.get("Object")),
        "material": _clean(row.get("Material")),
        "findspot": _clean(row.get("Findspot")),
        "origin": _clean(row.get("Origin")),
        "artist": _clean(row.get("Artist")),
        "category": _clean(row.get("Category")),
        "technique": _clean(row.get("Technique")),
        "keyword": _clean(row.get("Keyword")),
        "mythological_figure": _clean(row.get("Mythological figure")),
        "dating": _clean(row.get("Dating")),
        "inventory": _clean(row.get("Inventory")),
        "museum_url": _clean(row.get("Museum URL")),
    }

    return Doc(
        doc_id=f"LIMC::{rid}",
        source="LIMC",
        uri=uri or meta.get("ark_url") or "",
        title=title,
        text=text,
        meta=meta,
    )


def doc_from_beazley_desc_row(
    row: dict[str, Any],
    *,
    caption_mode: CaptionMode,
    nl_caption_text: str | None = None,
) -> Doc:
    uri = _clean(row.get("URI"))
    vase_no = _clean(row.get("Vase Number"))
    title = _clean(row.get("Decoration")) or _clean(row.get("Shape Name")) or f"Beazley {vase_no}"
    structured = structured_caption_beazley_desc(row)
    text = structured if caption_mode == "structured" else (_clean(nl_caption_text) or structured)

    meta = {
        "uri": uri,
        "vase_number": vase_no,
        "fabric": _clean(row.get("Fabric")),
        "technique": _clean(row.get("Technique")),
        "sub_technique": _clean(row.get("Sub Technique")),
        "shape_name": _clean(row.get("Shape Name")),
        "provenance": _clean(row.get("Provenance")),
        "date": _clean(row.get("Date")),
        "inscriptions": _clean(row.get("Inscriptions")),
        "attributed_to": _clean(row.get("Attributed To")),
        "decoration": _clean(row.get("Decoration")),
        "collection_record": _clean(row.get("Collection Record")),
        "publication_record": _clean(row.get("Publication Record")),
        "limc_id": _clean(row.get("LIMC ID")),
        "limc_web": _clean(row.get("LIMC Web")),
        "pleiades_uri": _clean(row.get("Pleiades URI")),
        "lat": _clean(row.get("Latitude")),
        "lon": _clean(row.get("Longitude")),
    }

    return Doc(
        doc_id=f"BEAZLEY_DESC::{vase_no or uri}",
        source="BEAZLEY_DESC",
        uri=uri,
        title=title,
        text=text,
        meta=meta,
    )


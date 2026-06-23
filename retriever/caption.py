from __future__ import annotations

from typing import Any


def _clean(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    return s


def _join(*parts: str) -> str:
    xs = [p.strip() for p in parts if p and p.strip()]
    return " · ".join(xs)


def structured_caption_limc(row: dict[str, Any]) -> str:
    obj = _clean(row.get("Object"))
    shape = _clean(row.get("Shape: Kerameikos URL"))
    mat = _clean(row.get("Material"))
    findspot = _clean(row.get("Findspot"))
    findspot_detail = _clean(row.get("Findspot detail"))
    origin = _clean(row.get("Origin"))
    country = _clean(row.get("Country"))
    artist = _clean(row.get("Artist"))
    category = _clean(row.get("Category"))
    technique = _clean(row.get("Technique"))
    keyword = _clean(row.get("Keyword"))
    scene = _clean(row.get("Scene"))
    myth = _clean(row.get("Mythological figure"))
    hist = _clean(row.get("Historical figure"))
    dating = _clean(row.get("Dating"))
    inventory = _clean(row.get("Inventory"))
    museum = _clean(row.get("Museum URL"))
    desc = _clean(row.get("Description"))
    label = _clean(row.get("Label"))
    rid = _clean(row.get("ID"))
    ark = _clean(row.get("ARK URL"))

    head = _join(
        f"LIMC entry {rid}" if rid else "",
        f"Label {label}" if label else "",
        obj,
        f"Category {category}" if category else "",
        f"Technique {technique}" if technique else "",
        f"Material {mat}" if mat else "",
    )

    place = _join(
        f"Findspot {findspot}" if findspot else "",
        findspot_detail,
        f"Origin {origin}" if origin else "",
        country,
    )

    icono = _join(
        f"Scene {scene}" if scene else "",
        f"Mythological figures {myth}" if myth else "",
        f"Historical figures {hist}" if hist else "",
        f"Keywords {keyword}" if keyword else "",
    )

    admin = _join(
        f"Artist {artist}" if artist else "",
        f"Dating {dating}" if dating else "",
        f"Inventory {inventory}" if inventory else "",
    )

    links = _join(
        f"ARK {ark}" if ark else "",
        f"Museum {museum}" if museum else "",
    )

    chunks = [x for x in (head, place, icono, admin, desc, links) if x]
    return "\n".join(chunks).strip()


def structured_caption_beazley_desc(row: dict[str, Any]) -> str:
    uri = _clean(row.get("URI"))
    vase_no = _clean(row.get("Vase Number"))
    fabric = _clean(row.get("Fabric"))
    technique = _clean(row.get("Technique"))
    sub_tech = _clean(row.get("Sub Technique"))
    shape = _clean(row.get("Shape Name"))
    prov = _clean(row.get("Provenance"))
    date = _clean(row.get("Date"))
    attributed = _clean(row.get("Attributed To"))
    deco = _clean(row.get("Decoration"))
    coll = _clean(row.get("Collection Record"))
    pub = _clean(row.get("Publication Record"))
    limc_id = _clean(row.get("LIMC ID"))
    limc_web = _clean(row.get("LIMC Web"))

    head = _join(
        f"Beazley vase {vase_no}" if vase_no else "",
        fabric,
        technique,
        sub_tech,
        shape,
    )

    context = _join(
        f"Provenance {prov}" if prov else "",
        f"Date {date}" if date else "",
        f"Attributed to {attributed}" if attributed else "",
    )

    refs = _join(
        coll,
        pub,
        f"LIMC {limc_id}" if limc_id else "",
    )

    links = _join(
        f"URI {uri}" if uri else "",
        f"LIMC web {limc_web}" if limc_web else "",
    )

    chunks = [x for x in (head, context, deco, refs, links) if x]
    return "\n".join(chunks).strip()


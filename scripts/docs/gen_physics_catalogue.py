"""Print docs/reference/physics_catalogue.md, the reference page of the physics catalogue, from the registry.

    python scripts/docs/gen_physics_catalogue.py > docs/reference/physics_catalogue.md

The page is generated from ``axionmultiphysics.physics(routes=True)`` (the JSON that ``axmp physics --json --routes``
prints), so it needs the compiled core. The output is deterministic; ``tests/test_physics_catalogue.py`` fails when the
committed page differs from it. Do not edit the page by hand: change the registry
(``cpp/src/physics_registry.cpp``) and regenerate.
"""
from __future__ import annotations

import sys

import axionmultiphysics as axmp


def esc(text) -> str:
    """Text for a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def code(names) -> str:
    return ", ".join(f"`{n}`" for n in names) if names else "none"


def data_keys(keys) -> str:
    if not keys:
        return "none"
    parts = []
    for k in keys:
        attrs = [k["type"]]
        if k["unit"]:
            attrs.append(k["unit"])
        if k["field_expression"]:
            attrs.append("field expression")
        if not k["required"]:
            attrs.append("optional")
        if k["rampable"]:
            attrs.append("rampable")
        if k["per_depth_twin"]:
            attrs.append(f"`{k['per_depth_twin']}` in a planar model")
        parts.append(f"`{k['name']}` ({'; '.join(attrs)})")
    return "; ".join(parts)


def requires(r) -> str:
    q = r["requires"]
    needs = []
    if q["mesher"]:
        needs.append("the mesher")
    if q["mpi"]:
        needs.append("an MPI build")
    if q["fluids"]:
        needs.append("the fluids library")
    if q["engine_3d"]:
        needs.append("the 3D engine")
    needs.append("any number of ranks" if q["max_ranks"] == 0 else f"at most {q['max_ranks']} rank")
    return ", ".join(needs)


def type_section(t, out) -> None:
    out.append(f"### `{t['name']}` ({t['status']})")
    out.append("")
    out.append(f"{t['title']}.")
    if t["aliases"]:
        out.append(f"Aliases (names other codes use; a document writes the name above): {code(t['aliases'])}.")
    if t["provides"]:
        out.append(f"Publishes the fields {code(t['provides'])}.")
    if t["absent_phrase"]:
        out.append(f"Not implemented in this version: a document that uses it is reported unavailable "
                   f"(\"{t['absent_phrase']}\").")
    out.append("")
    if t["options"]:
        out.append("| option | type | values | default | highest geometry order |")
        out.append("|---|---|---|---|---|")
        for o in t["options"]:
            mgo = ", ".join(f"`{v}`: {n}" for v, n in o["max_geometry_order"].items()) or "any"
            values = code(o["values"]) if o["values"] else "a number"
            default = f"`{o['default']}`" if o["default"] else "none (to be written)"
            out.append(f"| `{o['name']}` | {o['type']} | {values} | {default} | {mgo} |")
        out.append("")
    for label, kinds in (("condition kind", t["condition_kinds"]), ("source kind", t["source_kinds"])):
        if not kinds:
            continue
        out.append(f"| {label} | data keys | aliases |")
        out.append("|---|---|---|")
        for k in kinds:
            out.append(f"| `{k['name']}` | {esc(data_keys(k['data']))} | {code(k['aliases'])} |")
        out.append("")
    if t["quantities"]:
        out.append("| geometry | quantities |")
        out.append("|---|---|")
        for g in ("planar", "axisymmetric", "3d"):
            if g not in t["quantities"]:
                continue
            qs = "; ".join(f"`{q['name']}` [{q['unit']}]" + (f" ({esc(q['convention'])})" if q["convention"] else "")
                           for q in t["quantities"][g])
            out.append(f"| {g} | {qs} |")
        out.append("")
    if t["not_implemented"]:
        out.append("Not implemented:")
        out.append("")
        for n in t["not_implemented"]:
            out.append(f"- {n}")
        out.append("")


def status(r) -> str:
    """What the page says of a route's implementation, from the boolean ``implemented`` of its row."""
    return "implemented" if r["implemented"] else "not implemented yet"


def route_section(r, out) -> None:
    out.append(f"### `{r['name']}`")
    out.append("")
    geometry = ", ".join(r["geometries"]) or "no geometry"
    out.append(f"- **Status:** {status(r)}.")
    out.append(f"- **Physics:** {code(r['physics'])}; couplings: {code(r['couplings'])}; geometry: {geometry}; "
               f"study: {', '.join(r['study_types'])}.")
    if r["when"]:
        out.append("- **Selected by:** " + ", ".join(f"`{k}` = `{v}`" for k, v in r["when"].items()) + ".")
    if r["when_sources"]:
        sources = "; ".join(f"{t}: {code(s)}" for t, s in r["when_sources"].items())
        out.append(f"- **Selected by the sources:** {sources}.")
    out.append(f"- **Engine:** {r['engine']}.")
    out.append(f"- **Needs:** {requires(r)}.")
    if r["strategies"]:
        out.append(f"- **Coupling strategies:** {code(r['strategies'])} (default `{r['default_strategy']}`).")
    for t, kinds in r["condition_kinds"].items():
        out.append(f"- **Condition kinds offered** ({t}): {code(kinds)}.")
    for t, kinds in r["not_offered"].items():
        for k, why in kinds.items():
            road = r["roadmap"].get(t, {}).get(k)
            out.append(f"- **Not offered** ({t}): `{k}`: {why}" + (f" (roadmap item '{road}')" if road else "") + ".")
    for t, by_value in r["not_offered_with"].items():
        for value, kinds in by_value.items():
            for k, why in kinds.items():
                out.append(f"- **Not offered with** `{value}` ({t}): `{k}`: {why}.")
    for t, rules in r["options"].items():
        parts = []
        for o in rules:
            if o["forced"]:
                parts.append(f"`{o['name']}` forced to `{o['forced']}`")
            elif o["offered"]:
                parts.append(f"`{o['name']}`: {code(o['offered'])} (default `{o['default']}`)")
            else:
                parts.append(f"`{o['name']}`: a number (default `{o['default']}`)")
        out.append(f"- **Options** ({t}): " + "; ".join(parts) + ".")
    for t, blocks in r["material_blocks"].items():
        out.append(f"- **Material blocks** ({t}): {code(blocks)}.")
    for t, props in r["uniform"].items():
        out.append(f"- **Uniform over the instance** ({t}): {code(props)}.")
    for t, conds in r["uniform_when"].items():
        for u in conds:
            when = f"`{u['option']}`" + (f" and a `{u['output']}` output" if u["output"] else "")
            out.append(f"- **Uniform over the instance with** {when} ({t}): `{u['property']}` ({u['why']}).")
    for t, props in r["one_expression"].items():
        out.append(f"- **One expression over the instance** ({t}): {code(props)}.")
    out.append(f"- **Element types:** {code(r['element_types'])}.")
    if r["equal_domains"]:
        out.append("- **Equal domains:** every instance of the row has the same domain.")
    if r["force_reaction"]:
        out.append(f"- **Force output:** {r['force_reaction']}.")
    if r["axisymmetric_integrals"]:
        out.append(f"- **Axisymmetric integrals of the engine:** {r['axisymmetric_integrals'].replace('_', ' ')}.")
    if r["rampable"]:
        out.append(f"- **Rampable:** {code(r['rampable'])}.")
    if r["layer_rules"]:
        out.append(f"- **Layer rules:** {code(r['layer_rules'])}.")
    for t in dict.fromkeys(s["physics"] for s in r["solvers"]):
        settings = ", ".join(f"`{s['name']}` ({s['type']}, default `{_value(s['default'])}`)"
                             for s in r["solvers"] if s["physics"] == t)
        out.append(f"- **Solver settings** ({t}): {settings}.")
    for t in r["solvers_of_own_row"]:
        out.append(f"- **Solver settings** ({t}): those of the row of its own type that the instance's options "
                   "select in this geometry.")
    for limit in r["limits"]:
        out.append(f"- {limit}.")
    out.append("")


def _value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(v) if isinstance(v, float) else str(v)


def page(cat: dict) -> str:
    out = ["# Physics catalogue (beta)", ""]
    out.append("<!-- Generated by scripts/docs/gen_physics_catalogue.py from axionmultiphysics.physics(routes=True); "
               "do not edit by hand. -->")
    out.append("")
    out.append(
        "The physics catalogue is the registry that model documents (designs with a `physics` section) are validated "
        "against (`axmp validate`); this version does not solve them: the physics types, the coupling terms between them, and the route table, one row per combination of "
        "physics, couplings, geometry class and study type that an engine of this version solves, with what the row "
        "offers and what it needs. It is compiled into every build and is the same in each, so a document validates "
        "the same way everywhere. Everything on this page is **beta**, and no route is implemented in this version: a "
        "model document is validated and resolved, and a planar or axisymmetric one is meshed (`axmp mesh`), but none "
        "is solved.")
    out.append("")
    out.append(
        "Print it with `axmp physics` (`--routes` adds the route table, `--type T` restricts it to one type, `--all` "
        f"prints the details, `--json` the machine form of schema `{cat['schema']}`), or in Python with "
        "`axionmultiphysics.physics(routes=True, type=None)`.")
    out.append("")
    out.append("## Physics types")
    out.append("")
    out.append("| type | status | title | publishes |")
    out.append("|---|---|---|---|")
    for t in cat["types"]:
        out.append(f"| `{t['name']}` | {t['status']} | {esc(t['title'])} | {code(t['provides'])} |")
    out.append("")
    out.append(
        "Data-key types: `number`, `boolean`, `names` (a list of names), `boundary` (a boundary name), `number_vector` "
        "(a list of numbers), `vector` (a field expression per component: (x, y) planar, (r, z, theta) axisymmetric, "
        "(x, y, z) in 3D) and `meridional_vector` ((x, y), (r, z) and (x, y, z)). Option values are listed as text; an "
        "integer option is written as a number in a document.")
    out.append("")
    for t in cat["types"]:
        type_section(t, out)
    out.append("## Coupling terms")
    out.append("")
    out.append("| term | status | from | to | support | reads | data keys |")
    out.append("|---|---|---|---|---|---|---|")
    for t in cat["terms"]:
        out.append(f"| `{t['name']}` | {t['status']} | {code(t['producers'])} | {code(t['consumers'])} | "
                   f"{t['support']} | {code(t['reads'])} | {esc(data_keys(t['data']))} |")
    out.append("")
    out.append("## Routes")
    out.append("")
    out.append("| route | physics | couplings | geometry | study | status |")
    out.append("|---|---|---|---|---|---|")
    for r in cat["routes"]:
        geometry = ", ".join(r["geometries"]) or "none"
        out.append(f"| `{r['name']}` | {code(r['physics'])} | {code(r['couplings'])} | {geometry} | "
                   f"{', '.join(r['study_types'])} | {status(r)} |")
    out.append("")
    for r in cat["routes"]:
        route_section(r, out)
    out.append("## Layer rules")
    out.append("")
    out.append("Rules of this layer that are stricter than the engine on purpose; each route lists the ones it "
               "applies.")
    out.append("")
    out.append("| rule | what it refuses |")
    out.append("|---|---|")
    for rule in cat["layer_rules"]:
        out.append(f"| `{rule['name']}` | {esc(rule['description'])} |")
    out.append("")
    out.append("## Reserved instance ids")
    out.append("")
    out.append(f"A physics instance may not be called {code(cat['reserved_ids'])}.")
    out.append("")
    return "\n".join(out)


def main() -> int:
    sys.stdout.write(page(axmp.physics(routes=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

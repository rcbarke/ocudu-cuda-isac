"""Registry of sensing sources and the record every source produces.

A sensing source is a reference signal that a receiver observes: its channel, link direction,
resource elements, known transmitted symbols, ports, executing processor and slot timing. One
module per source registers a builder under the source's name. The observer and the sensing core
read the record and never the module, so a new source changes no other code.
"""

from math import isfinite

SCHEMA = "sensing-source/v1"
FIELDS = ("schema", "source", "channel", "link_direction", "reference_signal", "ports", "processor",
          "slot", "grid", "mask", "reference")
LINK_DIRECTIONS = ("uplink", "downlink")
_REGISTRY = {}


class SourceError(ValueError):
    """A sensing source is unknown or its record violates the documented interface."""


def register(name, builder):
    """Add one source. A name is registered once, so two modules cannot claim the same entry."""
    if not isinstance(name, str) or not name or not callable(builder):
        raise SourceError("a sensing source needs a name and a builder")
    if name in _REGISTRY:
        raise SourceError(f"sensing source {name!r} is already registered")
    _REGISTRY[name] = builder


def names():
    return sorted(_REGISTRY)


def resolve(name):
    """Return the builder of a registered source. An unknown name is refused, never defaulted."""
    try:
        return _REGISTRY[name]
    except (KeyError, TypeError):
        known = ", ".join(names()) or "none"
        raise SourceError(f"unknown sensing source {name!r} (registered: {known})") from None


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise SourceError(f"{name} must be an integer from {low} to {high}")
    return value


def validate(record):
    """Check one sensing-source record with its arrays inline and return it.

    The mask lists each pilot once as [symbol, subcarrier] in carrier coordinates. The reference
    holds one row per port with a nonzero [real, imaginary] pair per pilot. A pilot without
    reference power cannot yield an estimate, so it is refused here and not at the observer.
    """
    if not isinstance(record, dict) or set(record) != set(FIELDS):
        raise SourceError(f"a sensing source record requires exactly {', '.join(FIELDS)}")
    if record["schema"] != SCHEMA:
        raise SourceError(f"unsupported sensing source schema {record['schema']!r}")
    for key in ("source", "channel", "reference_signal"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise SourceError(f"{key} must be a nonempty string")
    if record["link_direction"] not in LINK_DIRECTIONS:
        raise SourceError("link_direction must be uplink or downlink")
    ports = record["ports"]
    if not isinstance(ports, list) or not ports:
        raise SourceError("ports must be a nonempty list")
    for port in ports:
        _integer(port, "port", 0, 4095)
    if len(set(ports)) != len(ports):
        raise SourceError("a port is listed twice")
    processor = record["processor"]
    if not isinstance(processor, dict) or set(processor) != {"name", "backend"} \
            or not isinstance(processor["name"], str) or not processor["name"].strip() \
            or not (processor["backend"] is None or isinstance(processor["backend"], str)):
        raise SourceError("processor requires a name and a backend (null until a run executes it)")
    grid = record["grid"]
    if not isinstance(grid, dict) or set(grid) != {"coordinates", "symbols", "subcarriers"}:
        raise SourceError("grid requires exactly coordinates, symbols, subcarriers")
    if grid["coordinates"] != "carrier":
        raise SourceError("pilot coordinates must be carrier coordinates")
    symbols = _integer(grid["symbols"], "grid symbols", 1, 14)
    subcarriers = _integer(grid["subcarriers"], "grid subcarriers", 1, 3300)
    slot = record["slot"]
    if not isinstance(slot, dict) or not {"scs_khz", "sfn", "slot", "symbols"} <= set(slot):
        raise SourceError("slot requires scs_khz, sfn, slot and symbols")
    mask = record["mask"]
    if not isinstance(mask, dict) or set(mask) != {"axes", "index"} or not isinstance(mask["index"], list) \
            or not mask["index"]:
        raise SourceError("mask requires axes and a nonempty index")
    seen = set()
    for entry in mask["index"]:
        if not isinstance(entry, list) or len(entry) != 2:
            raise SourceError("each mask entry is [symbol, subcarrier]")
        coordinate = (_integer(entry[0], "pilot symbol", 0, symbols - 1),
                      _integer(entry[1], "pilot subcarrier", 0, subcarriers - 1))
        if coordinate in seen:
            raise SourceError(f"duplicate pilot at symbol {coordinate[0]}, subcarrier {coordinate[1]}")
        seen.add(coordinate)
    if sorted(seen) != [tuple(entry) for entry in mask["index"]]:
        raise SourceError("pilots must be ordered by symbol, then by subcarrier")
    if sorted({symbol for symbol, _ in seen}) != slot["symbols"]:
        raise SourceError("the slot's reference symbols differ from the symbols in the mask")
    reference = record["reference"]
    if not isinstance(reference, dict) or set(reference) != {"axes", "amplitude", "values"}:
        raise SourceError("reference requires exactly axes, amplitude, values")
    amplitude = reference["amplitude"]
    if type(amplitude) not in (int, float) or not isfinite(amplitude) or amplitude <= 0:
        raise SourceError("reference amplitude must be a positive finite number")
    values = reference["values"]
    if not isinstance(values, list) or len(values) != len(ports):
        raise SourceError("reference values need one row per port")
    for row in values:
        if not isinstance(row, list) or len(row) != len(mask["index"]):
            raise SourceError("each reference row needs one value per pilot")
        for pair in row:
            if not isinstance(pair, list) or len(pair) != 2 \
                    or any(type(part) not in (int, float) or not isfinite(part) for part in pair):
                raise SourceError("reference values are finite [real, imaginary] pairs")
            if pair[0] == 0 and pair[1] == 0:
                raise SourceError("a configured pilot has zero reference power")
    return record


from . import pusch_dmrs  # noqa: E402  (registers the first entry)

register(pusch_dmrs.NAME, pusch_dmrs.build)

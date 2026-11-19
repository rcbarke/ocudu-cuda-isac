"""Small scalar reference for validating a recorded channel-to-grid handoff."""

from math import isfinite, prod


class ContractError(ValueError):
    """A replay record or timing trace violates the documented interface."""


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def object_fields(obj, fields, name):
    if not isinstance(obj, dict) or set(obj) != set(fields):
        raise ContractError(f"{name} requires exactly {', '.join(fields)}")
    return obj


def array(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ContractError(f"{name} must be an array of length {length}")
    return value


def complex_values(value, length, name):
    result = []
    for item in array(value, length, name):
        real, imag = array(item, 2, name)
        for component in (real, imag):
            if type(component) not in (int, float) or not isfinite(component):
                raise ContractError(f"{name} requires finite real/imaginary pairs")
        result.append(complex(real, imag))
    return result


def received_grid(shape, x, h, noise):
    """Return Y[r,m,k] from X[t,m,k], H[r,t,m,k], W[r,m,k], last axis fastest."""
    nr, nt, nm, nk = shape
    stride = nm * nk
    received = []
    for rx in range(nr):
        for resource_element in range(stride):
            value = noise[rx * stride + resource_element]
            for tx in range(nt):
                value += h[(rx * nt + tx) * stride + resource_element] * x[tx * stride + resource_element]
            if not isfinite(value.real) or not isfinite(value.imag):
                raise ContractError("channel application produced a nonfinite observation")
            received.append([value.real, value.imag])
    return received


def materialize(episode):
    """Validate one synthetic episode and construct disjoint ingress and oracle records."""
    object_fields(episode, ("schema", "episode_id", "clock_domain", "shape", "subcarrier_offsets_hz", "records"), "episode")
    if episode["schema"] != "fixture-grid/v1":
        raise ContractError("unsupported fixture-grid schema")
    eid, clock = episode["episode_id"], episode["clock_domain"]
    if not all(isinstance(x, str) and x.strip() for x in (eid, clock)):
        raise ContractError("episode_id and clock_domain must be nonempty strings")
    shape = [integer(x, "shape dimension", 1) for x in array(episode["shape"], 4, "shape")]
    nr, nt, nm, nk = shape
    frequencies = array(episode["subcarrier_offsets_hz"], nk, "subcarrier_offsets_hz")
    if any(type(f) is not int for f in frequencies) or len(set(frequencies)) != nk:
        raise ContractError("subcarrier offsets must be unique integer Hz values")
    records = episode["records"]
    if not isinstance(records, list) or not records:
        raise ContractError("records must be a nonempty array")
    ingress, oracle = [], []
    last_seq, previous_first = -1, -1
    for record in records:
        object_fields(record, ("seq", "symbol_times_ns", "ready_ns", "pilot_mask", "x", "h", "w", "case"), "slot")
        seq = integer(record["seq"], "seq")
        if seq <= last_seq:
            raise ContractError("seq must increase across source records")
        times = [integer(t, "symbol time") for t in array(record["symbol_times_ns"], nm, "symbol_times_ns")]
        ready = integer(record["ready_ns"], "ready_ns")
        if times[0] < previous_first or any(a >= b for a, b in zip(times, times[1:])) or ready < times[-1]:
            raise ContractError("symbol/ready times violate acquisition order")
        pilot_mask = array(record["pilot_mask"], nm * nk, "pilot_mask")
        if any(type(v) is not bool for v in pilot_mask):
            raise ContractError("pilot mask entries must be JSON booleans")
        if not isinstance(record["case"], str) or not record["case"].strip():
            raise ContractError("case must name the synthetic test condition")
        x = complex_values(record["x"], nt * nm * nk, "x")
        h = complex_values(record["h"], prod(shape), "h")
        w = complex_values(record["w"], nr * nm * nk, "w")
        identity = {"episode_id": eid, "seq": seq}
        ingress.append({
            **identity, "clock_domain": clock, "symbol_times_ns": times, "ready_ns": ready,
            "shape_rx_symbol_subcarrier": [nr, nm, nk], "subcarrier_offsets_hz": frequencies,
            "pilot_mask": pilot_mask, "y": received_grid(shape, x, h, w),
        })
        oracle.append({**identity, "synthetic_case": record["case"], "shape_rx_tx_symbol_subcarrier": shape,
                       "h": record["h"], "x": record["x"], "w": record["w"]})
        last_seq, previous_first = seq, times[0]
    return ingress, oracle


def join_timing(ingress, trace):
    """Join an explicit observer trace. Missing/dropped records remain in the denominator."""
    object_fields(trace, ("episode_id", "clock_domain", "deadline_ns", "events"), "trace")
    if not ingress or any(r["episode_id"] != trace["episode_id"] or r["clock_domain"] != trace["clock_domain"] for r in ingress):
        raise ContractError("trace episode/clock does not match ingress")
    deadline = integer(trace["deadline_ns"], "deadline_ns", 1)
    records_by_id = {r["seq"]: r for r in ingress}
    if len(records_by_id) != len(ingress):
        raise ContractError("duplicate ingress identifier")
    events = trace["events"]
    if not isinstance(events, list):
        raise ContractError("events must be an array")
    seen = {}
    for event in events:
        if not isinstance(event, dict) or event.get("status") not in ("published", "dropped"):
            raise ContractError("event status must be published or dropped")
        published = event["status"] == "published"
        object_fields(event, ("seq", "status", "publish_ns" if published else "reason"), "event")
        seq = integer(event["seq"], "event seq")
        if seq not in records_by_id or seq in seen:
            raise ContractError(f"unexpected or duplicate event identifier {seq}")
        if published:
            publish = integer(event["publish_ns"], "publish_ns")
            record = records_by_id[seq]
            if publish < record["ready_ns"]:
                raise ContractError(f"publication precedes readiness for seq {seq}")
            latency = publish - record["symbol_times_ns"][0]
            seen[seq] = {"outcome": "on_time" if latency <= deadline else "late", "latency_ns": latency}
        else:
            if not isinstance(event["reason"], str) or not event["reason"].strip():
                raise ContractError("dropped event requires a reason")
            seen[seq] = {"outcome": "dropped", "reason": event["reason"]}
    outcomes = [{"seq": seq, **seen.get(seq, {"outcome": "missing"})} for seq in records_by_id]
    counts = {key: 0 for key in ("on_time", "late", "dropped", "missing")}
    for row in outcomes:
        counts[row["outcome"]] += 1
    return {"episode_id": trace["episode_id"], "clock_domain": trace["clock_domain"], "deadline_ns": deadline,
            "scheduled_updates": len(ingress), "counts": counts, "updates": outcomes}

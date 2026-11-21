"""The uplink grant fixture of an episode: one legal grant per scheduled uplink slot (Sprint 2.2).

A pusch-grant-schedule/v1 record places PUSCH opportunities on the uplink slots of a TDD pattern.
Every opportunity carries its frame, its slot, its start time and the grant it executes. All
coordinates and times come from the accepted grant record: the subcarrier offsets from
Grant.subcarrier_offset_hz and the symbol times from Grant.symbol_times. Nothing is derived again.

The transmitted waveform of a grant depends on the slot index in the frame (through the DM-RS
sequence) and not on the frame number. One grant record per slot index therefore serves every
frame. The record of the base grant's own slot is the base grant itself, so the transmit grid that
Sprint 2.1 archived for it applies unchanged. Other slot indices get the base record with that
slot and the identifier "<grant_id>.sNN".

Standard library only. Times are exact integers in T_c (1,966,080,000 per second).
"""

import hashlib
import json

from . import grant as grants
from .grant import GrantError
from ..sensing.sources import pusch_dmrs

SCHEMA = "pusch-grant-schedule/v1"
FIELDS = ("schema", "schedule_id", "tdd", "start_sfn", "periods", "payload")
FRAME_TC = pusch_dmrs.TC_PER_SECOND // 100
MAX_OPPORTUNITIES = 1_000_000


class ScheduleError(ValueError):
    """A grant schedule violates the documented interface."""


def slots_per_frame(scs_khz):
    return 10 * scs_khz // 15


def slot_duration_tc(scs_khz, slot):
    """Length of one slot in T_c. Slots differ where only some carry the longer first prefix."""
    last = pusch_dmrs.symbol_times(scs_khz, slot)[-1]
    return last["useful_start_tc"] + last["useful_tc"]


def record_sha256(record):
    """Hash of a JSON record in canonical form, to bind a file to the record it was made for."""
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


class Opportunity:
    """One scheduled PUSCH transmission: where it lies in time and which grant it executes."""

    def __init__(self, index, period, sfn, slot, absolute_slot, slot_start_tc, grant):
        self.index, self.period, self.sfn, self.slot = index, period, sfn, slot
        self.absolute_slot, self.slot_start_tc, self.grant = absolute_slot, slot_start_tc, grant

    def symbol_time_tc(self, symbol):
        """Start of the useful part of one symbol, in T_c from the start of the schedule."""
        return self.slot_start_tc + self.grant.symbol_times()[symbol]["useful_start_tc"]

    def describe(self):
        return {"index": self.index, "period": self.period, "sfn": self.sfn, "slot": self.slot,
                "absolute_slot": self.absolute_slot, "slot_start_tc": self.slot_start_tc,
                "grant_id": self.grant.grant_id}


class Schedule:
    """A validated schedule: its record, the base grant, the grant of every slot index and the opportunities."""

    def __init__(self, record, base, slot_grants, opportunities):
        self._record, self.base, self.opportunities = record, base, opportunities
        self.schedule_id = record["schedule_id"]
        self.grants = {grant.grant_id: grant for grant in slot_grants.values()}
        self.pattern = record["tdd"]["pattern"]
        self.ul_slots = list(record["tdd"]["ul_slots"])
        self.periods = record["periods"]

    def record(self):
        return json.loads(json.dumps(self._record))

    def describe_grants(self):
        """What a consumer needs to know of each grant: its slot, its record hash and its symbol times."""
        result = {}
        for grant_id, grant in sorted(self.grants.items()):
            times = grant.symbol_times()
            result[grant_id] = {
                "slot": grant.slot, "is_base_grant": grant is self.base,
                "record_sha256": record_sha256(grant.record()),
                "symbol_useful_start_tc": [entry["useful_start_tc"] for entry in times],
                "symbol_useful_tc": times[0]["useful_tc"],
                "pilot_symbols": list(grant.dmrs["symbols"]), "pilots": len(grant.pilot_index),
                "pilot_index_sha256": hashlib.sha256(
                    json.dumps(grant.pilot_index, separators=(",", ":")).encode("ascii")).hexdigest()}
        return result

    def duration_tc(self):
        """From the start of the schedule to the end of its last opportunity."""
        last = self.opportunities[-1]
        return last.slot_start_tc + slot_duration_tc(self.base.scs_khz, last.slot)


def slot_grant(base, slot):
    """The grant that the base grant becomes in another slot of the frame. The base slot returns the base."""
    if type(slot) is not int or not 0 <= slot < slots_per_frame(base.scs_khz):
        raise ScheduleError(f"slot {slot!r} is not a slot of the frame")
    if slot == base.slot:
        return base
    record = base.record()
    record["slot"] = {"sfn": base.sfn, "slot": slot}
    record["grant_id"] = f"{base.grant_id}.s{slot:02d}"
    try:
        return grants.build(record)
    except GrantError as error:
        raise ScheduleError(f"the grant is not legal in slot {slot}: {error}") from None


def build(record, base, cache=None):
    """Validate one pusch-grant-schedule/v1 record against its base grant and return the Schedule.

    cache, a dict, keeps the grant of each slot index between schedules of the same base grant.
    """
    if not isinstance(record, dict) or set(record) != set(FIELDS):
        raise ScheduleError(f"a schedule requires exactly {', '.join(FIELDS)}")
    if record["schema"] != SCHEMA:
        raise ScheduleError(f"unsupported schedule schema {record['schema']!r}")
    name = record["schedule_id"]
    if not isinstance(name, str) or not name or not name.isascii() or len(name) > 64 or name.strip() != name:
        raise ScheduleError("schedule_id must be 1 to 64 ASCII characters")
    if not isinstance(base, grants.Grant):
        raise ScheduleError("a schedule is built on a validated grant")
    tdd = record["tdd"]
    if not isinstance(tdd, dict) or set(tdd) != {"pattern", "ul_slots"}:
        raise ScheduleError("tdd requires exactly pattern and ul_slots")
    pattern = tdd["pattern"]
    if not isinstance(pattern, str) or not pattern or set(pattern) - set("DSU"):
        raise ScheduleError("tdd.pattern is a string of D, S and U, one letter per slot")
    per_frame = slots_per_frame(base.scs_khz)
    if per_frame % len(pattern):
        raise ScheduleError(f"a pattern of {len(pattern)} slots does not repeat in a frame of {per_frame} slots")
    ul_slots = tdd["ul_slots"]
    if not isinstance(ul_slots, list) or not ul_slots or any(type(slot) is not int for slot in ul_slots):
        raise ScheduleError("tdd.ul_slots must be a nonempty list of slot indices in the pattern")
    if ul_slots != sorted(set(ul_slots)):
        raise ScheduleError("tdd.ul_slots must increase without repetition")
    for slot in ul_slots:
        if not 0 <= slot < len(pattern) or pattern[slot] != "U":
            raise ScheduleError(f"slot {slot} is not an uplink slot of the pattern {pattern}")
    if type(record["start_sfn"]) is not int or not 0 <= record["start_sfn"] <= 1023:
        raise ScheduleError("start_sfn must be an integer from 0 to 1023")
    periods = record["periods"]
    if type(periods) is not int or periods < 1 or periods * len(ul_slots) > MAX_OPPORTUNITIES:
        raise ScheduleError(f"periods must be a positive integer with at most {MAX_OPPORTUNITIES} opportunities")
    if record["payload"] != "fixed":
        raise ScheduleError('payload must be "fixed": every opportunity carries the base grant\'s transport block')
    if base.slot % len(pattern) not in ul_slots:
        raise ScheduleError(f"the base grant's slot {base.slot} is not one of the scheduled uplink slots")

    durations = [slot_duration_tc(base.scs_khz, slot) for slot in range(per_frame)]
    if sum(durations) != FRAME_TC:
        raise ScheduleError("the slot durations do not add up to one frame")
    starts, cursor = [], 0
    for duration in durations:
        starts.append(cursor)
        cursor += duration
    slot_grants, opportunities = {}, []
    cache = {} if cache is None else cache
    for period in range(periods):
        for ul_slot in ul_slots:
            absolute = period * len(pattern) + ul_slot
            frame, slot = divmod(absolute, per_frame)
            if slot not in slot_grants:
                key = (id(base), slot)
                if key not in cache:
                    cache[key] = slot_grant(base, slot)
                slot_grants[slot] = cache[key]
            opportunities.append(Opportunity(len(opportunities), period, (record["start_sfn"] + frame) % 1024, slot,
                                             absolute, frame * FRAME_TC + starts[slot], slot_grants[slot]))
    return Schedule(json.loads(json.dumps(record)), base, slot_grants, opportunities)

"""The observation record obs/v1 and the reference reader of the ring obs-ring/v1 (Sprint 3.2, selector snapshot_ring).

The receiver publishes every observation as one record: a header of 256 bytes, an opaque host
block, the validity bits and the payload. On one host the record travels in a shared-memory ring
with one producer and one consumer, first in, first out. native/obs_ring/ holds the byte-level
definition (include/obs/record.h, include/obs/ring.h), the producer and the native consumer. This
module is the second reader. It

  * holds the same field table and hashes the same definition text, so that a reader built from
    another definition is refused when it attaches;
  * unpacks and packs a record, for a slot of the ring and for a file of an archive;
  * attaches to a ring as its consumer: it creates the cursor page that it alone writes, passes
    it with an eventfd in HELLO, maps the segment read-only, reads records in sequence order
    between two generation checks and releases them in order;
  * reads the archive that either reader writes: one file per record and consumer.json.

Standard library only, so that the reference sensor needs no native binding. It is a reference:
its reads are correct and slow. No timed run uses it, and a latency it measures is its own.
"""

import argparse
import json
import mmap
import os
from pathlib import Path
import select
import socket
import struct
import sys
import time

RECORD_SCHEMA = 1
RECORD_HEADER_BYTES = 256
MAX_RANK = 4
MAX_PLANES = 16
RING_LAYOUT = 1
SEGMENT_HEADER_BYTES = 4096
CURSOR_BYTES = 4096
SLOT_PREFIX_BYTES = 64
RING_MAGIC = b"OBSRING1"
CURSOR_MAGIC = b"OBSCURS1"
CONTROL_MAGIC = 0x4353424F
CONSUMER_SCHEMA = "obs-consumer/v1"
STAGES = {1: "O1", 2: "O2", 3: "O3"}
SOURCES = {1: "pusch_dmrs"}
LAYOUTS = {1: "ocudu_o3_fp16", 2: "ocudu_o2_comb_c64", 3: "ocudu_o1_cbf16", 4: "aerial_hest_c64", 5: "aerial_iq_fp16"}
DTYPES = {1: "<c8", 2: "<cfp16", 3: "<cbf16"}
TIME_UNITS = {1: "tc", 2: "tai_ns"}
DIGESTS = {0: None, 1: "sha256"}
EVIDENCE = {1: "audit", 2: "timed"}
STEPS = {1: "pilot_ls", 2: "frequency_interpolation", 4: "smoothing", 8: "time_interpolation"}
WAIT_MODES = {"spin": 1, "wait": 2}
HELLO, EPOCH, BYE = 1, 2, 3
BYE_REASONS = {0: "done", 1: "another record definition", 2: "the ring has a consumer",
               3: "the cursor page is not a new obs-ring/v1 cursor page",
               4: "the released count moved backwards or passed the published count", 5: "closed"}

# The header: every field as (name, struct format, count). The offsets follow from the order.
FIELDS = (
    ("schema", "I", 1), ("header_bytes", "I", 1), ("epoch", "Q", 1), ("sequence", "Q", 1), ("episode", "Q", 1),
    ("opportunity", "Q", 1), ("grant", "Q", 1), ("absolute_slot", "Q", 1), ("cell", "I", 1), ("ue", "I", 1),
    ("stage", "B", 1), ("source", "B", 1), ("layout", "B", 1), ("dtype", "B", 1), ("producer_steps", "I", 1),
    ("scale", "d", 1), ("rank", "I", 1), ("shape", "I", MAX_RANK), ("strides", "I", MAX_RANK),
    ("first_subcarrier", "I", 1), ("subcarrier_step", "I", 1), ("subcarrier_count", "I", 1), ("dmrs_symbols", "I", 1),
    ("rx_count", "H", 1), ("layer_count", "H", 1), ("antenna_map", "H", MAX_PLANES), ("mask_bytes", "I", 1),
    ("payload_bytes", "I", 1), ("host_bytes", "I", 1), ("time_unit", "B", 1), ("digest_kind", "B", 1),
    ("evidence", "B", 1), ("port", "B", 1), ("t_first_sample", "Q", 1), ("t_last_sample", "Q", 1),
    ("t_ready_ns", "q", 1), ("t_publish_ns", "q", 1), ("digest", "32s", 1), ("valid", "I", 1), ("reserved", "I", 1),
)
HEADER = struct.Struct("<" + "".join(f"{count}{code}" if count > 1 else code for _, code, count in FIELDS))
# The segment header, the slot prefix and the cursor page, by offset.
RING_FIELDS = (
    ("magic", 0, 8), ("layout", 8, 4), ("header_bytes", 12, 4), ("definition", 16, 8), ("slot_count", 24, 4),
    ("slot_bytes", 28, 4), ("record_header_bytes", 32, 4), ("host_capacity", 36, 4), ("mask_capacity", 40, 4),
    ("payload_capacity", 44, 4), ("host_offset", 48, 4), ("mask_offset", 52, 4), ("payload_offset", 56, 4),
    ("producer_pid", 60, 4), ("epoch", 64, 8), ("published", 128, 8), ("admitted", 192, 8), ("dropped", 200, 8),
    ("reclaimed", 208, 8), ("faults", 216, 8), ("wakes", 224, 8), ("window_calls", 232, 8))
PREFIX_FIELDS = (("generation", 0, 8), ("sequence", 8, 8), ("epoch", 16, 8), ("depth", 24, 4), ("record_bytes", 28, 4))
CURSOR_FIELDS = (("released", 0, 8), ("waiting", 64, 4), ("wait_mode", 68, 4), ("magic", 72, 8), ("definition", 80, 8),
                 ("consumer_pid", 88, 4), ("layout", 92, 4))
CONTROL = struct.Struct("<IIQQIIIIIIQQ")


class ObsError(ValueError):
    """A record, a ring or an archive violates the documented interface."""


class RingClosed(Exception):
    """No record follows: the producer said so or is gone, or another consumer started a new epoch."""


def _definition_of(title, fields):
    return title + "".join(f"{name}:{offset}:{size};" for name, offset, size in fields)


def _record_offsets():
    offset, table = 0, []
    for name, code, count in FIELDS:
        size = struct.calcsize("<" + code) * count
        table.append((name, offset, size))
        offset += size
    return table


RECORD_DEFINITION = _definition_of("obs/v1 little-endian 256;", _record_offsets())
RING_DEFINITION = (_definition_of("obs-ring/v1 header 4096;", RING_FIELDS)
                   + _definition_of("slot prefix 64;", PREFIX_FIELDS) + _definition_of("cursor 4096;", CURSOR_FIELDS))


def fnv1a(text, value=14695981039346656037):
    """FNV-1a over an ASCII text, 64 bits. It names a definition and is never taken of a record."""
    for byte in text.encode("ascii"):
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


DEFINITION = fnv1a(RING_DEFINITION, fnv1a(RECORD_DEFINITION))


def slots_for(window_records, burst_records):
    """Slot count of a ring: the records a window holds in place, plus one burst, plus one, as a power of two."""
    if type(window_records) is not int or type(burst_records) is not int or window_records < 0 or burst_records < 0:
        raise ObsError("a ring is sized by two non-negative record counts")
    slots = 1
    while slots < window_records + burst_records + 1:
        slots <<= 1
    if slots > 1 << 16:
        raise ObsError("a ring holds at most 65536 slots")
    return slots


# ---------------------------------------------------------------- record
def unpack_header(data):
    """The fields of a record header. Raises ObsError for another version or another size."""
    if len(data) < RECORD_HEADER_BYTES:
        raise ObsError(f"a record header holds {RECORD_HEADER_BYTES} bytes")
    values, header, position = HEADER.unpack_from(data), {}, 0
    for name, _, count in FIELDS:
        header[name] = list(values[position:position + count]) if count > 1 else values[position]
        position += count
    if header["schema"] != RECORD_SCHEMA or header["header_bytes"] != RECORD_HEADER_BYTES:
        raise ObsError(f"the record is not an obs/v1 record: schema {header['schema']}, header of "
                       f"{header['header_bytes']} bytes")
    return header


def pack_header(header):
    """The 256 bytes of a header. Fields that are missing are zero."""
    values = []
    for name, code, count in FIELDS:
        value = header.get(name, b"" if code == "32s" else ([0] * count if count > 1 else 0))
        if count > 1:
            value = list(value) + [0] * (count - len(value))
            values += value[:count]
        else:
            values.append(value)
    return HEADER.pack(*values)


def new_header(**fields):
    """A header of this version with the given fields."""
    return dict({"schema": RECORD_SCHEMA, "header_bytes": RECORD_HEADER_BYTES}, **fields)


class Record:
    """One record: the header, the host block, the validity bits and the payload, as bytes."""

    def __init__(self, header, host, mask, payload, depth=None):
        self.header, self.host, self.mask, self.payload, self.depth = header, bytes(host), bytes(mask), bytes(payload), depth
        if (len(self.host), len(self.mask), len(self.payload)) \
                != (header["host_bytes"], header["mask_bytes"], header["payload_bytes"]):
            raise ObsError("the parts of the record do not have the sizes its header names")

    @property
    def sequence(self):
        return self.header["sequence"]

    @property
    def shape(self):
        return self.header["shape"][:self.header["rank"]]

    @property
    def strides(self):
        return self.header["strides"][:self.header["rank"]]

    def is_valid(self, element):
        return bool(self.mask[element // 8] >> (element % 8) & 1)

    def to_bytes(self):
        """The record as a file or a frame carries it: its parts without padding."""
        return pack_header(self.header) + self.host + self.mask + self.payload

    def described(self):
        """The header in words, for a person or a JSON record. The digest is hexadecimal or None."""
        header = self.header
        return {
            "keys": {"epoch": header["epoch"], "sequence": header["sequence"], "episode": f"{header['episode']:016x}",
                     "opportunity": f"{header['opportunity']:016x}", "grant": f"{header['grant']:016x}"},
            "radio": {"cell": header["cell"], "ue": header["ue"], "absolute_slot": header["absolute_slot"]},
            "what": {"stage": STAGES.get(header["stage"]), "source": SOURCES.get(header["source"]), "port": header["port"],
                     "layout": LAYOUTS.get(header["layout"]), "dtype": DTYPES.get(header["dtype"]),
                     "producer_steps": [name for bit, name in STEPS.items() if header["producer_steps"] & bit],
                     "scale": header["scale"], "evidence": EVIDENCE.get(header["evidence"])},
            "shape": {"rank": header["rank"], "shape": self.shape, "strides": self.strides},
            "allocation": {"first_subcarrier": header["first_subcarrier"], "subcarrier_step": header["subcarrier_step"],
                           "subcarrier_count": header["subcarrier_count"],
                           "dmrs_symbols": [symbol for symbol in range(32) if header["dmrs_symbols"] >> symbol & 1],
                           "rx_count": header["rx_count"], "layer_count": header["layer_count"],
                           "antenna_map": header["antenna_map"][:header["rx_count"]]},
            "validity": {"mask_bytes": header["mask_bytes"], "valid": header["valid"]},
            "time": {"unit": TIME_UNITS.get(header["time_unit"]), "t_first_sample": header["t_first_sample"],
                     "t_last_sample": header["t_last_sample"], "t_ready_ns": header["t_ready_ns"],
                     "t_publish_ns": header["t_publish_ns"]},
            "payload": {"bytes": header["payload_bytes"], "digest_kind": DIGESTS.get(header["digest_kind"]),
                        "digest": header["digest"].hex() if header["digest_kind"] else None},
            "host": {"bytes": header["host_bytes"]}}


def read_record(data):
    """A record from the bytes of a file or a frame. Raises ObsError for a record that is not whole."""
    header = unpack_header(data)
    host, mask, payload = header["host_bytes"], header["mask_bytes"], header["payload_bytes"]
    if len(data) != RECORD_HEADER_BYTES + host + mask + payload:
        raise ObsError(f"the record holds {len(data)} bytes, and its header names "
                       f"{RECORD_HEADER_BYTES + host + mask + payload}")
    start = RECORD_HEADER_BYTES
    return Record(header, data[start:start + host], data[start + host:start + host + mask], data[start + host + mask:])


STAGED_HOST = struct.Struct("<IIIIQIIq")
STAGED_HOST_FIELDS = ("kind", "opportunity_sequence", "grid_index", "sfn", "grid_generation", "slot", "reserved",
                      "t_read_ns")


def staged_host(data):
    """The host block that the staged harness writes: the position of the opportunity and the ticket of its grid."""
    if len(data) != STAGED_HOST.size:
        raise ObsError(f"the host block of the staged harness holds {STAGED_HOST.size} bytes")
    block = dict(zip(STAGED_HOST_FIELDS, STAGED_HOST.unpack(data)))
    if block["kind"] != 1:
        raise ObsError(f"the host block is of the kind {block['kind']}, not of the staged harness")
    return block


def element_coordinates(layout, shape, index):
    """What one element of a payload is, by the element order of its layout.

    ocudu_o3_fp16:      [dmrs_symbol][prb][rx_port][subcarrier_in_prb]
    ocudu_o2_comb_c64:  [rx_plane][reference_symbol][pilot]
    ocudu_o1_cbf16:     [rx_port][symbol][subcarrier]
    Returns the position in the payload and the named coordinates. The subcarrier of an O3
    element counts from the first subcarrier of the allocation.
    """
    if len(index) != len(shape) or any(type(i) is not int or not 0 <= i < size for i, size in zip(index, shape)):
        raise ObsError(f"the index {list(index)} is outside a payload of shape {list(shape)}")
    position = 0
    for i, size in zip(index, shape):
        position = position * size + i
    if layout == "ocudu_o3_fp16" and len(shape) == 4:
        named = {"dmrs_symbol": index[0], "rx_port": index[2], "subcarrier": 12 * index[1] + index[3]}
    elif layout == "ocudu_o2_comb_c64" and len(shape) == 3:
        named = {"rx_plane": index[0], "reference_symbol": index[1], "pilot": index[2]}
    elif layout == "ocudu_o1_cbf16" and len(shape) == 3:
        named = {"rx_port": index[0], "symbol": index[1], "subcarrier": index[2]}
    else:
        raise ObsError(f"no element order is defined here for the layout {layout!r} with {len(shape)} axes")
    return dict(named, element=position)


# ---------------------------------------------------------------- ring
def _address(path):
    """The address of a control socket. A path too long for a socket address is reached through its directory."""
    path = str(path)
    if len(path.encode()) < 100:
        return path, None
    directory = os.open(os.path.dirname(path) or ".", os.O_RDONLY | os.O_DIRECTORY)
    return f"/proc/self/fd/{directory}/{os.path.basename(path)}", directory


def _shared_page():
    """A page that only this process can write: a read-write mapping and a read-only descriptor for the peer."""
    name = f"/dev/shm/obs-cursor-{os.getpid()}-{time.monotonic_ns():x}"
    try:
        writable = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            readonly = os.open(name, os.O_RDONLY)
        finally:
            os.unlink(name)
    except OSError:
        writable = os.memfd_create("obs-cursor")
        readonly = os.open(f"/proc/self/fd/{writable}", os.O_RDONLY)
    os.ftruncate(writable, CURSOR_BYTES)
    page = mmap.mmap(writable, CURSOR_BYTES, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(writable)
    return page, readonly


class Consumer:
    """The consumer of one ring. Use it as a context manager, or call close()."""

    def __init__(self, socket_path, wait="wait", connect_timeout=5.0, definition=DEFINITION):
        if wait not in WAIT_MODES:
            raise ObsError('the wait mode is "spin" or "wait"')
        self.wait, self.definition = wait, definition
        self.read = self.stale = self.released = self.skipped = self.sleeps = 0
        self.position, self.done = 0, False
        self._cursor, cursor_fd = _shared_page()
        struct.pack_into("<II8sQII", self._cursor, 64, 0, WAIT_MODES[wait], CURSOR_MAGIC, definition, os.getpid(),
                         RING_LAYOUT)
        self._event = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._socket = self._segment = None
        deadline = time.monotonic() + connect_timeout
        try:
            reply, ring_fd = self._attach(socket_path, cursor_fd, deadline, connect_timeout)
        except BaseException:
            self.close(say_bye=False)
            raise
        finally:
            os.close(cursor_fd)
        try:
            size = os.fstat(ring_fd).st_size
            if size != reply["segment_bytes"]:
                raise ObsError("the segment does not have the size the producer named")
            # Read-only: this process cannot write what the producer published.
            self._segment = mmap.mmap(ring_fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
        except (OSError, ValueError) as error:
            self.close(say_bye=False)
            raise ObsError(f"the segment of the ring cannot be mapped read-only: {error}") from None
        finally:
            os.close(ring_fd)
        self.ring = {name: (self._segment[offset:offset + size] if name == "magic"
                            else int.from_bytes(self._segment[offset:offset + size], "little"))
                     for name, offset, size in RING_FIELDS[:14]}
        ring = self.ring
        if ring["magic"] != RING_MAGIC or ring["layout"] != RING_LAYOUT or ring["header_bytes"] != SEGMENT_HEADER_BYTES \
                or ring["record_header_bytes"] != RECORD_HEADER_BYTES or ring["slot_count"] & (ring["slot_count"] - 1) \
                or not ring["slot_count"] \
                or len(self._segment) != SEGMENT_HEADER_BYTES + ring["slot_count"] * ring["slot_bytes"]:
            self.close(say_bye=False)
            raise ObsError("the segment is not an obs-ring/v1 segment of the size the producer named")
        if ring["definition"] != definition:
            self.close(say_bye=False)
            raise ObsError(f"the ring refuses this consumer: it was built from the record definition {definition:016x}, "
                           f"and the segment from {ring['definition']:016x}")
        self.epoch = reply["epoch"]

    def _attach(self, socket_path, cursor_fd, deadline, connect_timeout):
        hello = CONTROL.pack(CONTROL_MAGIC, HELLO, self.definition, 0, os.getpid(), WAIT_MODES[self.wait], 0, 0, 0, 0, 0, 0)
        while True:
            try:
                address, directory = _address(socket_path)
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise ObsError(f"no producer opened the control channel {socket_path} within "
                                   f"{int(connect_timeout * 1000)} ms") from None
                time.sleep(0.001)  # the producer has not created the directory of its channel yet
                continue
            link = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            try:
                link.connect(address)
            except (FileNotFoundError, ConnectionRefusedError, BlockingIOError):
                link.close()  # the producer may not have opened its channel yet
            except OSError as error:
                link.close()
                raise ObsError(f"cannot connect to the control channel {socket_path}: {error}") from None
            else:
                socket.send_fds(link, [hello], [cursor_fd, self._event])
                link.settimeout(max(deadline - time.monotonic(), 0.001))
                try:
                    data, fds, _, _ = socket.recv_fds(link, CONTROL.size, 2)
                except (TimeoutError, OSError):
                    data, fds = b"", []
                if len(data) == CONTROL.size:
                    names = ("magic", "type", "definition", "epoch", "pid", "mode", "reason", "slot_count", "slot_bytes",
                             "reserved", "segment_bytes", "peer_definition")
                    reply = dict(zip(names, CONTROL.unpack(data)))
                    if reply["magic"] == CONTROL_MAGIC and reply["type"] == EPOCH and len(fds) == 1:
                        link.settimeout(None)
                        self._socket = link
                        return reply, fds[0]
                    for fd in fds:
                        os.close(fd)
                    link.close()
                    if reply["type"] == BYE and reply["reason"] == 1:
                        raise ObsError(f"the ring refuses this consumer: it was built from the record definition "
                                       f"{self.definition:016x}, and the producer from {reply['peer_definition']:016x}")
                    if reply["type"] != BYE or reply["reason"] != 2:
                        raise ObsError("the ring refuses this consumer: "
                                       + BYE_REASONS.get(reply["reason"], "its answer is not an EPOCH message"))
                else:
                    for fd in fds:
                        os.close(fd)
                    link.close()
            finally:
                if directory is not None:
                    os.close(directory)
            if time.monotonic() >= deadline:
                raise ObsError(f"no producer accepted this consumer on {socket_path} within "
                               f"{int(connect_timeout * 1000)} ms")
            time.sleep(0.001)

    # -- counts of the segment, read in place
    def _count(self, offset):
        return int.from_bytes(self._segment[offset:offset + 8], "little")

    @property
    def published(self):
        return self._count(128)

    def counts(self):
        return {"read": self.read, "stale": self.stale, "released": self.released, "skipped": self.skipped,
                "sleeps": self.sleeps}

    def producer_done(self):
        """Read the control channel without blocking. True once the producer said BYE or is gone."""
        if self.done or self._socket is None:
            return True
        try:
            data = self._socket.recv(CONTROL.size, socket.MSG_DONTWAIT)
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            data = b""
        if not data or (len(data) == CONTROL.size and CONTROL.unpack(data)[1] == BYE):
            self.done = True
        return self.done

    def next(self, timeout=1.0):
        """The next record in sequence order, or None when none arrived in time.

        Raises RingClosed once the producer said that no record follows and every record was
        read, or when another consumer started a new epoch. A slot that was rewritten is a
        counted stale read, and None is returned for it.
        """
        deadline = time.monotonic() + timeout
        while True:
            if self._count(64) != self.epoch:
                raise RingClosed
            if self.position < self.published:
                break
            if self.wait == "wait":
                struct.pack_into("<I", self._cursor, 64, 1)
                if self.position < self.published:
                    struct.pack_into("<I", self._cursor, 64, 0)
                    break
                left = deadline - time.monotonic()
                if left > 0:
                    self.sleeps += 1
                    waits = [self._event] if self.done or self._socket is None else [self._event, self._socket]
                    select.select(waits, [], [], left)
                struct.pack_into("<I", self._cursor, 64, 0)
                try:
                    os.read(self._event, 8)
                except BlockingIOError:
                    pass
            elif time.monotonic() < deadline:
                continue
            if self.position < self.published:
                break
            if time.monotonic() >= deadline or self.producer_done():
                finished = self.producer_done()
                if self.position < self.published:
                    break
                if finished:
                    raise RingClosed
                return None
        ring, which = self.ring, self.position
        base = SEGMENT_HEADER_BYTES + (which & (ring["slot_count"] - 1)) * ring["slot_bytes"]
        self.position += 1
        generation, sequence, epoch, depth, _ = struct.unpack_from("<QQQII", self._segment, base)
        if generation & 1 or sequence != which or epoch != self.epoch:
            self.stale += 1
            return None
        data = self._segment[base + SLOT_PREFIX_BYTES:base + ring["slot_bytes"]]
        try:
            header = unpack_header(data)
            sizes = (header["host_bytes"], header["mask_bytes"], header["payload_bytes"])
            if sizes[0] > ring["host_capacity"] or sizes[1] > ring["mask_capacity"] or sizes[2] > ring["payload_capacity"]:
                raise ObsError("a part of the record is larger than its slot")
            offsets = [ring[name] - SLOT_PREFIX_BYTES for name in ("host_offset", "mask_offset", "payload_offset")]
            record = Record(header, *(data[offset:offset + size] for offset, size in zip(offsets, sizes)), depth=depth)
        except ObsError:
            self.stale += 1
            return None
        # The second generation check: the copy above is the record only if the slot was not rewritten.
        if struct.unpack_from("<Q", self._segment, base)[0] != generation:
            self.stale += 1
            return None
        self.read += 1
        return record

    def release(self, sequence=None):
        """The work on every record below sequence is complete. Default: on every record read so far."""
        upto = self.position if sequence is None else min(sequence, self.position)
        now = struct.unpack_from("<Q", self._cursor, 0)[0]
        if upto > now:
            self.released += upto - now
            struct.pack_into("<Q", self._cursor, 0, upto)

    def close(self, say_bye=True):
        if self._socket is not None:
            if say_bye:
                try:
                    self._socket.send(CONTROL.pack(CONTROL_MAGIC, BYE, self.definition, getattr(self, "epoch", 0),
                                                   os.getpid(), 0, 0, 0, 0, 0, 0, 0))
                except OSError:
                    pass
            self._socket.close()
            self._socket = None
        if self._event is not None:
            os.close(self._event)
            self._event = None
        for name in ("_segment", "_cursor"):
            mapping = getattr(self, name, None)
            if mapping is not None:
                mapping.close()
                setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------- archive
def record_name(epoch, sequence):
    return f"{epoch:04d}-{sequence:08d}.obs"


def tail(socket_path, output_dir=None, wait="wait", connect_timeout=10.0, idle_timeout=30.0, hold_us=0, window=0,
         on_record=None):
    """Attach, read every record until the producer says that none follows, and keep them.

    Returns the report that is also written as consumer.json. With output_dir every record is
    written as one file, after the last record was read. on_record, when given, is called with
    each record after its read and before its release: the work of a reference sensor.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.is_dir() or (output_dir / "consumer.json").exists():
            raise ObsError(f"the output directory {output_dir} does not exist or holds a result already")
    kept, rows, ended = [], [], "idle"
    with Consumer(socket_path, wait, connect_timeout) as reader:
        attached, last = time.monotonic_ns(), time.monotonic()
        while True:
            try:
                record = reader.next(0.05)
            except RingClosed:
                ended = "closed"
                break
            if record is None:
                if time.monotonic() - last > idle_timeout:
                    break
                continue
            now, last = time.monotonic_ns(), time.monotonic()
            if hold_us:
                until = time.monotonic_ns() + 1000 * hold_us
                while time.monotonic_ns() < until:
                    pass
            if on_record is not None:
                on_record(record)
            header = record.header
            kept.append(record)
            rows.append({"epoch": header["epoch"], "sequence": header["sequence"],
                         "opportunity": f"{header['opportunity']:016x}", "depth": record.depth,
                         "bytes": len(record.to_bytes()), "t_publish_ns": header["t_publish_ns"], "t_read_ns": now,
                         "delay_ns": now - header["t_publish_ns"], "intact": True,
                         "file": True if output_dir is not None else None})
            reader.release(max(record.sequence + 1 - window, 0))
        reader.release()
        report = {"schema": CONSUMER_SCHEMA, "reader": "reference", "definition": f"{DEFINITION:016x}", "wait": wait,
                  "epoch": reader.epoch, "pid": os.getpid(), "cpu": -1, "hold_us": hold_us, "window": window,
                  "slot_count": reader.ring["slot_count"], "slot_bytes": reader.ring["slot_bytes"],
                  "attached_ns": attached, "ended": ended,
                  "counts": dict(reader.counts(), kept=len(kept), not_kept=0, system_calls=None), "records": rows}
    if output_dir is not None:
        for record in kept:
            (output_dir / record_name(record.header["epoch"], record.sequence)).write_bytes(record.to_bytes())
        (output_dir / "consumer.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    return report


class Archive:
    """What a consumer kept of one epoch: its report and its records, by sequence."""

    def __init__(self, directory, report, records):
        self.directory, self.report, self.records = directory, report, records
        self.by_opportunity = {record.header["opportunity"]: record for record in records}


def open_archive(directory):
    """Verify the archive of a consumer and return it.

    Every record that the report lists as kept has one file whose header names the epoch, the
    sequence and the opportunity of its line and whose size is the one the line names. The
    sequences of the lines rise by one. The directory holds nothing else.
    """
    directory = Path(directory)
    try:
        report = json.loads((directory / "consumer.json").read_text(encoding="ascii"))
    except (OSError, ValueError) as error:
        raise ObsError(f"{directory} holds no readable consumer.json: {error}") from None
    if not isinstance(report, dict) or report.get("schema") != CONSUMER_SCHEMA or not isinstance(report.get("records"), list):
        raise ObsError(f"{directory / 'consumer.json'} is not an {CONSUMER_SCHEMA} report")
    if report.get("definition") != f"{DEFINITION:016x}":
        raise ObsError(f"the archive was written with the record definition {report.get('definition')}, and this reader "
                       f"has {DEFINITION:016x}")
    records, names, expected = [], {"consumer.json"}, None
    for line in report["records"]:
        if expected is not None and line["sequence"] != expected:
            raise ObsError(f"the archive lists the sequence {line['sequence']} where {expected} follows: the ring is "
                           "first in, first out")
        expected = line["sequence"] + 1
        if not line.get("file"):
            continue
        name = record_name(line["epoch"], line["sequence"])
        try:
            record = read_record((directory / name).read_bytes())
        except OSError as error:
            raise ObsError(f"the archive lists a record it does not hold: {error}") from None
        header = record.header
        if (header["epoch"], header["sequence"], f"{header['opportunity']:016x}") \
                != (line["epoch"], line["sequence"], line["opportunity"]) or len(record.to_bytes()) != line["bytes"] \
                or header["t_publish_ns"] != line["t_publish_ns"]:
            raise ObsError(f"the record {name} differs from its line of the report")
        names.add(name)
        records.append(record)
    extra = sorted(path.name for path in directory.iterdir() if path.name not in names)
    if extra:
        raise ObsError(f"the archive holds {extra[0]}, which its report does not list")
    return Archive(directory, report, records)


def percentile(values, share):
    """The value at a share of a sorted copy of values: the nearest rank at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(int(share * (len(ordered) - 1)), len(ordered) - 1)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reference reader of the observation ring obs-ring/v1.")
    commands = parser.add_subparsers(dest="command", required=True)
    follow = commands.add_parser("tail", help="attach to a ring, read every record and keep it")
    follow.add_argument("--socket", required=True, type=Path, help="control channel of the producer's ring")
    follow.add_argument("--output-dir", type=Path, help="existing, empty directory for the records and consumer.json")
    follow.add_argument("--wait", choices=sorted(WAIT_MODES), default="wait")
    follow.add_argument("--connect-ms", type=int, default=10000)
    follow.add_argument("--idle-ms", type=int, default=30000)
    follow.add_argument("--hold-us", type=int, default=0, help="work on each record for this long before its release")
    follow.add_argument("--window", type=int, default=0, help="records to hold in place")
    show = commands.add_parser("show", help="verify an archive and print one record")
    show.add_argument("directory", type=Path)
    show.add_argument("--sequence", type=int, default=0)
    commands.add_parser("definition", help="print the hash and the text of the record and ring definition")
    args = parser.parse_args(argv)
    try:
        if args.command == "definition":
            print(f"{DEFINITION:016x}\n{RECORD_DEFINITION}\n{RING_DEFINITION}")
            return 0
        if args.command == "tail":
            report = tail(args.socket, args.output_dir, args.wait, args.connect_ms / 1000, args.idle_ms / 1000,
                          args.hold_us, args.window)
            counts = report["counts"]
            print(f"obs: epoch {report['epoch']}, {report['wait']}, {counts['read']} records read, {counts['stale']} "
                  f"stale, {counts['released']} released, ended {report['ended']}")
            return 0
        archive = open_archive(args.directory)
        found = [record for record in archive.records if record.sequence == args.sequence]
        if not found:
            raise ObsError(f"the archive holds no record with the sequence {args.sequence}")
        record = found[0]
        shown = dict(record.described(), records=len(archive.records), reader=archive.report["reader"],
                     wait=archive.report["wait"], file=record_name(record.header["epoch"], record.sequence))
        try:
            shown["staged_host"] = staged_host(record.host)
        except ObsError:
            shown["staged_host"] = None
        print(json.dumps(shown, indent=2))
        return 0
    except ObsError as error:
        parser.exit(2, f"obs: {error}\n")


if __name__ == "__main__":
    sys.exit(main())

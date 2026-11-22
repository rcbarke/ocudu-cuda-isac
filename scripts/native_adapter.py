"""Build the project's native adapter out of tree against the accepted OCUDU build.

The adapter must compile and link exactly like pusch_e2e_pipeline_test of the pinned source, so
that it runs the same OCUDU objects the Sprint 1 gates accepted. This script reads that test's
compile flags (flags.make) and link line (link.txt) from the accepted build directory, writes them
to an import file for native/ocudu_adapter/CMakeLists.txt, builds, runs the adapter's own tests and
leaves a receipt. It reads the OCUDU source and build and never writes into either.

With --o3 (Sprint 5.2) the adapter is built a second time, into a directory of its own, against the
build of the patched receiver that 'native_ocudu.py build --patched' produced. Only that build has
the route to the receiver's own channel estimate compiled in (ISAC_O3_ROUTE), and its receipt
carries the identity of the patch series. The build against the accepted Sprint 1.1 build is
unchanged and has no such route.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

try:  # imported as scripts.native_adapter by the tests, run as a file otherwise
    from scripts import receiver_patch
except ImportError:
    import receiver_patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "native/ocudu_adapter"
# The observation ring is built with the adapter and linked by its producer glue (Sprint 3.2).
RING = ROOT / "native/obs_ring"
PIN_TARGET = "pusch_e2e_pipeline_test"
PIN_TARGET_DIR = Path("tests/integrationtests/phy/upper/channel_processors")
DEFAULT_BUILD = ROOT / "build/ocudu-adapter"
# Sprint 5.2: the adapter with the O3 route, and the build of the patched receiver it links.
DEFAULT_BUILD_O3 = ROOT / "build/ocudu-adapter-o3"
DEFAULT_OCUDU_O3 = ROOT / "build/ocudu-spark-o3"
RECEIPT = "isac-adapter.json"
RECEIPT_SCHEMA = "isac-adapter-build/v1"
OCUDU_RECEIPT = "isac-native.json"
EXECUTABLES = {"grant_fixture": Path("fixtures/grant_fixture"), "grid_apply": Path("replay/grid_apply"),
               "replay_pusch": Path("receiver/replay_pusch"), "pilot_ls": Path("observer/pilot_ls"),
               "obs_tail": Path("obs_ring/obs_tail"), "obs_feed": Path("obs_ring/obs_feed")}
GENERATOR = "Unix Makefiles"


class AdapterError(ValueError):
    """The adapter cannot be built or its build cannot be trusted. The message says what to do."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locked_source():
    """The OCUDU source directory. ext/lock.json is the only owner of that path."""
    for item in json.loads((ROOT / "ext/lock.json").read_text())["dependencies"]:
        if item.get("name") == "ocudu":
            return ROOT / item["path"]
    raise AdapterError("ext/lock.json has no dependency named 'ocudu'")


def cache_values(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"([A-Za-z0-9_.+-]+):[A-Z]+=(.*)$", line)
        if match:
            values[match[1]] = match[2]
    return values


def accepted_build(bin_dir=None):
    """The OCUDU build that Sprint 1.1 published into the environment, read from its link."""
    link = (bin_dir or ROOT / "isac/bin") / PIN_TARGET
    target = link.resolve() if link.is_symlink() else None
    suffix = PIN_TARGET_DIR / PIN_TARGET
    if target is None or not str(target).endswith("/" + str(suffix)):
        raise AdapterError(f"{link} is not a link into an OCUDU build. Run 'make accept-s1.1' first, or name the "
                           "build with --ocudu-build-dir.")
    return Path(str(target)[:-len(str(suffix)) - 1])


def parse_flags(text):
    """Compile flags of one target from its flags.make: definitions, include directories and options."""
    found = {}
    for line in text.splitlines():
        match = re.match(r"(CXX_DEFINES|CXX_INCLUDES|CXX_FLAGS)\s*=(.*)$", line)
        if match:
            found[match[1]] = shlex.split(match[2])
    if set(found) != {"CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"}:
        raise AdapterError("flags.make does not hold CXX_DEFINES, CXX_INCLUDES and CXX_FLAGS")
    includes, system, words = [], [], iter(found["CXX_INCLUDES"])
    for word in words:
        if word == "-isystem":
            system.append(next(words, ""))
        elif word.startswith("-isystem"):
            system.append(word[len("-isystem"):])
        elif word.startswith("-I"):
            includes.append(word[2:])
        else:
            raise AdapterError(f"unexpected include option {word!r} in flags.make")
    if not all(includes + system):
        raise AdapterError("flags.make names an empty include directory")
    definitions = []
    for word in found["CXX_DEFINES"]:
        if not word.startswith("-D") or len(word) == 2:
            raise AdapterError(f"unexpected definition {word!r} in flags.make")
        definitions.append(word[2:])
    # A warning in adapter code is reported by the build and must not stop it on a new compiler.
    options = [word for word in found["CXX_FLAGS"] if word != "-Werror"]
    return {"definitions": definitions, "includes": includes, "system_includes": system, "options": options}


def parse_link(text, target_dir):
    """Split the link line of one target into the compiler, link options, archives and other items."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise AdapterError("link.txt does not hold exactly one link command")
    words = shlex.split(lines[0])
    if "-o" not in words or words.index("-o") + 1 >= len(words):
        raise AdapterError("link.txt has no output option")
    split = words.index("-o")
    if Path(words[split + 1]).name != PIN_TARGET:
        raise AdapterError(f"link.txt links {words[split + 1]!r}, expected {PIN_TARGET}")
    compiler, before, after = words[0], words[1:split], words[split + 2:]
    options = [word for word in before
               if not word.endswith(".o") and not word.startswith("-D")
               and not (word.startswith("-W") and not word.startswith("-Wl,"))]
    archives, items = [], []
    for word in after:
        if word.endswith(".o"):
            raise AdapterError(f"link.txt links the object {word!r} after the output; the import cannot reuse it")
        if word.endswith(".a"):
            path = Path(word) if os.path.isabs(word) else Path(os.path.normpath(target_dir / word))
            if not path.is_file():
                raise AdapterError(f"the OCUDU build has no {path}. Build it with 'make native-build'.")
            if str(path) not in archives:
                archives.append(str(path))
        else:
            items.append(word)
    if not archives:
        raise AdapterError("link.txt names no static library of the OCUDU build")
    return {"compiler": compiler, "options": options, "archives": archives, "items": items}


def cmake_list(name, values):
    """One CMake set() command. Each value becomes one quoted argument."""
    def quote(value):
        return '"' + re.sub(r'([\\"$;])', r"\\\1", str(value)) + '"'
    body = "".join(f"\n  {quote(value)}" for value in values)
    return f"set({name}{body})\n"


def pin_import(ocudu_build, source):
    """Read the pinned test's compile and link recipe from a build and return (text, facts)."""
    ocudu_build, source = Path(ocudu_build).resolve(), Path(source).resolve()
    cache_path = ocudu_build / "CMakeCache.txt"
    if not cache_path.is_file():
        raise AdapterError(f"{ocudu_build} is not a configured OCUDU build (no CMakeCache.txt)")
    cache = cache_values(cache_path)
    if cache.get("CMAKE_GENERATOR") != GENERATOR:
        raise AdapterError(f"the OCUDU build uses the generator {cache.get('CMAKE_GENERATOR')!r}. The import reads "
                           f"flags.make and link.txt, which only {GENERATOR!r} writes.")
    home = cache.get("CMAKE_HOME_DIRECTORY")
    if not home or Path(home).resolve() != source:
        raise AdapterError(f"the OCUDU build was configured from {home}, not from the locked source {source}")
    target_dir = ocudu_build / PIN_TARGET_DIR
    recipe = target_dir / "CMakeFiles" / f"{PIN_TARGET}.dir"
    for name in ("flags.make", "link.txt"):
        if not (recipe / name).is_file():
            raise AdapterError(f"the OCUDU build has no {recipe / name}. Build with BUILD_TESTING=ON "
                               "('make native-build').")
    flags = parse_flags((recipe / "flags.make").read_text())
    link = parse_link((recipe / "link.txt").read_text(), target_dir)
    # The cache names the C++ compiler of the build. The link line starts with it, except where
    # another language's launcher links the target.
    compiler = cache.get("CMAKE_CXX_COMPILER") or link["compiler"]
    test_source = source / PIN_TARGET_DIR
    if not (test_source / "pxsch_bler_test_factories.h").is_file():
        raise AdapterError(f"the locked source has no test helpers in {test_source}")
    text = ("# Written by scripts/native_adapter.py from the accepted OCUDU build. Do not edit.\n"
            f"# OCUDU build: {ocudu_build}\n# Target: {PIN_TARGET}\n"
            + cmake_list("OCUDU_PIN_CXX_COMPILER", [compiler])
            + cmake_list("OCUDU_PIN_CXX_FLAGS", flags["options"])
            + cmake_list("OCUDU_PIN_CXX_DEFINES", flags["definitions"])
            + cmake_list("OCUDU_PIN_INCLUDES", flags["includes"])
            + cmake_list("OCUDU_PIN_SYSTEM_INCLUDES", flags["system_includes"])
            + cmake_list("OCUDU_PIN_TEST_SOURCE_DIR", [test_source])
            + cmake_list("OCUDU_PIN_LINK_OPTIONS", link["options"])
            + cmake_list("OCUDU_PIN_ARCHIVES", link["archives"])
            + cmake_list("OCUDU_PIN_LINK_ITEMS", link["items"]))
    facts = {"compiler": compiler, "linker_driver": link["compiler"], "cuda": "ENABLE_CUDA" in flags["definitions"],
             "options": flags["options"], "definitions": flags["definitions"], "archives": len(link["archives"]),
             "link_items": link["items"], "build_type": cache.get("CMAKE_BUILD_TYPE"),
             "flags_make_sha256": digest(recipe / "flags.make"), "link_txt_sha256": digest(recipe / "link.txt"),
             "pin_test_sha256": digest(target_dir / PIN_TARGET) if (target_dir / PIN_TARGET).is_file() else None}
    return text, facts


def sources():
    """Every file of the adapter tree and of the ring it links, with its hash. The receipt binds the executables to them."""
    return {str(path.relative_to(ROOT)): digest(path)
            for tree in (ADAPTER, RING)
            for path in sorted(tree.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}


def record(args, run_dir, log_name, cwd=None):
    """Run one tool with its output on the screen and in a log. Return its exit status."""
    command = shlex.join(str(arg) for arg in args)
    with (run_dir / "commands.sh").open("a") as history:
        history.write(command + "\n")
    print(f"{command}\n  log: {run_dir / log_name}", flush=True)
    environment = {key: value for key, value in os.environ.items() if key != "CLICOLOR_FORCE"}
    with (run_dir / log_name).open("wb") as log:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, env=environment)
        with process.stdout:
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line.decode("utf-8", "replace"))
                sys.stdout.flush()
        return process.wait()


def patched_receiver(ocudu_build):
    """The source tree and the patch identity of a build of the patched receiver, from its receipt.

    The receipt must name the series of patches/ocudu/ as it is now, and the tree it was built from
    must still be the pinned commit with exactly that series applied.
    """
    path = Path(ocudu_build) / OCUDU_RECEIPT
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError):
        raise AdapterError(f"{ocudu_build} has no {OCUDU_RECEIPT}: it is not a build of the patched receiver. Run: "
                           "python scripts/native_ocudu.py build --patched") from None
    patch = receipt.get("patch")
    if not isinstance(patch, dict):
        raise AdapterError(f"{ocudu_build} is a build of the unpatched receiver: its receipt names no patch. The O3 route "
                           "needs the build of: python scripts/native_ocudu.py build --patched")
    try:
        series = receiver_patch.load_series()
        if not receiver_patch.same_series(patch, receiver_patch.identity(series)):
            raise AdapterError(f"{ocudu_build} was built with another patch series than patches/ocudu/ holds now. "
                               "Rebuild it: python scripts/native_ocudu.py build --patched")
        current = receiver_patch.verify_tree(receipt["source"], series)
    except receiver_patch.PatchError as error:
        raise AdapterError(str(error)) from None
    return Path(receipt["source"]), current


def build(args):
    o3 = bool(getattr(args, "o3", False))
    patch = None
    if o3:
        ocudu_build = Path(args.ocudu_build_dir).resolve() if args.ocudu_build_dir else DEFAULT_OCUDU_O3
        source, patch = patched_receiver(ocudu_build)
    else:
        source = locked_source()
        ocudu_build = Path(args.ocudu_build_dir).resolve() if args.ocudu_build_dir else accepted_build()
    build_dir = Path(args.build_dir).resolve() if args.build_dir else (DEFAULT_BUILD_O3 if o3 else DEFAULT_BUILD)
    for parent in (source.resolve(), locked_source().resolve(), ocudu_build):
        if build_dir == parent or parent in build_dir.parents:
            raise AdapterError(f"the adapter build directory {build_dir} lies inside {parent}")
    ocudu_receipt = ocudu_build / OCUDU_RECEIPT
    if not ocudu_receipt.is_file() and not args.no_receipt:
        raise AdapterError(f"{ocudu_build} has no {OCUDU_RECEIPT}: it is not a build that 'make accept-s1.1' "
                           "produced. Use --no-receipt only on a development host.")
    if not o3 and ocudu_receipt.is_file():
        # The adapter without the route is the control. It is never linked against the patched receiver.
        try:
            named = json.loads(ocudu_receipt.read_text()).get("patch")
        except ValueError:
            named = None
        if named:
            raise AdapterError(f"{ocudu_build} is the build of the patched receiver. Build the adapter against it with "
                               "--o3, into its own directory.")
    for tool in ("cmake", "ctest"):
        if not shutil.which(tool):
            raise AdapterError(f"{tool} is not on PATH")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else build_dir / "evidence"
    run_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    text, facts = pin_import(ocudu_build, source)
    import_file = build_dir / "ocudu-pin.cmake"
    import_file.write_text(text)
    shutil.copyfile(import_file, run_dir / "ocudu-pin.cmake")
    receipt_path = build_dir / RECEIPT
    receipt_path.unlink(missing_ok=True)  # a failed build leaves no receipt of an earlier one

    stages = (("configure", ["cmake", "-S", ADAPTER, "-B", build_dir, f"-DOCUDU_PIN_IMPORT={import_file}",
                             f"-DISAC_O3_ROUTE={'ON' if o3 else 'OFF'}", "-G", GENERATOR], None),
              ("build", ["cmake", "--build", build_dir, f"-j{args.jobs}"], None),
              ("test", ["ctest", "--output-on-failure"], build_dir))
    for stage, command, cwd in stages:
        status = record(command, run_dir, f"adapter-{stage}.log", cwd)
        if status:
            raise AdapterError(f"the adapter {stage} step exited with status {status} (see {run_dir / ('adapter-' + stage + '.log')})")
    tests = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)",
                      (run_dir / "adapter-test.log").read_text(errors="replace"))
    if not tests or int(tests[3]) == 0 or int(tests[2]) != 0:
        raise AdapterError("the adapter's own tests did not all run and pass (see adapter-test.log)")
    executables = {}
    for name, relative in EXECUTABLES.items():
        path = build_dir / relative
        if not path.is_file():
            raise AdapterError(f"the build produced no {path}")
        executables[name] = {"path": str(path), "sha256": digest(path)}
    warnings = len(re.findall(r"\bwarning:", (run_dir / "adapter-build.log").read_text(errors="replace")))
    receipt = {
        "schema": RECEIPT_SCHEMA, "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "adapter_build": str(build_dir), "build_evidence": str(run_dir),
        "ocudu_build": str(ocudu_build), "ocudu_source": str(source.resolve()),
        "ocudu_receipt_sha256": digest(ocudu_receipt) if ocudu_receipt.is_file() else None,
        "import_sha256": digest(import_file), "pin": facts, "sources": sources(), "executables": executables,
        "tests_run": int(tests[3]), "compiler_warnings": warnings,
    }
    if o3:
        # What the route was compiled against. A build without these two keys has no O3 route. The
        # tool is asked itself: it answers from the accessor of the receiver patch and starts no CUDA work.
        said = subprocess.run([executables["pilot_ls"]["path"], "--route"], capture_output=True, text=True, timeout=120,
                              check=False)
        (run_dir / "adapter-route.log").write_text(said.stdout + said.stderr)
        route = re.search(r"pilot_ls route: O3 from the patched receiver, tap interface version (\d+)", said.stdout)
        if said.returncode != 0 or not route:
            raise AdapterError("pilot_ls of this build does not report the O3 route (see adapter-route.log)")
        receipt["o3_route"] = {"compiled": True, "tap_interface_version": int(route[1])}
        receipt["patch"] = patch
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(receipt_path, run_dir / RECEIPT)
    print(f"Adapter build: {build_dir}\nReceipt: {receipt_path}\nEvidence: {run_dir}")
    return 0


def verified_receipt(build_dir):
    """Read a receipt and require that the executables, the adapter sources and the OCUDU build still match it."""
    build_dir = Path(build_dir).resolve()
    path = build_dir / RECEIPT
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError):
        raise AdapterError(f"no adapter receipt in {build_dir}. Run: python scripts/native_adapter.py build") from None
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise AdapterError(f"{path} is not a {RECEIPT_SCHEMA} receipt")
    for name, entry in receipt["executables"].items():
        if not Path(entry["path"]).is_file() or digest(entry["path"]) != entry["sha256"]:
            raise AdapterError(f"the executable {name} differs from the adapter receipt. Rebuild the adapter.")
    if sources() != receipt["sources"]:
        raise AdapterError("the adapter sources changed after the build. Rebuild the adapter.")
    ocudu_receipt = Path(receipt["ocudu_build"]) / OCUDU_RECEIPT
    current = digest(ocudu_receipt) if ocudu_receipt.is_file() else None
    if current != receipt["ocudu_receipt_sha256"]:
        raise AdapterError("the OCUDU build was rebuilt after the adapter. Rebuild the adapter.")
    recipe = Path(receipt["ocudu_build"]) / PIN_TARGET_DIR / "CMakeFiles" / f"{PIN_TARGET}.dir"
    if digest(recipe / "link.txt") != receipt["pin"]["link_txt_sha256"] \
            or digest(recipe / "flags.make") != receipt["pin"]["flags_make_sha256"]:
        raise AdapterError("the OCUDU build was reconfigured after the adapter. Rebuild the adapter.")
    if receipt.get("patch") or receipt.get("o3_route"):
        # The O3 adapter: the patched tree and the series must still be what it was built against.
        _, current = patched_receiver(receipt["ocudu_build"])
        if not receiver_patch.same_series(receipt.get("patch"), current) or not (receipt.get("o3_route") or {}).get("compiled"):
            raise AdapterError("the patch series changed after the O3 adapter was built. Rebuild the patched receiver "
                               "and the adapter.")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build", help="configure, build and test the adapter; write its receipt")
    make.add_argument("--ocudu-build-dir", help="OCUDU build to link against (default: the one behind isac/bin)")
    make.add_argument("--build-dir", help="adapter build directory (default: build/ocudu-adapter, or "
                      "build/ocudu-adapter-o3 with --o3)")
    make.add_argument("--o3", action="store_true",
                      help="build against the patched receiver (default: build/ocudu-spark-o3) with the O3 route")
    make.add_argument("--run-dir", help="directory for the logs (default: <build-dir>/evidence)")
    make.add_argument("--jobs", type=int, default=4)
    make.add_argument("--no-receipt", action="store_true",
                      help="development host: accept an OCUDU build without a Sprint 1.1 receipt")
    check = commands.add_parser("verify", help="check a build against its receipt and print the receipt")
    check.add_argument("--build-dir", help="adapter build directory (default as for build)")
    check.add_argument("--o3", action="store_true", help="the adapter with the O3 route, and require that it has one")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            return build(args)
        receipt = verified_receipt(args.build_dir or (DEFAULT_BUILD_O3 if args.o3 else DEFAULT_BUILD))
        if args.o3 and not (receipt.get("o3_route") or {}).get("compiled"):
            raise AdapterError("this adapter build has no O3 route. Build it with: python scripts/native_adapter.py "
                               "build --o3")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except AdapterError as error:
        print(f"native_adapter: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""The sensing core has no host: python tests/unit/test_core_imports.py -v

ocudu_cuda_isac.sensing receives observation records and returns features and decisions. It may
import the standard library, NumPy, PyTorch and itself. It may not import an episode module, a
host adapter, an OCUDU binding or a Sionna package: an adapter produces records for it, and a
simulator runs offline. The rule holds from Sprint 3.3 on and is checked twice, in the source of
every module of the package and in the modules that an interpreter holds after importing them.
"""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "ocudu_cuda_isac.sensing"
SOURCE = ROOT / "src/ocudu_cuda_isac/sensing"
# What the reference of the sensing core is written in.
LIBRARIES = ("numpy", "torch")
# Named so that a failure says what was imported: the offline side, the hosts and the simulators.
FORBIDDEN = ("ocudu_cuda_isac.episodes", "ocudu_cuda_isac.replay", "ocudu_cuda_isac.telemetry",
             "ocudu_cuda_isac.evaluation", "sionna", "sionna_rt", "mitsuba", "drjit", "tensorflow", "ocudu", "srsran",
             "aerial", "pyaerial", "native_adapter", "native_ocudu")


def module_name(path):
    relative = path.relative_to(SOURCE.parents[1]).with_suffix("")
    parts = list(relative.parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports_of(text, name, is_package):
    """Every module a source text imports, as absolute names."""
    found = []
    package = name if is_package else name.rpartition(".")[0]
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[:len(base) - (node.level - 1)]
                prefix = ".".join(base + ([node.module] if node.module else []))
                found += [prefix] if node.module else [f"{prefix}.{alias.name}" for alias in node.names]
            else:
                found.append(node.module)
    return found


def violations(text, name, is_package=False):
    """The imports of a module of the sensing package that leave what the core may import."""
    found = []
    for imported in imports_of(text, name, is_package):
        top = imported.split(".")[0]
        if imported == PACKAGE or imported.startswith(PACKAGE + "."):
            continue
        if top in sys.stdlib_module_names or top in LIBRARIES:
            continue
        found.append(imported)
    return found


def forbidden(name):
    return any(name == entry or name.startswith(entry + ".") for entry in FORBIDDEN)


class CoreImportTests(unittest.TestCase):
    def modules(self):
        paths = sorted(SOURCE.rglob("*.py"))
        self.assertGreaterEqual(len(paths), 5)
        return paths

    def test_no_module_of_the_sensing_package_imports_a_host_or_a_simulator(self):
        names = []
        for path in self.modules():
            name = module_name(path)
            names.append(name)
            with self.subTest(module=name):
                self.assertEqual(violations(path.read_text(encoding="ascii"), name, path.name == "__init__.py"), [])
        for needed in ("cusense", "features", "models", "models.cusense_cnn", "native_estimate", "obs", "pilot_ls",
                       "presence", "sources", "sources.pusch_dmrs"):
            self.assertIn(f"{PACKAGE}.{needed}", names)

    def test_the_rule_finds_what_it_forbids(self):
        cases = (("from ..episodes import episode", ["ocudu_cuda_isac.episodes"]),
                 ("from .. import replay", ["ocudu_cuda_isac.replay"]),
                 ("from ..episodes.pilot_features import oracle", ["ocudu_cuda_isac.episodes.pilot_features"]),
                 ("import sionna.rt", ["sionna.rt"]), ("import mitsuba as mi\nimport drjit", ["mitsuba", "drjit"]),
                 ("def late():\n    import native_adapter", ["native_adapter"]),
                 ("from . import obs\nfrom .sources import pusch_dmrs\nimport numpy as np\nimport torch\nimport json", []))
        for text, expected in cases:
            with self.subTest(text=text):
                found = violations(text, f"{PACKAGE}.features")
                self.assertEqual(found, expected)
                self.assertTrue(all(forbidden(name) for name in found))
        self.assertEqual(violations("from . import obs\nfrom .. import episodes", PACKAGE, True),
                         ["ocudu_cuda_isac.episodes"])
        # The evaluator side may import the core. The reverse is what the rule forbids.
        evaluator = (ROOT / "src/ocudu_cuda_isac/episodes/pilot_features.py").read_text(encoding="ascii")
        self.assertIn(PACKAGE, imports_of(evaluator, "ocudu_cuda_isac.episodes.pilot_features", False))
        # The same holds for the head of Sprint 3.4: the split, the labels and the plans are the evaluator's.
        evaluator = (ROOT / "src/ocudu_cuda_isac/episodes/presence_head.py").read_text(encoding="ascii")
        self.assertIn(PACKAGE, imports_of(evaluator, "ocudu_cuda_isac.episodes.presence_head", False))
        self.assertEqual(violations("from . import features\nfrom . import obs\nimport torch\nimport numpy as np",
                                    f"{PACKAGE}.presence"), [])
        # And for the cuSense-like branch of Sprint 3.5: the model entry is a module of the core, under models/,
        # and the positions of a target are the evaluator's.
        evaluator = (ROOT / "src/ocudu_cuda_isac/episodes/cusense_frontend.py").read_text(encoding="ascii")
        self.assertIn(PACKAGE, imports_of(evaluator, "ocudu_cuda_isac.episodes.cusense_frontend", False))
        self.assertEqual(violations("from . import presence\nfrom .models import cusense_cnn\nimport torch",
                                    f"{PACKAGE}.cusense"), [])
        self.assertEqual(violations("import torch\nfrom torch import nn\nimport numpy as np",
                                    f"{PACKAGE}.models.cusense_cnn"), [])
        self.assertEqual(violations("from ...episodes import cusense_frontend", f"{PACKAGE}.models.cusense_cnn"),
                         ["ocudu_cuda_isac.episodes"])

    def test_an_interpreter_that_imported_the_core_holds_no_host_module(self):
        names = [module_name(path) for path in self.modules()]
        # The reference of the features is NumPy, and that of the head and of the cuSense-like branch PyTorch on
        # top of it.
        if importlib.util.find_spec("numpy") is None:
            names = [name for name in names if not name.endswith((".features", ".presence", ".cusense", ".cusense_cnn"))]
        elif importlib.util.find_spec("torch") is None:
            names = [name for name in names if not name.endswith((".presence", ".cusense", ".cusense_cnn"))]
        script = ("import importlib, json, sys\n"
                  f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
                  f"for name in {names!r}:\n"
                  "    importlib.import_module(name)\n"
                  "print(json.dumps(sorted(sys.modules)))\n")
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = json.loads(result.stdout)
        self.assertEqual([name for name in loaded if forbidden(name)], [])
        self.assertEqual([name for name in loaded if name.startswith("ocudu_cuda_isac.") and not name.startswith(PACKAGE)],
                         [])
        for name in names:
            self.assertIn(name, loaded)


if __name__ == "__main__":
    unittest.main()

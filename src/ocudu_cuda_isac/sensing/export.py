"""Export of a frozen artifact for the engine host of the native core (Sprint 3.6, selector sense_core).

The native sensing core runs the model of a cusense-artifact/v1 artifact through an engine host.
This module writes what a backend of that host loads, from the four files of the artifact and
from nothing else:

  write_onnx     model.onnx and export.json: the model as an ONNX graph, built node by node from
                 weights.f32 and the configuration of model.json. scripts/native_core.py engine
                 hands it to sense_engine, which builds the TensorRT engine on the GPU that runs
                 it. The graph takes "input" [1, A, K] as float32 and gives "map" [1, rows * cols],
                 the softmax of the grid logits, and "presence_logit" [1].
  write_module   model.ts and engine.json: the same model as a TorchScript module, for the
                 declared fallback backend libtorch on a host without a usable TensorRT.
  run_onnx       the graph on one input through ONNX Runtime, where that is installed: a check of
                 the export on a host without a GPU, used by the unit gate.

The graph holds the arithmetic of ocudu_cuda_isac.sensing.models.cusense_cnn in evaluation mode:
the convolutions along the subcarrier axis as convolutions of height one, the batch
normalization with the epsilon of torch.nn.BatchNorm1d, and no dropout. It is written with the
operators of the default ONNX domain at one fixed operator set, so the same artifact gives the
same file on every host.

ONNX is imported when a graph is written and PyTorch when a module is, so the package imports
without either. Nothing here is imported by a sensor.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from . import cusense

EXPORT_SCHEMA = "sense-export/v1"
ENGINE_SCHEMA = "sense-engine/v1"
ONNX_FILE = "model.onnx"
EXPORT_FILE = "export.json"
MODULE_FILE = "model.ts"
ENGINE_FILE = "engine.json"
INPUT, MAP, LOGIT = "input", "map", "presence_logit"
OPSET = 17
IR_VERSION = 9
BATCH_NORM_EPSILON = 1e-5  # the default of torch.nn.BatchNorm1d, which the reference model uses


class ExportError(ValueError):
    """An artifact cannot be exported, or an export is not that of its artifact."""


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _artifact(artifact):
    try:
        return artifact if isinstance(artifact, cusense.Artifact) else cusense.open_artifact(artifact)
    except cusense.CuSenseError as error:
        raise ExportError(str(error)) from None


def tensors_of(artifact):
    """The tensors of the weight file of an artifact by name, as float32 arrays of their shape."""
    values = np.frombuffer(artifact.weights, dtype="<f4")
    made = {}
    for item in artifact.manifest["weights"]["tensors"]:
        made[item["name"]] = values[item["offset"]:item["offset"] + item["count"]].reshape(item["shape"])
    return made


def onnx_model(artifact):
    """The model of an artifact as an ONNX ModelProto."""
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as error:
        raise ExportError(f"the export needs the onnx package in this interpreter: {error}") from None
    artifact = _artifact(artifact)
    manifest = artifact.manifest
    config, weights = manifest["model"]["config"], tensors_of(artifact)
    _, antennas, positions = manifest["input"]["shape"]
    cells = manifest["grid"]["rows"] * manifest["grid"]["cols"]
    nodes, initializers = [], []

    def constant(name, values):
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(values), name))
        return name

    def tensor(name):
        if name not in weights:
            raise ExportError(f"the weight file names no tensor {name}: another architecture wrote it")
        return weights[name]

    def convolution(source, name, norm, stride):
        kernel = tensor(f"{name}.weight")
        pad = kernel.shape[2] // 2
        nodes.append(helper.make_node(
            "Conv", [source, constant(f"{name}.weight", kernel.reshape(kernel.shape[0], kernel.shape[1], 1, -1))],
            [f"{name}.out"], name=name, kernel_shape=[1, kernel.shape[2]], strides=[1, stride], pads=[0, pad, 0, pad]))
        nodes.append(helper.make_node(
            "BatchNormalization",
            [f"{name}.out"] + [constant(f"{norm}.{part}", tensor(f"{norm}.{part}"))
                               for part in ("weight", "bias", "running_mean", "running_var")],
            [f"{norm}.out"], name=norm, epsilon=BATCH_NORM_EPSILON))
        nodes.append(helper.make_node("Relu", [f"{norm}.out"], [f"{norm}.relu"], name=f"{norm}.relu"))
        return f"{norm}.relu"

    def dense(source, name, output=None):
        nodes.append(helper.make_node(
            "Gemm", [source, constant(f"{name}.weight", tensor(f"{name}.weight")),
                     constant(f"{name}.bias", tensor(f"{name}.bias"))], [output or f"{name}.out"], name=name, transB=1))
        return output or f"{name}.out"

    # [1, A, K] as an image of height one: [1, A, 1, K].
    nodes.append(helper.make_node("Unsqueeze", [INPUT, constant("axis_height", np.array([2], dtype=np.int64))],
                                  ["image"], name="as_image"))
    last = convolution("image", "stem.0", "stem.1", 1)
    nodes.append(helper.make_node("MaxPool", [last], ["stem.pool"], name="stem.3", kernel_shape=[1, 2], strides=[1, 2]))
    last = "stem.pool"
    for block in range(3):
        last = convolution(last, f"blocks.{block}.0", f"blocks.{block}.1", 2)
        last = convolution(last, f"blocks.{block}.3", f"blocks.{block}.4", 1)
    nodes.append(helper.make_node("GlobalAveragePool", [last], ["pooled"], name="pool"))
    nodes.append(helper.make_node("Flatten", ["pooled"], ["features"], name="flatten", axis=1))
    last = "features"
    for layer in ("trunk.0", "trunk.3"):
        nodes.append(helper.make_node("Relu", [dense(last, layer)], [f"{layer}.relu"], name=f"{layer}.relu"))
        last = f"{layer}.relu"
    nodes.append(helper.make_node("Softmax", [dense(last, "location")], [MAP], name="map_softmax", axis=1))
    nodes.append(helper.make_node("Reshape", [dense(last, "presence"), constant("one_value", np.array([1], dtype=np.int64))],
                                  [LOGIT], name="presence_value"))
    if config["stem_channels"] != weights["stem.0.weight"].shape[0] or weights["stem.0.weight"].shape[1] != antennas \
            or weights["location.weight"].shape[0] != cells or weights["presence.weight"].shape[0] != 1:
        raise ExportError("the weights are not those of the configuration, the array and the grid of the artifact")
    graph = helper.make_graph(
        nodes, f"cusense_cnn-{artifact.sha256[:16]}",
        [helper.make_tensor_value_info(INPUT, TensorProto.FLOAT, [1, antennas, positions])],
        [helper.make_tensor_value_info(MAP, TensorProto.FLOAT, [1, cells]),
         helper.make_tensor_value_info(LOGIT, TensorProto.FLOAT, [1])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR_VERSION,
                              producer_name="ocudu-cuda-isac", producer_version="sense-export/v1",
                              doc_string="cusense_cnn in evaluation mode: input [1, A, K], map [1, rows * cols] and "
                                         "presence_logit [1], float32")
    for key, value in (("artifact_sha256", artifact.sha256), ("weights_sha256", manifest["weights"]["sha256"]),
                       ("model_id", manifest["model_id"])):
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as error:
        raise ExportError(f"the exported graph is not a valid ONNX model: {error}") from None
    return model


def _empty(directory):
    directory = Path(directory)
    if not directory.is_dir() or any(directory.iterdir()):
        raise ExportError(f"the output directory {directory} does not exist or is not empty")
    return directory


def write_onnx(artifact, directory):
    """Write model.onnx and export.json into an existing, empty directory. Returns the export record."""
    import onnx
    directory, artifact = _empty(directory), _artifact(artifact)
    manifest = artifact.manifest
    data = onnx_model(artifact).SerializeToString()
    (directory / ONNX_FILE).write_bytes(data)
    record = {"schema": EXPORT_SCHEMA, "format": "onnx", "file": ONNX_FILE, "bytes": len(data), "sha256": _sha256(data),
              "opset": OPSET, "ir_version": IR_VERSION, "onnx": onnx.__version__, "model_id": manifest["model_id"],
              "entry": manifest["entries"]["model"], "artifact_sha256": artifact.sha256,
              "weights_sha256": manifest["weights"]["sha256"],
              "input": {"name": INPUT, "shape": list(manifest["input"]["shape"]), "dtype": "<f4"},
              "outputs": {MAP: [1, manifest["grid"]["rows"] * manifest["grid"]["cols"]], LOGIT: [1]},
              "batch_norm_epsilon": BATCH_NORM_EPSILON,
              "from": "weights.f32 and the configuration of model.json: no other file and no trained module"}
    (directory / EXPORT_FILE).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return record


def check_onnx(directory, artifact):
    """Check that a directory holds the ONNX export of this artifact, byte for byte. Returns the record."""
    directory, artifact = Path(directory), _artifact(artifact)
    try:
        record = json.loads((directory / EXPORT_FILE).read_text(encoding="ascii"))
        data = (directory / ONNX_FILE).read_bytes()
    except (OSError, ValueError) as error:
        raise ExportError(f"{directory} is not a complete export: {error}") from None
    if not isinstance(record, dict) or record.get("schema") != EXPORT_SCHEMA or record.get("sha256") != _sha256(data) \
            or record.get("artifact_sha256") != artifact.sha256:
        raise ExportError("the export does not have the hash it names, or is that of another artifact")
    if data != onnx_model(artifact).SerializeToString():
        raise ExportError("the ONNX file is not the export of this artifact with this onnx package")
    return record


def run_onnx(path, x):
    """One input [1, A, K] through the graph with ONNX Runtime on the CPU: (map [1, rows * cols], logit [1])."""
    try:
        import onnxruntime
    except ImportError as error:
        raise ExportError(f"ONNX Runtime is not installed in this interpreter: {error}") from None
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    drawn, logit = session.run([MAP, LOGIT], {INPUT: np.ascontiguousarray(x, dtype=np.float32)})
    return drawn, logit


def write_module(artifact, directory):
    """Write model.ts and engine.json of the backend libtorch into an existing, empty directory. Returns the manifest."""
    try:
        import torch
    except ImportError as error:
        raise ExportError(f"the TorchScript module needs PyTorch in this interpreter: {error}") from None
    from .models import cusense_cnn
    directory, artifact = _empty(directory), _artifact(artifact)
    manifest = artifact.manifest
    try:
        net = cusense_cnn.unpack(cusense_cnn.CuSenseCNN(manifest["model"]["config"]), artifact.weights,
                                 manifest["weights"]["tensors"]).eval()
    except cusense_cnn.ModelError as error:
        raise ExportError(str(error)) from None

    class Served(torch.nn.Module):
        """The two outputs an engine has: the map as the softmax of the grid logits, and the presence logit."""

        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            shared = self.net.trunk(self.net.pool(self.net.blocks(self.net.stem(x))).flatten(1))
            return torch.softmax(self.net.location(shared), dim=1), self.net.presence(shared).reshape(-1)

    shape = list(manifest["input"]["shape"])
    with torch.no_grad():
        traced = torch.jit.trace(Served(net).eval(), torch.zeros(shape, dtype=torch.float32), check_trace=False)
    traced.save(str(directory / MODULE_FILE))
    data = (directory / MODULE_FILE).read_bytes()
    made = {"schema": ENGINE_SCHEMA, "backend": "libtorch", "model_id": manifest["model_id"],
            "artifact_sha256": artifact.sha256, "weights_sha256": manifest["weights"]["sha256"],
            "library": {"name": "torch", "version": torch.__version__.split("+")[0]}, "precision": "fp32",
            "device": {"name": "any", "capability": "any"},
            "input": {"name": INPUT, "shape": shape},
            "outputs": {MAP: [1, manifest["grid"]["rows"] * manifest["grid"]["cols"]], LOGIT: [1]},
            "file": MODULE_FILE, "bytes": len(data), "sha256": _sha256(data),
            "role": "the declared fallback: it blocks, and a run on it is no timed run of the native core"}
    (directory / ENGINE_FILE).write_text(json.dumps(made, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return made

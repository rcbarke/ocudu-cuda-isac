"""The cuSense location model with a presence logit: the model entry cusense_cnn (Sprint 3.5, selector cusense_frontend).

The network is the plain-block 1D CNN that the cuSense paper draws (Villa et al.,
arXiv:2512.06493v3, figure 14): the input is one real value per antenna and subcarrier, and the
convolutions run along the subcarrier axis with the antennas as channels.

  stem        Conv1d(A -> c0, kernel 7), BatchNorm, ReLU, MaxPool(2)
  blocks      three plain blocks without a residual connection, each
              Conv1d(stride 2), BatchNorm, ReLU, Conv1d, BatchNorm, ReLU
  pool        AdaptiveAvgPool1d(1): any number of subcarriers gives one vector
  trunk       Linear, ReLU, Dropout, Linear, ReLU, Dropout
  location    Linear to H * W grid logits; the map is their softmax, [B, H, W], summing to one
  presence    Linear to one logit, on the same trunk

The published model has the widths 64, 128, 256 and 512, a trunk of 512 and 256 and dropouts of
0.3 and 0.15. The presence logit is this project's addition: the published model has no detection
stage, and a demonstration that scores target-absent episodes needs one. The two output layers
start at zero, so an untrained model gives a uniform map and the presence logit 0: the loss of a
fit then starts at a value that follows from the grid alone.

Losses, each with its unit and its eligibility:

  location_grid  KL divergence from the predicted map to a Gaussian-smoothed target, in nats,
                 on target-present samples only. A target-absent sample has no position, adds
                 nothing to this loss and sends no gradient into the grid logits.
  presence       binary cross-entropy of the presence logit, in nats, on every sample.

The model uses convolutions and plain matrix products and no batched product with broadcast
operands. This module imports NumPy and PyTorch and nothing of a host, a simulator or an episode.
"""

import hashlib
import json
import math

import numpy as np
import torch
from torch import nn

ENTRY = "cusense_cnn"
CONFIG_FIELDS = ("entry", "antennas", "rows", "cols", "stem_channels", "stem_kernel", "block_channels", "block_kernel",
                 "hidden", "dropout", "presence_head")
HEADS = ("location_grid", "presence")
LOSSES = {
    "location_grid": {"entry": "kl_gaussian", "unit": "nats per target-present sample",
                      "eligible": "target-present samples: a target-absent sample has no position",
                      "does": "sum over the cells of P_Y log(P_Y / P_X), with P_Y the Gaussian-smoothed cell of the target"},
    "presence": {"entry": "bce_logit", "unit": "nats per sample", "eligible": "every sample",
                 "does": "binary cross-entropy of the presence logit against the label"},
}
# Figure 14 of the cuSense paper. The kernel of the blocks is not printed there: 3, as in the plain
# ResNet blocks the paper cites.
PUBLISHED = {"stem_channels": 64, "stem_kernel": 7, "block_channels": [128, 256, 512], "block_kernel": 3,
             "hidden": [512, 256], "dropout": [0.3, 0.15]}


class ModelError(ValueError):
    """A model configuration, a weight file or an input violates the documented interface."""


def _count(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ModelError(f"{name} must be an integer from {low} to {high}")
    return value


def check_config(config):
    """Check the configuration of one cusense_cnn model and return it."""
    if not isinstance(config, dict) or set(config) != set(CONFIG_FIELDS):
        raise ModelError(f"a {ENTRY} configuration requires exactly {', '.join(CONFIG_FIELDS)}")
    if config["entry"] != ENTRY:
        raise ModelError(f"unknown model entry {config['entry']!r}: this module defines {ENTRY}")
    _count(config["antennas"], "antennas", 1, 64)
    _count(config["rows"], "rows", 1, 4096)
    _count(config["cols"], "cols", 1, 4096)
    _count(config["stem_channels"], "stem_channels", 1, 4096)
    for name in ("stem_kernel", "block_kernel"):
        if _count(config[name], name, 1, 63) % 2 == 0:
            raise ModelError(f"{name} must be odd, so that a convolution keeps the centre of its window")
    blocks, hidden, dropout = config["block_channels"], config["hidden"], config["dropout"]
    if not isinstance(blocks, list) or len(blocks) != 3:
        raise ModelError("block_channels must list the width of the three plain blocks")
    if not isinstance(hidden, list) or len(hidden) != 2:
        raise ModelError("hidden must list the two widths of the trunk")
    for value in blocks + hidden:
        _count(value, "a width", 1, 8192)
    if not isinstance(dropout, list) or len(dropout) != 2 \
            or any(type(value) not in (int, float) or not 0 <= value < 1 for value in dropout):
        raise ModelError("dropout must list two rates from 0 to below 1")
    if config["presence_head"] is not True:
        raise ModelError("presence_head must be true: target-absent windows are scored by the presence logit")
    return config


def published_config(antennas, rows, cols):
    """The configuration with the published widths, for a grid of rows by cols cells."""
    return check_config(dict(json.loads(json.dumps(PUBLISHED)), entry=ENTRY, antennas=antennas, rows=rows, cols=cols,
                             presence_head=True))


def config_sha256(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def _block(inputs, outputs, kernel):
    pad = kernel // 2
    return nn.Sequential(nn.Conv1d(inputs, outputs, kernel, stride=2, padding=pad, bias=False), nn.BatchNorm1d(outputs),
                         nn.ReLU(inplace=True),
                         nn.Conv1d(outputs, outputs, kernel, stride=1, padding=pad, bias=False), nn.BatchNorm1d(outputs),
                         nn.ReLU(inplace=True))


class CuSenseCNN(nn.Module):
    """One cusense_cnn model. forward takes [B, A, K] and returns the grid logits [B, H * W] and the presence logits [B]."""

    def __init__(self, config):
        super().__init__()
        self.config = check_config(config)
        stem, kernel = config["stem_channels"], config["stem_kernel"]
        widths, hidden, rates = config["block_channels"], config["hidden"], config["dropout"]
        self.antennas, self.rows, self.cols = config["antennas"], config["rows"], config["cols"]
        self.stem = nn.Sequential(nn.Conv1d(self.antennas, stem, kernel, padding=kernel // 2, bias=False),
                                  nn.BatchNorm1d(stem), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.blocks = nn.Sequential(_block(stem, widths[0], config["block_kernel"]),
                                    _block(widths[0], widths[1], config["block_kernel"]),
                                    _block(widths[1], widths[2], config["block_kernel"]))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.trunk = nn.Sequential(nn.Linear(widths[2], hidden[0]), nn.ReLU(inplace=True), nn.Dropout(rates[0]),
                                   nn.Linear(hidden[0], hidden[1]), nn.ReLU(inplace=True), nn.Dropout(rates[1]))
        self.location = nn.Linear(hidden[1], self.rows * self.cols)
        self.presence = nn.Linear(hidden[1], 1)
        # An untrained model answers with a uniform map and the logit 0.
        for layer in (self.location, self.presence):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        if x.dim() != 3 or x.shape[1] != self.antennas:
            raise ModelError(f"the model takes [batch, {self.antennas}, subcarriers], and this input has the shape "
                             f"{list(x.shape)}")
        if x.shape[2] < 16:
            raise ModelError(f"the model needs 16 subcarriers or more, and this input has {x.shape[2]}")
        shared = self.trunk(self.pool(self.blocks(self.stem(x))).flatten(1))
        return self.location(shared), self.presence(shared).reshape(-1)

    def grid(self, x):
        """The location map [B, H, W], each summing to one, and the presence logits [B]."""
        logits, presence = self.forward(x)
        return torch.softmax(logits, dim=1).reshape(-1, self.rows, self.cols), presence


def parameters_of(model):
    """How many trained values a model holds."""
    return sum(parameter.numel() for parameter in model.parameters())


# ---------------------------------------------------------------- weights as one flat file
def tensors_of(model):
    """The tensors of a model that its arithmetic reads, by name, in the order of its state.

    These are its parameters and the running statistics of its normalization layers. The counter
    of batches that a normalization layer keeps is not read by an inference and is left out.
    """
    return [(name, tensor) for name, tensor in model.state_dict().items() if tensor.is_floating_point()]


def pack(model):
    """The bytes of a weight file and its layout: one little-endian float32 per value, tensor after tensor."""
    layout, parts, offset = [], [], 0
    for name, tensor in tensors_of(model):
        values = tensor.detach().to("cpu", torch.float32).contiguous()
        if not bool(torch.isfinite(values).all()):
            raise ModelError(f"the tensor {name} holds a value that is not finite")
        layout.append({"name": name, "shape": list(values.shape), "offset": offset, "count": values.numel()})
        parts.append(values.numpy().astype("<f4").tobytes())
        offset += values.numel()
    return b"".join(parts), layout


def unpack(model, data, layout):
    """Load a weight file into a model. The layout must be that of the model, tensor for tensor."""
    wanted = [(name, list(tensor.shape)) for name, tensor in tensors_of(model)]
    if [(item.get("name"), item.get("shape")) for item in layout] != wanted:
        raise ModelError("the layout of the weight file is not that of the model: another architecture wrote it")
    total = sum(item["count"] for item in layout)
    if len(data) != 4 * total:
        raise ModelError(f"the weight file holds {len(data)} bytes, and its layout names {4 * total}")
    values = torch.from_numpy(np.frombuffer(data, dtype="<f4").astype(np.float32))
    if not bool(torch.isfinite(values).all()):
        raise ModelError("the weight file holds a value that is not finite")
    state, offset = {}, 0
    for item in layout:
        if item["offset"] != offset or item["count"] != math.prod(item["shape"]):
            raise ModelError("the layout of the weight file is not contiguous")
        state[item["name"]] = values[offset:offset + item["count"]].reshape(item["shape"]).clone()
        offset += item["count"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.endswith("num_batches_tracked") for name in missing):
        raise ModelError("the weight file does not name every tensor of the model")
    return model


# ---------------------------------------------------------------- targets and losses
def smoothed_target(cells, rows, cols, sigma_cells):
    """The training target of each sample: its cell smoothed by a Gaussian and normalized on the grid.

    cells is an integer tensor [B, 2] of rows and columns. The result is [B, rows, cols] and sums
    to one per sample: the one-hot grid of the cell convolved with a Gaussian of sigma_cells
    cells and divided by its sum over the grid, so a cell at the edge keeps a whole distribution.
    With sigma_cells of 0 the target is the one-hot grid.
    """
    if cells.dim() != 2 or cells.shape[1] != 2:
        raise ModelError("cells is an integer tensor of shape [batch, 2]: a row and a column per sample")
    row, col = cells[:, 0], cells[:, 1]
    if bool(((row < 0) | (row >= rows) | (col < 0) | (col >= cols)).any()):
        raise ModelError(f"a cell lies outside the grid of {rows} by {cols}")
    if type(sigma_cells) not in (int, float) or not sigma_cells >= 0 or not math.isfinite(sigma_cells):
        raise ModelError("sigma_cells must be zero or positive")
    along_rows = torch.arange(rows, device=cells.device, dtype=torch.float64).reshape(1, -1) - row.reshape(-1, 1)
    along_cols = torch.arange(cols, device=cells.device, dtype=torch.float64).reshape(1, -1) - col.reshape(-1, 1)
    if sigma_cells == 0:
        by_row, by_col = (along_rows == 0).double(), (along_cols == 0).double()
    else:
        by_row = torch.exp(-0.5 * (along_rows / sigma_cells) ** 2)
        by_col = torch.exp(-0.5 * (along_cols / sigma_cells) ** 2)
    by_row = by_row / by_row.sum(dim=1, keepdim=True)
    by_col = by_col / by_col.sum(dim=1, keepdim=True)
    return (by_row.unsqueeze(2) * by_col.unsqueeze(1)).float()


def kl_divergence(target, logits):
    """KL divergence of each sample, in nats: sum of P_Y log(P_Y / P_X) with P_X the softmax of the logits.

    target is [B, cells] or [B, rows, cols] and sums to one per sample. A cell with a target of
    zero adds nothing, whatever the prediction holds there.
    """
    target = target.reshape(target.shape[0], -1)
    if logits.shape != target.shape:
        raise ModelError(f"the target has the shape {list(target.shape)} and the logits {list(logits.shape)}")
    log_prediction = torch.log_softmax(logits, dim=1)
    log_target = torch.where(target > 0, torch.log(target.clamp_min(1e-45)), torch.zeros_like(target))
    return (target * (log_target - log_prediction)).sum(dim=1)


def joint_loss(grid_logits, presence_logits, targets, labels, grid_weight=1.0):
    """The loss of one batch and its two parts: (loss, presence loss, location loss).

    labels is 1 for a target-present sample and 0 for a target-absent one. The presence part is
    the mean binary cross-entropy over every sample. The location part is the mean KL divergence
    over the target-present samples, and zero with no gradient when the batch holds none.
    """
    labels = labels.to(presence_logits.dtype)
    presence = nn.functional.binary_cross_entropy_with_logits(presence_logits, labels)
    eligible = labels.sum()
    location = (kl_divergence(targets, grid_logits) * labels).sum() / eligible.clamp_min(1.0)
    return presence + grid_weight * location, presence, location

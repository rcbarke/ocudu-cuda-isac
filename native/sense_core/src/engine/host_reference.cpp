// The model cusense_cnn in plain C++ on the CPU, from the weight file of an artifact (Sprint 3.6,
// selector sense_core).
//
// It is the engine of the host twin and of the unit gate: the published plain-block 1D CNN of
// ocudu_cuda_isac.sensing.models.cusense_cnn in evaluation mode, in float32, one window at a
// time. Every tensor is looked up by the name the weight file gives it.
//
//   stem        Conv1d(A -> c0, kernel k0, same padding), BatchNorm, ReLU, MaxPool(2)
//   blocks      three times: Conv1d(stride 2), BatchNorm, ReLU, Conv1d, BatchNorm, ReLU
//   pool        the mean over the subcarrier axis
//   trunk       Linear, ReLU, Linear, ReLU (dropout is the identity in evaluation mode)
//   location    Linear to rows * cols logits, then their softmax: the map
//   presence    Linear to one logit
//
// The sums run in another order than those of PyTorch, so a score agrees with the reference
// within rounding and not bit for bit. tolerance.h declares the bound. Every buffer is
// allocated when the engine is built. It is no timed runtime. ASCII only.

#include "core/sha256.h"
#include "engine/engine.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace sense {

namespace {

constexpr double BATCH_NORM_EPSILON = 1e-5; // the default of torch.nn.BatchNorm1d

struct convolution {
  const float*       weight = nullptr; // [out][in][kernel]
  std::size_t        in = 0, out = 0, kernel = 0, stride = 1, pad = 0;
  std::vector<float> scale, shift; // the batch normalization behind it, folded per channel
};

struct linear {
  const float* weight = nullptr; // [out][in]
  const float* bias   = nullptr;
  std::size_t  in = 0, out = 0;
};

std::size_t length_after(const convolution& layer, std::size_t length)
{
  return (length + 2 * layer.pad - layer.kernel) / layer.stride + 1;
}

class host_reference_engine final : public engine
{
public:
  explicit host_reference_engine(const artifact& model) : rows(model.rows), cols(model.cols), positions(model.subcarriers)
  {
    hash = model.weights_sha256;
    std::size_t in = model.antennas, length = positions, widest = in * length;
    layers.push_back(conv(model, "stem.0", "stem.1", in, model.stem_channels, model.stem_kernel, 1));
    in     = model.stem_channels;
    widest = std::max(widest, in * length);
    length /= 2; // MaxPool(2)
    if (positions < 16 || length == 0) {
      throw engine_error("the model needs 16 subcarriers or more");
    }
    for (unsigned block = 0; block != 3; ++block) {
      const std::string name  = "blocks." + std::to_string(block);
      const std::size_t width = model.block_channels[block];
      layers.push_back(conv(model, name + ".0", name + ".1", in, width, model.block_kernel, 2));
      length = length_after(layers.back(), length);
      layers.push_back(conv(model, name + ".3", name + ".4", width, width, model.block_kernel, 1));
      in     = width;
      widest = std::max(widest, in * length);
      if (length == 0) {
        throw engine_error("the model has no subcarrier left after its blocks");
      }
    }
    trunk[0]  = dense(model, "trunk.0", in, model.hidden[0]);
    trunk[1]  = dense(model, "trunk.3", model.hidden[0], model.hidden[1]);
    location  = dense(model, "location", model.hidden[1], static_cast<std::size_t>(rows) * cols);
    presence  = dense(model, "presence", model.hidden[1], 1);
    first.assign(widest, 0.0f);
    second.assign(widest, 0.0f);
    pooled.assign(in, 0.0f);
    hidden[0].assign(model.hidden[0], 0.0f);
    hidden[1].assign(model.hidden[1], 0.0f);
  }

  const char* backend() const override { return "host_reference"; }
  std::string detail() const override { return "cusense_cnn in C++ on the CPU, float32, from weights.f32"; }
  std::string sha256() const override { return hash; }
  bool        on_stream() const override { return false; }
  void        bind(const engine_binding& binding) override { bound = binding; }

  void infer() override
  {
    if (bound.input == nullptr || bound.map == nullptr || bound.logit == nullptr) {
      throw engine_error("the engine was not bound to its input and its outputs");
    }
    std::size_t length = positions;
    // Stem: convolution, normalization, ReLU and the maximum of each pair.
    run(layers[0], bound.input, length, first.data());
    const std::size_t stem = layers[0].out;
    length /= 2;
    for (std::size_t channel = 0; channel != stem; ++channel) {
      const float* from = first.data() + channel * positions;
      float*       to   = second.data() + channel * length;
      for (std::size_t t = 0; t != length; ++t) {
        to[t] = from[2 * t] > from[2 * t + 1] ? from[2 * t] : from[2 * t + 1];
      }
    }
    float* now  = second.data();
    float* next = first.data();
    for (std::size_t index = 1; index != layers.size(); ++index) {
      run(layers[index], now, length, next);
      length = length_after(layers[index], length);
      std::swap(now, next);
    }
    // The mean over the subcarrier axis.
    const std::size_t channels = layers.back().out;
    for (std::size_t channel = 0; channel != channels; ++channel) {
      float sum = 0.0f;
      for (std::size_t t = 0; t != length; ++t) {
        sum += now[channel * length + t];
      }
      pooled[channel] = sum / static_cast<float>(length);
    }
    apply(trunk[0], pooled.data(), hidden[0].data(), true);
    apply(trunk[1], hidden[0].data(), hidden[1].data(), true);
    apply(location, hidden[1].data(), bound.map, false);
    apply(presence, hidden[1].data(), bound.logit, false);
    // The softmax of the grid logits.
    const std::size_t cells = static_cast<std::size_t>(rows) * cols;
    float             top   = bound.map[0];
    for (std::size_t cell = 1; cell != cells; ++cell) {
      top = bound.map[cell] > top ? bound.map[cell] : top;
    }
    float total = 0.0f;
    for (std::size_t cell = 0; cell != cells; ++cell) {
      bound.map[cell] = std::exp(bound.map[cell] - top);
      total += bound.map[cell];
    }
    for (std::size_t cell = 0; cell != cells; ++cell) {
      bound.map[cell] /= total;
    }
  }

private:
  static convolution conv(const artifact& model, const std::string& name, const std::string& norm, std::size_t in,
                          std::size_t out, std::size_t kernel, std::size_t stride)
  {
    const tensor_entry& weight = model.tensor(name + ".weight");
    if (weight.shape != std::vector<std::size_t>{out, in, kernel}) {
      throw engine_error("the tensor " + weight.name + " does not have the shape of its layer");
    }
    convolution layer;
    layer.weight = model.values(weight);
    layer.in     = in;
    layer.out    = out;
    layer.kernel = kernel;
    layer.stride = stride;
    layer.pad    = kernel / 2;
    const tensor_entry& gain = model.tensor(norm + ".weight");
    const tensor_entry& bias = model.tensor(norm + ".bias");
    const tensor_entry& mean = model.tensor(norm + ".running_mean");
    const tensor_entry& var  = model.tensor(norm + ".running_var");
    if (gain.count != out || bias.count != out || mean.count != out || var.count != out) {
      throw engine_error("the normalization " + norm + " does not have one value per channel");
    }
    layer.scale.resize(out);
    layer.shift.resize(out);
    for (std::size_t channel = 0; channel != out; ++channel) {
      const double scale   = static_cast<double>(model.values(gain)[channel]) /
                             std::sqrt(static_cast<double>(model.values(var)[channel]) + BATCH_NORM_EPSILON);
      layer.scale[channel] = static_cast<float>(scale);
      layer.shift[channel] = static_cast<float>(static_cast<double>(model.values(bias)[channel]) -
                                                static_cast<double>(model.values(mean)[channel]) * scale);
    }
    return layer;
  }

  static linear dense(const artifact& model, const std::string& name, std::size_t in, std::size_t out)
  {
    const tensor_entry& weight = model.tensor(name + ".weight");
    const tensor_entry& bias   = model.tensor(name + ".bias");
    if (weight.shape != std::vector<std::size_t>{out, in} || bias.count != out) {
      throw engine_error("the tensor " + weight.name + " does not have the shape of its layer");
    }
    linear layer;
    layer.weight = model.values(weight);
    layer.bias   = model.values(bias);
    layer.in     = in;
    layer.out    = out;
    return layer;
  }

  /// One convolution with its normalization and its ReLU: [in][length] to [out][length after].
  static void run(const convolution& layer, const float* from, std::size_t length, float* to)
  {
    const std::size_t after = length_after(layer, length);
    std::memset(to, 0, sizeof(float) * layer.out * after);
    for (std::size_t o = 0; o != layer.out; ++o) {
      float* out = to + o * after;
      for (std::size_t i = 0; i != layer.in; ++i) {
        const float* in     = from + i * length;
        const float* weight = layer.weight + (o * layer.in + i) * layer.kernel;
        for (std::size_t j = 0; j != layer.kernel; ++j) {
          // Output t reads input t * stride + j - pad. Only the inputs inside the signal take part.
          const std::size_t first_t = j >= layer.pad ? 0 : (layer.pad - j + layer.stride - 1) / layer.stride;
          if (length + layer.pad <= j) {
            continue;
          }
          const std::size_t last_in = length - 1 + layer.pad - j; // largest t * stride that is inside
          const std::size_t end_t   = std::min(after, last_in / layer.stride + 1);
          const float       w       = weight[j];
          const float*      shifted = in + (first_t * layer.stride + j - layer.pad); // never below `in`
          for (std::size_t t = first_t; t < end_t; ++t, shifted += layer.stride) {
            out[t] += w * *shifted;
          }
        }
      }
      const float scale = layer.scale[o], shift = layer.shift[o];
      for (std::size_t t = 0; t != after; ++t) {
        const float value = out[t] * scale + shift;
        out[t]            = value > 0.0f ? value : 0.0f;
      }
    }
  }

  static void apply(const linear& layer, const float* from, float* to, bool relu)
  {
    for (std::size_t o = 0; o != layer.out; ++o) {
      const float* weight = layer.weight + o * layer.in;
      float        sum    = layer.bias[o];
      for (std::size_t i = 0; i != layer.in; ++i) {
        sum += weight[i] * from[i];
      }
      to[o] = relu && sum < 0.0f ? 0.0f : sum;
    }
  }

  std::uint32_t            rows, cols, positions;
  std::string              hash;
  std::vector<convolution> layers; // the stem, then two per block
  linear                   trunk[2], location, presence;
  std::vector<float>       first, second, pooled, hidden[2];
  engine_binding           bound;
};

} // namespace

std::unique_ptr<engine> make_host_reference_engine(const artifact& model)
{
  try {
    return std::unique_ptr<engine>(new host_reference_engine(model));
  } catch (const artifact_error& fault) {
    throw engine_error(fault.what());
  }
}

} // namespace sense

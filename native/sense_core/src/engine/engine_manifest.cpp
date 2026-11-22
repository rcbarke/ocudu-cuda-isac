// The manifest beside an engine, and the check that refuses an engine anywhere but where it was
// built (Sprint 3.6, selector sense_core).
//
// An engine directory holds the engine file and engine.json, a sense-engine/v1 object. It names
// the artifact the engine was built from by its hash, the library and its version, the precision,
// the GPU it was built on and the hash of the engine file. A plan loads an engine only when all
// of that is what runs it now: the same artifact, the same library version, the same GPU. An
// engine is rebuilt on every GPU that runs it.
//
// This file holds no call of TensorRT, LibTorch or CUDA, so the refusals are unit tested on a
// host that has none of them. ASCII only.

#include "core/sha256.h"
#include "engine/engine.h"

namespace sense {

namespace {

std::string member(const json::value& object, const char* key)
{
  const json::value* found = object.find(key);
  return found != nullptr && found->is_string() ? found->text : std::string();
}

} // namespace

std::string check_engine_manifest(const std::string& directory, const artifact& model, const engine_host& host,
                                  const char* file_name)
{
  std::string text;
  try {
    text = read_file(directory + "/" + ENGINE_MANIFEST_FILE);
  } catch (const artifact_error&) {
    throw engine_error(directory + " holds no " + ENGINE_MANIFEST_FILE + ": it is not an engine directory. Build one: "
                       "python scripts/native_core.py engine --artifact <artifact>");
  }
  try {
    const json::value manifest = json::parse(text);
    if (!manifest.is_object() || manifest.string_at("schema") != ENGINE_MANIFEST_SCHEMA) {
      throw engine_error(std::string(ENGINE_MANIFEST_FILE) + " is not a " + ENGINE_MANIFEST_SCHEMA + " object");
    }
    if (manifest.string_at("backend") != host.backend) {
      throw engine_error("the engine is one of the backend " + manifest.string_at("backend") + ", and the plan asks for " +
                         host.backend);
    }
    if (manifest.string_at("artifact_sha256") != model.sha256) {
      throw engine_error("the engine was built from the artifact " + manifest.string_at("artifact_sha256") +
                         ", and the plan holds " + model.sha256 + ": an engine carries the hash of its artifact");
    }
    if (manifest.string_at("weights_sha256") != model.weights_sha256) {
      throw engine_error("the engine was built from other weights than the artifact holds");
    }
    const json::value& library = manifest.object_at("library");
    if (library.string_at("version") != host.library_version) {
      throw engine_error("the engine was built with " + library.string_at("name") + " " + library.string_at("version") +
                         ", and this host runs " + host.library_version + ": rebuild the engine");
    }
    // The GPU. A unique identifier decides where both sides have one, and the name and the
    // compute capability otherwise.
    const json::value& built = manifest.object_at("device");
    const json::value  here  = json::parse(host.device_identity);
    const std::string  built_uuid = member(built, "uuid"), here_uuid = member(here, "uuid");
    const bool same = member(built, "name") == member(here, "name") &&
                      member(built, "capability") == member(here, "capability") &&
                      (built_uuid.empty() || here_uuid.empty() || built_uuid == here_uuid);
    if (!same) {
      throw engine_error("the engine was built for " + member(built, "name") + " (capability " + member(built, "capability") +
                         ", " + (built_uuid.empty() ? std::string("no identifier") : built_uuid) + "), and this is " +
                         member(here, "name") + " (capability " + member(here, "capability") + ", " +
                         (here_uuid.empty() ? std::string("no identifier") : here_uuid) +
                         "): an engine is built on the GPU that runs it");
    }
    const json::value& shape = manifest.object_at("input").array_at("shape");
    const json::value& cells = manifest.object_at("outputs").array_at(ENGINE_MAP);
    if (shape.items.size() != 3 || shape.items[0].number != 1 || shape.items[1].number != model.rx ||
        shape.items[2].number != model.subcarriers || cells.items.size() != 2 ||
        cells.items[1].number != static_cast<double>(model.rows) * model.cols) {
      throw engine_error("the engine does not take [1, A, K] of the artifact or does not give its grid");
    }
    if (manifest.string_at("file") != file_name) {
      throw engine_error(std::string("the manifest names the file ") + manifest.string_at("file") + ", and the backend "
                         "loads " + file_name);
    }
    const std::string data = read_file(directory + "/" + file_name);
    const std::string hash = sha256_hex(data);
    if (static_cast<std::size_t>(manifest.integer_at("bytes")) != data.size() || manifest.string_at("sha256") != hash) {
      throw engine_error(std::string(file_name) + " does not have the size and the hash its manifest names: the engine "
                         "was altered after it was built");
    }
    return hash;
  } catch (const json::error& fault) {
    throw engine_error(std::string(ENGINE_MANIFEST_FILE) + " is incomplete: " + fault.what());
  } catch (const artifact_error& fault) {
    throw engine_error(fault.what());
  }
}

} // namespace sense

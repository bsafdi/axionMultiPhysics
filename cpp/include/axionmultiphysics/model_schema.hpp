// The JSON Schema of the model document (beta), added to the schema of the electromagnetic design format.
//
// The schema of a design is one schema for both kinds of document: `if` the key "physics" is present, `then` the rules
// of a model document, `else` the electromagnetic rules of 0.1.0, unchanged.  The root keeps its `properties` and
// `additionalProperties: false`: the electromagnetic sections stay root properties, the sections of a model document
// are added to them (each marked "x-status": "beta"), and the sections that both kinds share (regions, materials,
// mesh) are the permissive union at the root and strict inside `then` and `else`.  What a physics type offers
// (options, condition kinds and their data keys, coupling terms) is generated from the physics registry.
#pragma once
#include <nlohmann/json.hpp>

namespace axmp::model {

// `em` is the schema of the electromagnetic design format; returns the schema of both kinds of document.
nlohmann::json add_model_branch(const nlohmann::json& em);

}  // namespace axmp::model

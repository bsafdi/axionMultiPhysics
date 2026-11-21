// Explanation and patching of a model document (beta).
//
// explain_model lists the resolved model of a document: what each section means after the defaults are applied, which
// boundary conditions each physics has and the form the engine gives them, the coupling terms, the study with its
// coupling strategy and solver defaults, and the route each group of physics takes with what it needs.  It refuses a
// document that validate_model refuses (the first error) and an electromagnetic design.
//
// patch_document applies an RFC 6902 JSON Patch to a copy of a document, one operation at a time, and validates the
// result (a model document with validate_model, an electromagnetic design with the electromagnetic parser).  Either
// every operation applies and the result validates, or an axmp::Error is thrown and the document is unchanged; an
// operation that cannot be applied is the code patch.operation_failed with the index of the operation.
#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include "axionmultiphysics/core.hpp"

namespace axmp::model {

// The listing as text (`as_json` false) or as the JSON text of schema "axionmultiphysics.explain/1" (`as_json` true).
std::string explain_model(const nlohmann::json& doc, bool as_json);

// The patched document.
nlohmann::json patch_document(const nlohmann::json& doc, const nlohmann::json& ops);

}  // namespace axmp::model

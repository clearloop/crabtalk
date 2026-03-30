use serde_json::{Map, Value};

const MAX_DEPTH: u32 = 32;

/// Resolve all `$ref` pointers from `$defs` and remove the `$defs` key.
///
/// Walks the schema tree, replacing `{"$ref": "#/$defs/Name"}` with the
/// corresponding definition body. Handles transitive references within
/// definitions. Depth-capped at 32 to guard against pathological input.
pub fn inline_refs(schema: &mut Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    let Some(Value::Object(defs)) = obj.remove("$defs") else {
        return;
    };
    resolve(schema, &defs, 0);
}

fn resolve(node: &mut Value, defs: &Map<String, Value>, depth: u32) {
    if depth > MAX_DEPTH {
        return;
    }
    match node {
        Value::Object(map) => {
            if let Some(Value::String(r)) = map.get("$ref")
                && let Some(name) = r.strip_prefix("#/$defs/")
                && let Some(def) = defs.get(name)
            {
                *node = def.clone();
                resolve(node, defs, depth + 1);
                return;
            }
            for v in map.values_mut() {
                resolve(v, defs, depth);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                resolve(v, defs, depth);
            }
        }
        _ => {}
    }
}

/// Remove `$schema` and `$id` meta-annotations from the top level.
pub fn strip_schema_meta(schema: &mut Value) {
    if let Some(obj) = schema.as_object_mut() {
        obj.remove("$schema");
        obj.remove("$id");
    }
}

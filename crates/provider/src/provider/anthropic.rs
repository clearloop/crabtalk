use crabtalk_core::Error;

pub fn not_implemented(name: &str) -> Error {
    Error::Internal(format!("anthropic {name} not yet implemented"))
}

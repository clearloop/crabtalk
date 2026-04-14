use utoipa::openapi::{
    ComponentsBuilder, HttpMethod, InfoBuilder, PathItem, PathsBuilder, Tag,
    path::{OperationBuilder, ParameterBuilder, ParameterIn},
    security::{HttpAuthScheme, HttpBuilder, SecurityRequirement, SecurityScheme},
};

const TAG_API: &str = "API";
const TAG_ADMIN_KEYS: &str = "Admin / Keys";
const TAG_ADMIN_PROVIDERS: &str = "Admin / Providers";
const TAG_ADMIN_USAGE: &str = "Admin / Usage";
const TAG_INFRA: &str = "Infrastructure";

fn op(tag: &str, summary: &str) -> OperationBuilder {
    OperationBuilder::new().summary(Some(summary)).tag(tag)
}

/// Build a PathItem with multiple HTTP methods sharing one tag.
fn multi(tag: &str, ops: &[(HttpMethod, &str)]) -> PathItem {
    let mut item = PathItem::default();
    for (method, summary) in ops {
        let operation = op(tag, summary).build();
        match method {
            HttpMethod::Get => item.get = Some(operation),
            HttpMethod::Post => item.post = Some(operation),
            HttpMethod::Put => item.put = Some(operation),
            HttpMethod::Patch => item.patch = Some(operation),
            HttpMethod::Delete => item.delete = Some(operation),
            _ => {}
        }
    }
    item
}

fn query(name: &str) -> ParameterBuilder {
    ParameterBuilder::new()
        .name(name)
        .parameter_in(ParameterIn::Query)
}

pub fn spec() -> utoipa::openapi::OpenApi {
    let paths = PathsBuilder::new()
        .path(
            "/v1/chat/completions",
            PathItem::new(HttpMethod::Post, op(TAG_API, "Create a chat completion")),
        )
        .path(
            "/v1/messages",
            PathItem::new(
                HttpMethod::Post,
                op(TAG_API, "Create a message (Anthropic format)"),
            ),
        )
        .path(
            "/v1/embeddings",
            PathItem::new(HttpMethod::Post, op(TAG_API, "Create embeddings")),
        )
        .path(
            "/v1/images/generations",
            PathItem::new(HttpMethod::Post, op(TAG_API, "Generate images")),
        )
        .path(
            "/v1/audio/speech",
            PathItem::new(HttpMethod::Post, op(TAG_API, "Generate speech audio")),
        )
        .path(
            "/v1/audio/transcriptions",
            PathItem::new(HttpMethod::Post, op(TAG_API, "Transcribe audio")),
        )
        .path(
            "/v1/models",
            PathItem::new(HttpMethod::Get, op(TAG_API, "List available models")),
        )
        .path(
            "/v1/usage",
            PathItem::new(
                HttpMethod::Get,
                op(TAG_API, "Get usage for the authenticated key").parameter(query("model")),
            ),
        )
        .path(
            "/v1/admin/keys",
            multi(
                TAG_ADMIN_KEYS,
                &[
                    (HttpMethod::Post, "Create a virtual API key"),
                    (HttpMethod::Get, "List all virtual keys"),
                ],
            ),
        )
        .path(
            "/v1/admin/keys/{name}",
            multi(
                TAG_ADMIN_KEYS,
                &[
                    (HttpMethod::Get, "Get key details"),
                    (HttpMethod::Patch, "Update a key (models, rate_limit)"),
                    (HttpMethod::Delete, "Revoke a virtual key"),
                ],
            ),
        )
        .path(
            "/v1/admin/providers",
            multi(
                TAG_ADMIN_PROVIDERS,
                &[
                    (HttpMethod::Post, "Create a provider"),
                    (HttpMethod::Get, "List all providers"),
                ],
            ),
        )
        .path(
            "/v1/admin/providers/{name}",
            multi(
                TAG_ADMIN_PROVIDERS,
                &[
                    (HttpMethod::Get, "Get provider details"),
                    (HttpMethod::Patch, "Update a provider"),
                    (HttpMethod::Delete, "Delete a provider"),
                ],
            ),
        )
        .path(
            "/v1/admin/providers/reload",
            PathItem::new(
                HttpMethod::Post,
                op(TAG_ADMIN_PROVIDERS, "Reload provider registry from config"),
            ),
        )
        .path(
            "/v1/admin/usage",
            PathItem::new(
                HttpMethod::Get,
                op(TAG_ADMIN_USAGE, "Global usage view")
                    .parameter(query("name"))
                    .parameter(query("model")),
            ),
        )
        .path(
            "/v1/admin/logs",
            PathItem::new(
                HttpMethod::Get,
                op(TAG_ADMIN_USAGE, "Query audit logs")
                    .parameter(query("key"))
                    .parameter(query("model"))
                    .parameter(query("since"))
                    .parameter(query("until"))
                    .parameter(query("limit")),
            ),
        )
        .path(
            "/v1/budget",
            PathItem::new(
                HttpMethod::Get,
                op(TAG_ADMIN_USAGE, "Get budget status per key"),
            ),
        )
        .path(
            "/v1/cache",
            PathItem::new(
                HttpMethod::Delete,
                op(TAG_ADMIN_USAGE, "Clear response cache"),
            ),
        )
        .path(
            "/health",
            PathItem::new(HttpMethod::Get, op(TAG_INFRA, "Health check")),
        )
        .path(
            "/metrics",
            PathItem::new(HttpMethod::Get, op(TAG_INFRA, "Prometheus metrics")),
        )
        .build();

    let tags = vec![
        Tag::new(TAG_API),
        Tag::new(TAG_ADMIN_KEYS),
        Tag::new(TAG_ADMIN_PROVIDERS),
        Tag::new(TAG_ADMIN_USAGE),
        Tag::new(TAG_INFRA),
    ];

    utoipa::openapi::OpenApiBuilder::new()
        .info(
            InfoBuilder::new()
                .title("CrabLLM API")
                .version(env!("CARGO_PKG_VERSION"))
                .description(Some("High-performance LLM API gateway"))
                .build(),
        )
        .paths(paths)
        .tags(Some(tags))
        .security(Some(vec![SecurityRequirement::new(
            "BearerAuth",
            Vec::<String>::new(),
        )]))
        .components(Some(
            ComponentsBuilder::new()
                .security_scheme(
                    "BearerAuth",
                    SecurityScheme::Http(HttpBuilder::new().scheme(HttpAuthScheme::Bearer).build()),
                )
                .build(),
        ))
        .build()
}

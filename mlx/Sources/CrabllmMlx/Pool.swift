// Pool.swift — Swift-side multi-model pool with idle eviction.
//
// The pool owns `ModelContainer` instances keyed by local directory
// path. Models are loaded on first request (via `loadModelContainer`)
// and evicted after the idle timeout. An internal `Task` wakes every
// 60 seconds to sweep expired entries.
//
// The pool is an actor so all mutations are serialized by Swift's
// concurrency runtime. FFI entry points bridge from the blocking C
// ABI via `blockingAwait` — same contract as the session functions.

import Foundation
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

private let DEFAULT_IDLE_TIMEOUT: TimeInterval = 30 * 60

// MARK: - Pool actor

actor MlxPool {
    private var models: [String: Entry] = [:]
    private let idleTimeout: TimeInterval
    private var monitorTask: Task<Void, Never>?

    struct Entry {
        let container: ModelContainer
        var lastUsed: Date
    }

    init(idleTimeout: TimeInterval) {
        self.idleTimeout = idleTimeout > 0 ? idleTimeout : DEFAULT_IDLE_TIMEOUT
    }

    func start() {
        monitorTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(60))
                await self?.evictExpired()
            }
        }
    }

    func ensureLoaded(_ modelDir: String) async throws -> ModelContainer {
        if var entry = models[modelDir] {
            entry.lastUsed = Date()
            models[modelDir] = entry
            return entry.container
        }

        let url = URL(fileURLWithPath: modelDir)
        let container = try await loadModelContainer(
            from: url,
            using: #huggingFaceTokenizerLoader()
        )
        models[modelDir] = Entry(container: container, lastUsed: Date())
        return container
    }

    func evict(_ modelDir: String) {
        models.removeValue(forKey: modelDir)
    }

    func stopAll() {
        models.removeAll()
        monitorTask?.cancel()
        monitorTask = nil
    }

    private func evictExpired() {
        let now = Date()
        let expired = models.filter { now.timeIntervalSince($0.value.lastUsed) > idleTimeout }
        for key in expired.keys {
            models.removeValue(forKey: key)
        }
    }

    deinit {
        monitorTask?.cancel()
    }
}

// MARK: - FFI wrappers

// The pool handle is an `Unmanaged<MlxPoolBox>` because actors cannot
// be directly retained via `Unmanaged`. We box the actor in a plain
// class and manage that.
private final class MlxPoolBox: @unchecked Sendable {
    let pool: MlxPool
    init(_ pool: MlxPool) { self.pool = pool }
}

@_cdecl("crabllm_mlx_pool_new")
public func crabllm_mlx_pool_new(
    _ idleTimeoutSecs: UInt64,
    _ outPool: UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
    _ outError: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let outPool = outPool else {
        if let outError = outError {
            outError.pointee = cString("out_pool is NULL")
        }
        return CRABLLM_MLX_ERR_INVALID_ARG
    }

    let pool = MlxPool(idleTimeout: TimeInterval(idleTimeoutSecs))

    // Start the idle monitor via blockingAwait since actor methods are
    // async. `start()` is fast (just spawns a Task), so this is
    // effectively instant.
    do {
        try blockingAwait { await pool.start() }
    } catch {
        if let outError = outError {
            outError.pointee = cString("pool start failed: \(error)")
        }
        return CRABLLM_MLX_ERR_UNKNOWN
    }

    let box = MlxPoolBox(pool)
    outPool.pointee = Unmanaged.passRetained(box).toOpaque()
    return CRABLLM_MLX_OK
}

@_cdecl("crabllm_mlx_pool_free")
public func crabllm_mlx_pool_free(_ pool: UnsafeMutableRawPointer?) {
    guard let pool = pool else { return }
    let box = Unmanaged<MlxPoolBox>.fromOpaque(pool).takeRetainedValue()
    // Best-effort stop. If the caller forgot to call pool_stop_all,
    // at least cancel the monitor task so it doesn't leak.
    _ = box  // ARC releases the box, which deinits the actor, which cancels the monitor.
}

@_cdecl("crabllm_mlx_pool_generate")
public func crabllm_mlx_pool_generate(
    _ pool: UnsafeMutableRawPointer?,
    _ modelDirPath: UnsafePointer<CChar>?,
    _ request: UnsafeRawPointer?,
    _ result: UnsafeMutableRawPointer?
) -> Int32 {
    guard let result = result else { return CRABLLM_MLX_ERR_INVALID_ARG }
    resultClear(result)

    guard let pool = pool else {
        resultSetError(result, "pool is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let modelDir = swiftString(modelDirPath), !modelDir.isEmpty else {
        resultSetError(result, "model_dir_path is NULL or empty")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let request = request else {
        resultSetError(result, "request is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let view = parseRequest(request) else {
        resultSetError(result, "messages_json is NULL, empty, or not valid UTF-8")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }

    let actor = Unmanaged<MlxPoolBox>.fromOpaque(pool).takeUnretainedValue().pool
    do {
        let container = try blockingAwait { try await actor.ensureLoaded(modelDir) }
        let out = try runGenerationWithContainer(container, view, onChunk: nil)
        resultSetText(result, out.text)
        resultSetToolCallsJson(result, out.toolCallsJson)
        resultSetPromptTokens(result, out.promptTokens)
        resultSetCompletionTokens(result, out.completionTokens)
        return CRABLLM_MLX_OK
    } catch let e as FFIError {
        resultSetError(result, e.message)
        return e.status
    } catch {
        resultSetError(result, "pool generate error: \(error)")
        return CRABLLM_MLX_ERR_UNKNOWN
    }
}

@_cdecl("crabllm_mlx_pool_generate_stream")
public func crabllm_mlx_pool_generate_stream(
    _ pool: UnsafeMutableRawPointer?,
    _ modelDirPath: UnsafePointer<CChar>?,
    _ request: UnsafeRawPointer?,
    _ tokenCb: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Int32)?,
    _ userData: UnsafeMutableRawPointer?,
    _ result: UnsafeMutableRawPointer?
) -> Int32 {
    guard let result = result else { return CRABLLM_MLX_ERR_INVALID_ARG }
    resultClear(result)

    guard let pool = pool else {
        resultSetError(result, "pool is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let modelDir = swiftString(modelDirPath), !modelDir.isEmpty else {
        resultSetError(result, "model_dir_path is NULL or empty")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let request = request else {
        resultSetError(result, "request is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let view = parseRequest(request) else {
        resultSetError(result, "messages_json is NULL, empty, or not valid UTF-8")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }
    guard let tokenCb = tokenCb else {
        resultSetError(result, "token_cb is NULL")
        return CRABLLM_MLX_ERR_INVALID_ARG
    }

    let actor = Unmanaged<MlxPoolBox>.fromOpaque(pool).takeUnretainedValue().pool
    do {
        let container = try blockingAwait { try await actor.ensureLoaded(modelDir) }
        let out = try runGenerationWithContainer(container, view, onChunk: { chunk in
            let stop = chunk.withCString { ptr -> Int32 in
                tokenCb(ptr, userData)
            }
            return stop != 0
        })
        resultSetToolCallsJson(result, out.toolCallsJson)
        resultSetPromptTokens(result, out.promptTokens)
        resultSetCompletionTokens(result, out.completionTokens)
        _ = out.text
        return CRABLLM_MLX_OK
    } catch let e as FFIError {
        resultSetError(result, e.message)
        return e.status
    } catch {
        resultSetError(result, "pool generate_stream error: \(error)")
        return CRABLLM_MLX_ERR_UNKNOWN
    }
}

@_cdecl("crabllm_mlx_pool_evict")
public func crabllm_mlx_pool_evict(
    _ pool: UnsafeMutableRawPointer?,
    _ modelDirPath: UnsafePointer<CChar>?
) {
    guard let pool = pool, let dir = swiftString(modelDirPath) else { return }
    let actor = Unmanaged<MlxPoolBox>.fromOpaque(pool).takeUnretainedValue().pool
    try? blockingAwait { await actor.evict(dir) }
}

@_cdecl("crabllm_mlx_pool_stop_all")
public func crabllm_mlx_pool_stop_all(_ pool: UnsafeMutableRawPointer?) {
    guard let pool = pool else { return }
    let actor = Unmanaged<MlxPoolBox>.fromOpaque(pool).takeUnretainedValue().pool
    try? blockingAwait { await actor.stopAll() }
}

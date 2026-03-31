TARGET := x86_64-unknown-linux-gnu
LINUX_AMD64_ENV := CC=x86_64-linux-gnu-gcc AR=x86_64-linux-gnu-ar

.PHONY: prod docker bench-runner bench-image bench bench-chart

# Build crabllm prod binary for linux-amd64
prod:
	$(LINUX_AMD64_ENV) cargo build --profile prod -p crabllm --target $(TARGET)
	mkdir -p dist
	cp target/$(TARGET)/prod/crabllm dist/crabllm

# Build Docker image from pre-built binary
docker: prod
	docker build -t crabllm:local .

# Base image: debian-slim + oha + curl + jq
bench-runner:
	docker build -t crabllm-bench-runner:local -f docker/bench-runner.Dockerfile docker

# Stage prod binaries and build the bench image
bench-image: bench-runner
	mkdir -p crates/bench/bin
	cp target/prod/crabllm target/prod/crabllm-bench crates/bench/bin/
	cd crates/bench && docker compose build

# Run the full competitive benchmark
bench: bench-image
	cd crates/bench && mkdir -p results && \
	docker compose up -d mock crabllm bifrost litellm && \
	BENCH_ARGS="$(ARGS)" docker compose up runner ; \
	docker compose down

# Generate charts from results
bench-chart:
	cd crates/bench && python3 bench.py --chart-only --output results

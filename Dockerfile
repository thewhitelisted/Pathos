# syntax=docker/dockerfile:1
FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY web ./web
RUN cargo build --release --locked

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/pathos /usr/local/bin/pathos
ENV PATHOS_CACHE_DIR=/data
VOLUME /data
EXPOSE 3000
# FinBERT weights (~440 MB) are downloaded and checksum-verified on first start
# and cached in the /data volume.
CMD ["pathos", "serve", "--addr", "0.0.0.0:3000"]

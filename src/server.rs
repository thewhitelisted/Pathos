//! Web dashboard and JSON API.
//!
//! - `GET  /`             the single-page dashboard (embedded in the binary)
//! - `GET  /api/health`   liveness check
//! - `GET  /api/defaults` default analysis parameters
//! - `POST /api/analyze`  run an analysis; body is [`AnalysisParams`] as JSON
//! - `GET  /api/evaluation` the latest `pathos evaluate` report, if any

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use tokio::sync::Semaphore;

use crate::params::AnalysisParams;
use crate::pipeline::Analyzer;

const INDEX_HTML: &str = include_str!("../web/index.html");

#[derive(Clone)]
struct AppState {
    analyzer: Arc<Analyzer>,
    /// Inference is CPU-bound; running many analyses at once only makes all
    /// of them slow, so queue beyond a small limit.
    permits: Arc<Semaphore>,
}

pub fn router(analyzer: Arc<Analyzer>) -> Router {
    let state = AppState {
        analyzer,
        permits: Arc::new(Semaphore::new(2)),
    };
    Router::new()
        .route("/", get(index))
        .route(
            "/api/health",
            get(|| async { Json(json!({ "status": "ok" })) }),
        )
        .route(
            "/api/defaults",
            get(|| async { Json(AnalysisParams::default()) }),
        )
        .route("/api/analyze", post(analyze))
        .route("/api/evaluation", get(evaluation))
        .with_state(state)
}

pub async fn serve(analyzer: Arc<Analyzer>, addr: SocketAddr) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("dashboard listening on http://{}", listener.local_addr()?);
    axum::serve(listener, router(analyzer))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

/// Where `pathos evaluate` installs its latest report for the dashboard.
pub fn latest_evaluation_path() -> std::path::PathBuf {
    crate::http::cache_root().join("evaluation.json")
}

async fn evaluation() -> Response {
    match tokio::fs::read_to_string(latest_evaluation_path()).await {
        Ok(body) => ([(header::CONTENT_TYPE, "application/json")], body).into_response(),
        Err(_) => error(
            StatusCode::NOT_FOUND,
            "no evaluation yet — run `pathos evaluate` to measure the signal on historical data",
        ),
    }
}

async fn index() -> impl IntoResponse {
    ([(header::CACHE_CONTROL, "no-cache")], Html(INDEX_HTML))
}

async fn analyze(State(state): State<AppState>, Json(params): Json<AnalysisParams>) -> Response {
    let params = match params.validated() {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("{e:#}")),
    };
    let Ok(_permit) = state.permits.acquire().await else {
        return error(StatusCode::SERVICE_UNAVAILABLE, "server is shutting down");
    };
    tracing::info!(tickers = ?params.tickers, "analysis requested");
    match state.analyzer.analyze(params).await {
        Ok(report) => Json(report).into_response(),
        Err(e) => {
            tracing::warn!(error = %format!("{e:#}"), "analysis failed");
            error(StatusCode::UNPROCESSABLE_ENTITY, &format!("{e:#}"))
        }
    }
}

fn error(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({ "error": message }))).into_response()
}

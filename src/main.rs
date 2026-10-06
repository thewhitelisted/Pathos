use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use pathos::data::normalize_ticker;
use pathos::http::Fetcher;
use pathos::model::calibration::Calibration;
use pathos::params::AnalysisParams;
use pathos::pipeline::Analyzer;
use pathos::research::evaluate::{DEFAULT_UNIVERSE, EvalParams, evaluate};
use pathos::sentiment::SentimentModel;
use pathos::sentiment::download::{default_model_dir, ensure_model};
use pathos::sentiment::finbert::FinBert;
use pathos::sentiment::store::ScoreStore;

#[derive(Parser)]
#[command(version, about = "Sentiment-aware Black-Litterman portfolio optimizer")]
struct Cli {
    /// Directory holding (or to download) the FinBERT weights.
    #[arg(long, global = true, env = "PATHOS_MODEL_DIR")]
    model_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Analyze tickers and print a recommended allocation.
    Analyze(AnalyzeArgs),
    /// Start the web dashboard.
    Serve {
        #[arg(long, default_value = "127.0.0.1:3000", env = "PATHOS_ADDR")]
        addr: SocketAddr,
        #[command(flatten)]
        calibration: CalibrationArgs,
    },
    /// Measure whether the signals predict returns on historical data, fit the
    /// signal-to-view mapping, and run a walk-forward backtest.
    Evaluate(EvaluateArgs),
    /// Score one or more sentences with FinBERT and print the probabilities.
    Score {
        #[arg(required = true)]
        texts: Vec<String>,
    },
    /// Download and verify the FinBERT weights, then exit.
    DownloadModel,
}

#[derive(Args)]
struct CalibrationArgs {
    /// Calibration file from `pathos evaluate` [default: the one saved in
    /// the cache directory by the last evaluation, if any].
    #[arg(long)]
    calibration: Option<PathBuf>,
    /// Ignore any calibration and use the hand-tuned view mapping.
    #[arg(long)]
    no_calibration: bool,
}

impl CalibrationArgs {
    fn load(&self) -> Result<Option<Calibration>> {
        if self.no_calibration {
            return Ok(None);
        }
        let (path, explicit) = match &self.calibration {
            Some(p) => (p.clone(), true),
            None => (Calibration::default_path(), false),
        };
        if !explicit && !path.exists() {
            return Ok(None);
        }
        let cal =
            Calibration::load(&path).with_context(|| format!("reading {}", path.display()))?;
        tracing::info!(
            "using calibrated views from {}: {}",
            path.display(),
            cal.summary()
        );
        Ok(Some(cal))
    }
}

#[derive(Args)]
struct EvaluateArgs {
    /// Universe to evaluate [default: 24 liquid large caps across sectors].
    tickers: Vec<String>,
    /// Years of history to evaluate, ending today.
    #[arg(long, default_value_t = 2.0)]
    years: f64,
    /// Maximum headlines per ticker per week from the news archive.
    #[arg(long, default_value_t = 15)]
    per_week: usize,
    /// Use a CSV of dated headlines (date, ticker, title columns) instead of
    /// the Google News archive, e.g. FNSPID.
    #[arg(long)]
    headlines_csv: Option<PathBuf>,
    /// Minimum headlines in the 7-day window for a ticker to have a signal.
    #[arg(long, default_value_t = 3)]
    min_headlines: usize,
    /// Rebalance interval in trading days for the backtest.
    #[arg(long, default_value_t = 21)]
    rebalance_days: usize,
    /// One-way transaction cost in basis points.
    #[arg(long, default_value_t = 10.0)]
    cost_bps: f64,
    /// Maximum weight per position in the backtest.
    #[arg(long, default_value_t = AnalysisParams::default().max_weight)]
    max_weight: f64,
    /// Directory for evaluation.json and calibration.json.
    #[arg(long, default_value = "evaluation")]
    out: PathBuf,
    /// Do not install the fitted calibration for `analyze` / `serve`.
    #[arg(long)]
    no_install: bool,
}

#[derive(Args)]
struct AnalyzeArgs {
    /// Tickers to allocate across, e.g. AAPL MSFT NVDA.
    #[arg(required = true, num_args = 2..)]
    tickers: Vec<String>,
    /// Cash to allocate.
    #[arg(long, default_value_t = AnalysisParams::default().budget)]
    budget: f64,
    /// Maximum weight per position (0–1).
    #[arg(long, default_value_t = AnalysisParams::default().max_weight)]
    max_weight: f64,
    /// Days of price history for the risk model.
    #[arg(long, default_value_t = AnalysisParams::default().lookback_days)]
    lookback_days: u32,
    /// Only score headlines from the last N days.
    #[arg(long, default_value_t = AnalysisParams::default().news_days)]
    news_days: u32,
    /// Risk aversion δ.
    #[arg(long, default_value_t = AnalysisParams::default().risk_aversion)]
    risk_aversion: f64,
    /// Black-Litterman τ.
    #[arg(long, default_value_t = AnalysisParams::default().tau)]
    tau: f64,
    /// View tilt κ, in units of each asset's volatility.
    #[arg(long, default_value_t = AnalysisParams::default().view_scale)]
    view_scale: f64,
    /// Share of the signal from news sentiment vs fundamentals (0–1).
    #[arg(long, default_value_t = AnalysisParams::default().sentiment_weight)]
    sentiment_weight: f64,
    /// Also write the full report as JSON to this path.
    #[arg(long)]
    json: Option<PathBuf>,
    #[command(flatten)]
    calibration: CalibrationArgs,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "pathos=info".into()))
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let model_dir = cli.model_dir.unwrap_or_else(default_model_dir);
    let fetcher = Fetcher::new()?;
    ensure_model(fetcher.client(), &model_dir)
        .await
        .context("fetching FinBERT weights")?;
    if let Command::DownloadModel = cli.command {
        println!("FinBERT weights ready in {}", model_dir.display());
        return Ok(());
    }

    let dir = model_dir.clone();
    let model = tokio::task::spawn_blocking(move || FinBert::load(&dir))
        .await?
        .with_context(|| format!("loading FinBERT from {}", model_dir.display()))?;
    if let Command::Score { texts } = &cli.command {
        for (text, p) in texts.iter().zip(model.predict(texts)?) {
            println!(
                "{:+.3}  pos {:.4}  neg {:.4}  neu {:.4}  {text}",
                p.score(),
                p.positive,
                p.negative,
                p.neutral
            );
        }
        return Ok(());
    }
    let scores = Arc::new(ScoreStore::open(ScoreStore::default_path())?);
    let model: Arc<dyn SentimentModel> = Arc::new(model);

    match cli.command {
        Command::Evaluate(args) => run_evaluation(&fetcher, model, scores, args).await?,
        Command::Analyze(args) => {
            let analyzer =
                Analyzer::new(fetcher, model, scores).with_calibration(args.calibration.load()?);
            let params = AnalysisParams {
                tickers: args.tickers,
                budget: args.budget,
                max_weight: args.max_weight,
                lookback_days: args.lookback_days,
                news_days: args.news_days,
                risk_aversion: args.risk_aversion,
                tau: args.tau,
                view_scale: args.view_scale,
                sentiment_weight: args.sentiment_weight,
                ..AnalysisParams::default()
            };
            let report = analyzer.analyze(params).await?;
            print!("{}", report.to_text());
            if let Some(path) = args.json {
                std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
                println!("\nFull report written to {}", path.display());
            }
        }
        Command::Serve { addr, calibration } => {
            let analyzer =
                Analyzer::new(fetcher, model, scores).with_calibration(calibration.load()?);
            pathos::server::serve(Arc::new(analyzer), addr).await?
        }
        Command::Score { .. } | Command::DownloadModel => unreachable!(),
    }
    Ok(())
}

async fn run_evaluation(
    fetcher: &Fetcher,
    model: Arc<dyn SentimentModel>,
    scores: Arc<ScoreStore>,
    args: EvaluateArgs,
) -> Result<()> {
    let tickers: Vec<String> = if args.tickers.is_empty() {
        DEFAULT_UNIVERSE.iter().map(|t| t.to_string()).collect()
    } else {
        args.tickers.iter().map(|t| normalize_ticker(t)).collect()
    };
    let end = chrono::Utc::now().date_naive();
    let start = end - chrono::Duration::days((args.years * 365.25).round() as i64);
    let params = EvalParams {
        tickers,
        start,
        end,
        per_week: args.per_week,
        headlines_csv: args.headlines_csv,
        horizons: vec![1, 5, 21],
        calibration_horizon: 21,
        min_headlines: args.min_headlines,
        rebalance_every: args.rebalance_days,
        cost_bps: args.cost_bps,
        max_weight: args.max_weight,
    };
    let report = evaluate(fetcher, model, scores, params).await?;
    print!("{}", report.to_text());

    std::fs::create_dir_all(&args.out)?;
    let calibration = serde_json::to_string_pretty(&report.calibration)?;
    std::fs::write(
        args.out.join("evaluation.json"),
        serde_json::to_string(&report)?,
    )?;
    std::fs::write(args.out.join("calibration.json"), &calibration)?;
    println!(
        "\nWrote {0}/evaluation.json and {0}/calibration.json",
        args.out.display()
    );
    if !args.no_install {
        let path = Calibration::default_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &calibration)?;
        std::fs::write(
            pathos::server::latest_evaluation_path(),
            serde_json::to_string(&report)?,
        )?;
        println!(
            "Installed calibration for `analyze`/`serve` at {} (disable with --no-calibration)",
            path.display()
        );
    }
    Ok(())
}

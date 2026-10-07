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
use pathos::research::evaluate::{DEFAULT_UNIVERSE, EvalParams, SMALL_CAP_UNIVERSE, evaluate};
use pathos::research::universe;
use pathos::sentiment::benchmark::{Variant, compare, sample};
use pathos::sentiment::download::{F32_FILE, Q8_FILE, default_model_dir, ensure_model};
use pathos::sentiment::finbert::FinBert;
use pathos::sentiment::store::ScoreStore;
use pathos::sentiment::{Precision, SentimentModel};

#[derive(Parser)]
#[command(version, about = "Sentiment-aware Black-Litterman portfolio optimizer")]
struct Cli {
    /// Directory holding (or to download) the FinBERT weights.
    #[arg(long, global = true, env = "PATHOS_MODEL_DIR")]
    model_dir: Option<PathBuf>,
    /// FinBERT weight precision: int8 (smaller, faster) or the original f32.
    #[arg(
        long,
        global = true,
        value_enum,
        default_value = "q8",
        env = "PATHOS_PRECISION"
    )]
    precision: Precision,
    /// Keep the 438 MB f32 checkpoint after quantizing it.
    #[arg(long, global = true)]
    keep_f32: bool,

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
    /// Compare the f32, int8 and native-int8 models on cached headlines:
    /// size, speed and agreement.
    BenchmarkModel {
        /// Number of headlines to compare (sampled deterministically).
        #[arg(long, default_value_t = 2000)]
        limit: usize,
    },
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

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Universe {
    /// 24 liquid US large caps across sectors.
    Large,
    /// 34 US small and mid caps across sectors.
    Small,
    /// A random, point-in-time sample of small caps (see --sample, --seed).
    #[value(name = "sampled-small")]
    SampledSmall,
}

#[derive(Args)]
struct EvaluateArgs {
    /// Tickers to evaluate [default: the `--universe` preset].
    tickers: Vec<String>,
    /// Preset universe used when no tickers are given.
    #[arg(long, value_enum, default_value = "large")]
    universe: Universe,
    /// Market proxy for excess returns [default: SPY for large caps, IWM for
    /// small caps].
    #[arg(long)]
    benchmark: Option<String>,
    /// Number of companies to draw for `--universe sampled-small`.
    #[arg(long, default_value_t = 80)]
    sample: usize,
    /// Random seed for `--universe sampled-small`.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Market-cap range on the start date for `--universe sampled-small`, in $.
    #[arg(long, default_value_t = 3e8)]
    min_cap: f64,
    #[arg(long, default_value_t = 3e9)]
    max_cap: f64,
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
    /// Directory for evaluation.json and calibration.json [default:
    /// `evaluation`, or `evaluation/small-caps` for the small-cap preset].
    #[arg(long)]
    out: Option<PathBuf>,
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
    if let Command::BenchmarkModel { limit } = cli.command {
        return benchmark_models(&fetcher, &model_dir, limit).await;
    }
    ensure_model(fetcher.client(), &model_dir, cli.precision, cli.keep_f32)
        .await
        .context("fetching FinBERT weights")?;
    if let Command::DownloadModel = cli.command {
        println!(
            "FinBERT ({}) ready in {}",
            cli.precision.tag(),
            model_dir.display()
        );
        return Ok(());
    }

    let model = load_model(&model_dir, cli.precision).await?;
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
    let scores = Arc::new(ScoreStore::open(ScoreStore::default_path(cli.precision))?);
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
        Command::Score { .. } | Command::DownloadModel | Command::BenchmarkModel { .. } => {
            unreachable!()
        }
    }
    Ok(())
}

async fn load_model(dir: &std::path::Path, precision: Precision) -> Result<FinBert> {
    let d = dir.to_path_buf();
    tokio::task::spawn_blocking(move || FinBert::load(&d, precision))
        .await?
        .with_context(|| {
            format!(
                "loading FinBERT ({}) from {}",
                precision.tag(),
                dir.display()
            )
        })
}

async fn benchmark_models(fetcher: &Fetcher, dir: &std::path::Path, limit: usize) -> Result<()> {
    // Quantize while keeping the f32 checkpoint, so both can be loaded.
    ensure_model(fetcher.client(), dir, Precision::F32, true).await?;
    ensure_model(fetcher.client(), dir, Precision::Q8, true).await?;
    let store = ScoreStore::open(ScoreStore::default_path(Precision::F32))?;
    let texts = sample(store.texts(), limit);
    if texts.is_empty() {
        anyhow::bail!("no cached headlines yet: run `pathos analyze` or `pathos evaluate` first");
    }
    let mb = |f: &str| {
        std::fs::metadata(dir.join(f))
            .map(|m| m.len() as f64 / 1e6)
            .unwrap_or(f64::NAN)
    };
    let (f32_mb, q8_mb) = (mb(F32_FILE), mb(Q8_FILE));
    let f32_model = load_model(dir, Precision::F32).await?;
    let q8_model = load_model(dir, Precision::Q8).await?;
    let native_model = load_model(dir, Precision::Q8Native).await?;
    tracing::info!("scoring {} headlines with each precision", texts.len());
    let result = tokio::task::spawn_blocking(move || {
        compare(
            &[
                Variant {
                    name: "f32",
                    model: &f32_model,
                    file_mb: f32_mb,
                },
                Variant {
                    name: "q8",
                    model: &q8_model,
                    file_mb: q8_mb,
                },
                Variant {
                    name: "q8-native",
                    model: &native_model,
                    file_mb: q8_mb,
                },
            ],
            &texts,
        )
    })
    .await??;
    print!("{}", result.to_text());
    Ok(())
}

async fn run_evaluation(
    fetcher: &Fetcher,
    model: Arc<dyn SentimentModel>,
    scores: Arc<ScoreStore>,
    args: EvaluateArgs,
) -> Result<()> {
    let end = chrono::Utc::now().date_naive();
    let start = end - chrono::Duration::days((args.years * 365.25).round() as i64);
    let small = args.universe != Universe::Large;
    let benchmark = args
        .benchmark
        .map(|b| normalize_ticker(&b))
        .unwrap_or_else(|| if small { "IWM" } else { "SPY" }.to_string());
    let out = args.out.unwrap_or_else(|| {
        PathBuf::from(match args.universe {
            Universe::Large => "evaluation",
            Universe::Small => "evaluation/small-caps",
            Universe::SampledSmall => "evaluation/sampled-small-caps",
        })
    });

    let mut sampled = None;
    let tickers: Vec<String> = if !args.tickers.is_empty() {
        args.tickers.iter().map(|t| normalize_ticker(t)).collect()
    } else {
        match args.universe {
            Universe::Large => DEFAULT_UNIVERSE.iter().map(|t| t.to_string()).collect(),
            Universe::Small => SMALL_CAP_UNIVERSE.iter().map(|t| t.to_string()).collect(),
            Universe::SampledSmall => {
                // Same lookback as the evaluation, so price fetches are reused.
                let lookback = (end - start).num_days() as u32 + 420;
                tracing::info!(
                    "sampling {} companies worth ${:.1}B–${:.1}B on {start} (seed {})",
                    args.sample,
                    args.min_cap / 1e9,
                    args.max_cap / 1e9,
                    args.seed
                );
                let u = universe::sample(
                    fetcher,
                    start,
                    args.sample,
                    args.seed,
                    (args.min_cap, args.max_cap),
                    lookback,
                )
                .await?;
                tracing::info!(
                    "sampled {} of {} candidates after screening {}",
                    u.tickers.len(),
                    u.candidates,
                    u.screened
                );
                let tickers = u.tickers.clone();
                sampled = Some(u);
                tickers
            }
        }
    };
    let params = EvalParams {
        tickers,
        benchmark,
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

    std::fs::create_dir_all(&out)?;
    if let Some(u) = &sampled {
        std::fs::write(out.join("universe.json"), serde_json::to_string_pretty(u)?)?;
    }
    let calibration = serde_json::to_string_pretty(&report.calibration)?;
    std::fs::write(out.join("evaluation.json"), serde_json::to_string(&report)?)?;
    std::fs::write(out.join("calibration.json"), &calibration)?;
    println!(
        "\nWrote {0}/evaluation.json and {0}/calibration.json",
        out.display()
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

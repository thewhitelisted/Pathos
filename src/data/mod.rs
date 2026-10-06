//! Upstream data sources.

pub mod news;
pub mod prices;
pub mod sec;

/// Normalize user input like ` brk.b ` into the `BRK-B` form Yahoo and SEC use.
pub fn normalize_ticker(raw: &str) -> String {
    raw.trim().to_uppercase().replace('.', "-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_tickers() {
        assert_eq!(normalize_ticker(" aapl "), "AAPL");
        assert_eq!(normalize_ticker("brk.b"), "BRK-B");
    }
}

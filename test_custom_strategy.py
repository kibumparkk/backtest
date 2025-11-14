"""
Test script for the custom conditions strategy
"""
import sys
sys.path.append('/home/user/backtest')

from crypto_portfolio_strategy_comparison_fixed import CryptoPortfolioComparisonFixed

def test_custom_strategy():
    """Test the custom conditions strategy on BTC only"""
    print("\n" + "="*80)
    print("Testing Custom Conditions Strategy on BTC_KRW")
    print("="*80 + "\n")

    # Create backtester with only BTC
    comparison = CryptoPortfolioComparisonFixed(
        symbols=['BTC_KRW'],
        start_date='2020-01-01',
        end_date='2023-12-31',
        slippage=0.002
    )

    # Load data
    comparison.load_data()

    # Test the custom strategy
    print("\nRunning custom conditions strategy...")
    df = comparison.data['BTC_KRW'].copy()
    result = comparison.strategy_custom_conditions(df, C1=10, C2=10, C3=0.5, C4=0.5)

    # Print basic statistics
    print("\n" + "="*80)
    print("Strategy Results:")
    print("="*80)
    print(f"\nTotal data points: {len(result)}")
    print(f"Number of trades: {(result['returns'] != 0).sum()}")
    print(f"Final cumulative return: {result['cumulative'].iloc[-1]:.4f}")
    print(f"Total return: {(result['cumulative'].iloc[-1] - 1) * 100:.2f}%")

    # Show sample data
    print("\nSample data (first 10 rows with signals):")
    signal_rows = result[result['returns'] != 0].head(10)
    if len(signal_rows) > 0:
        print(signal_rows[['Open', 'High', 'Low', 'Close', 'position', 'returns', 'cumulative']].to_string())
    else:
        print("No trades found in first rows. Showing overall stats:")
        print(result[['position', 'returns', 'cumulative']].describe())

    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80 + "\n")

    return result

if __name__ == "__main__":
    try:
        result = test_custom_strategy()
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

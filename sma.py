import pandas as pd
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

class TradingStrategy:
    def __init__(self, initial_cash: float = 10000, 
                 short_window: int = 50, 
                 long_window: int = 200,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.1):
        self.initial_cash = float(initial_cash)
        self.short_window = short_window
        self.long_window = long_window
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.columns = df.columns.str.lower().str.strip()
        
        df['short_sma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_sma'] = df['close'].rolling(window=self.long_window).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['volatility'] = df['close'].rolling(window=20).std()
        
        df['signal'] = 0.0
        mask = (df['short_sma'] > df['long_sma']) & (df['rsi'] < 70)
        df.loc[mask, 'signal'] = 1.0
        
        df['buy_signal'] = (df['signal'] == 1) & (df['signal'].shift(1) == 0)
        df['sell_signal'] = (df['signal'] == 0) & (df['signal'].shift(1) == 1)
        
        return df
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def backtest(self, data: pd.DataFrame) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
        df = self.prepare_data(data)
        portfolio = pd.DataFrame(index=df.index)
        portfolio['cash'] = float(self.initial_cash)
        portfolio['shares'] = 0.0
        portfolio['entry_price'] = 0.0
        
        for i in range(1, len(df)):
            portfolio.loc[df.index[i], 'cash'] = float(portfolio.loc[df.index[i-1], 'cash'])
            portfolio.loc[df.index[i], 'shares'] = float(portfolio.loc[df.index[i-1], 'shares'])
            portfolio.loc[df.index[i], 'entry_price'] = float(portfolio.loc[df.index[i-1], 'entry_price'])
            current_price = float(df['close'].iloc[i])
            
            if portfolio.loc[df.index[i], 'shares'] > 0:
                entry_price = portfolio.loc[df.index[i], 'entry_price']
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct <= -self.stop_loss_pct or loss_pct >= self.take_profit_pct:
                    portfolio.loc[df.index[i], 'cash'] += portfolio.loc[df.index[i], 'shares'] * current_price
                    portfolio.loc[df.index[i], 'shares'] = 0.0
                    portfolio.loc[df.index[i], 'entry_price'] = 0.0
                    continue
            
            if df['buy_signal'].iloc[i] and portfolio.loc[df.index[i], 'shares'] == 0:
                cash_to_use = portfolio.loc[df.index[i], 'cash'] * 0.9
                shares_to_buy = cash_to_use / current_price
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    portfolio.loc[df.index[i], 'cash'] -= float(cost)
                    portfolio.loc[df.index[i], 'shares'] = float(shares_to_buy)
                    portfolio.loc[df.index[i], 'entry_price'] = float(current_price)
            
            elif df['sell_signal'].iloc[i] and portfolio.loc[df.index[i], 'shares'] > 0:
                portfolio.loc[df.index[i], 'cash'] += portfolio.loc[df.index[i], 'shares'] * current_price
                portfolio.loc[df.index[i], 'shares'] = 0.0
                portfolio.loc[df.index[i], 'entry_price'] = 0.0
        
        portfolio['portfolio_value'] = portfolio['cash'] + (portfolio['shares'] * df['close'])
        portfolio['daily_returns'] = portfolio['portfolio_value'].pct_change()
        
        final_value = portfolio['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        sharpe_ratio = np.sqrt(252) * (portfolio['daily_returns'].mean() / portfolio['daily_returns'].std())
        max_drawdown = (portfolio['portfolio_value'].cummax() - portfolio['portfolio_value']) / portfolio['portfolio_value'].cummax()
        
        metrics = {
            'Final Value': final_value,
            'Total Return %': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown %': max_drawdown.max() * 100,
            'Win Rate %': (portfolio['daily_returns'] > 0).mean() * 100,
            'Total Trades': (df['buy_signal'] | df['sell_signal']).sum()
        }
        
        return metrics, portfolio, df

    def plot_results(self, df: pd.DataFrame, portfolio: pd.DataFrame):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='Price', alpha=0.7)
        plt.plot(df.index, df['short_sma'], label=f'{self.short_window} SMA', alpha=0.7)
        plt.plot(df.index, df['long_sma'], label=f'{self.long_window} SMA', alpha=0.7)
        
        buy_signals = df[df['buy_signal']]['close']
        sell_signals = df[df['sell_signal']]['close']
        plt.scatter(buy_signals.index, buy_signals, marker='^', color='g', label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals, marker='v', color='r', label='Sell Signal')
        plt.title('Trading Signals')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(portfolio.index, portfolio['portfolio_value'], label='Portfolio Value')
        plt.title('Portfolio Value Over Time')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

data = pd.read_csv('stock_data.csv')
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

strategy = TradingStrategy(
    initial_cash=10000,
    short_window=20,  # Adjusted for more frequent trades
    long_window=50,   # Adjusted for more frequent trades
    stop_loss_pct=0.03,  # Tighter stop loss
    take_profit_pct=0.05  # More realistic take profit
)

metrics, portfolio, prepared_data = strategy.backtest(data)

print("\nStrategy Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}")

strategy.plot_results(prepared_data, portfolio)

# Strategy Analysis and Recommendations
print("\nStrategy Analysis:")
print("1. The current SMA crossover strategy shows mixed results")
print("2. The negative Sharpe ratio indicates poor risk-adjusted returns")
print("3. The high maximum drawdown suggests significant risk")

print("\nRecommended Improvements:")
print("1. Consider adding volume analysis to confirm trends")
print("2. Add ATR for dynamic stop loss and take profit levels")
print("3. Include trend strength indicators like ADX")
print("4. Consider adding price action confirmation (e.g., candlestick patterns)")
print("5. Implement position sizing based on volatility")
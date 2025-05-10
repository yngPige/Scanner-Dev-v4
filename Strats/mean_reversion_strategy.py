
"""
Mean Reversion Strategy Implementation
This module implements a mean reversion strategy based on RSI, Bollinger Bands, and other oscillators.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import talib
from src.trading.strategies import TradingStrategy
from src.utils.ta import SafeTrend, SafeMomentum, SafeVolatility, SafeVolume
import ta

class MeanReversionStrategy(TradingStrategy):
    """
    A mean reversion strategy that looks for overbought/oversold conditions and price deviations from the mean.
    """

    def __init__(self, name="Mean Reversion", description="Identifies overbought/oversold conditions for counter-trend trades"):
        """
        Initialize the mean reversion strategy.

        Args:
            name (str): Name of the strategy
            description (str): Description of the strategy
        """
        super().__init__(name, description)

        # Strategy parameters
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.bb_period = 20
        self.bb_std_dev = 2.0
        self.stoch_k_period = 14
        self.stoch_d_period = 3
        self.stoch_slowing = 3
        self.stoch_overbought = 80
        self.stoch_oversold = 20
        self.cci_period = 20
        self.cci_overbought = 100
        self.cci_oversold = -100

        # Moving average periods (aligned with trend following strategy)
        self.ma_periods = [9, 20, 50, 100, 200]

    def generate_signal(self, symbol, klines, ticker=None):
        """
        Generate a trading signal based on mean reversion indicators.

        Args:
            symbol (str): Trading symbol (e.g., 'BTC-USDT')
            klines (pd.DataFrame): Historical price data
            ticker (dict, optional): Current ticker data

        Returns:
            dict: Signal with recommendation, confidence, etc.
        """
        try:
            # Ensure we have enough data for the longest MA (200) plus some buffer
            if len(klines) < 220:  # 200 + 20 buffer
                return {
                    'source': self.name,
                    'recommendation': 'HOLD',
                    'confidence': 0,
                    'reasoning': f"Insufficient data for {self.name} strategy (need at least 220 periods)",
                    'timestamp': datetime.now().isoformat()
                }

            # Get current price
            current_price = float(ticker['last']) if ticker else klines['close'].iloc[-1]

            # Calculate indicators
            klines = self._calculate_indicators(klines)

            # Analyze mean reversion signals
            reversion_signals, signal_strength, signal_details = self._analyze_reversion_signals(klines)

            # Generate recommendation based on signals
            recommendation, confidence, reasoning = self._generate_recommendation(
                reversion_signals, signal_strength, signal_details, klines
            )

            # Calculate support and resistance levels
            support_levels, resistance_levels = self._calculate_support_resistance(klines)

            # Calculate stop loss and take profit levels
            if recommendation == 'BUY':
                stop_loss = self._calculate_stop_loss(current_price, 'LONG', klines)
                take_profit = self._calculate_take_profit(current_price, stop_loss)
            elif recommendation == 'SELL':
                stop_loss = self._calculate_stop_loss(current_price, 'SHORT', klines)
                take_profit = self._calculate_take_profit(current_price, stop_loss)
            else:  # HOLD
                stop_loss = 0
                take_profit = 0

            # Format and return the signal
            return self._format_signal(
                recommendation=recommendation,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                reasoning=reasoning,
                key_indicators=signal_details
            )

        except Exception as e:
            # Return a HOLD signal with error information
            return {
                'source': self.name,
                'recommendation': 'HOLD',
                'confidence': 0,
                'reasoning': f"Error in {self.name} strategy: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_indicators(self, klines):
        """
        Calculate technical indicators for mean reversion analysis.

        Args:
            klines (pd.DataFrame): Historical price data

        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df = klines.copy()

        # Calculate RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)

        # Calculate Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'],
            timeperiod=self.bb_period,
            nbdevup=self.bb_std_dev,
            nbdevdn=self.bb_std_dev,
            matype=0
        )

        # Calculate Bollinger Band width
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Calculate Bollinger Band %B
        df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Calculate Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'],
            df['low'],
            df['close'],
            fastk_period=self.stoch_k_period,
            slowk_period=self.stoch_slowing,
            slowk_matype=0,
            slowd_period=self.stoch_d_period,
            slowd_matype=0
        )

        # Calculate CCI (Commodity Channel Index)
        df['cci'] = talib.CCI(
            df['high'],
            df['low'],
            df['close'],
            timeperiod=self.cci_period
        )

        # Calculate all Moving Averages
        for period in self.ma_periods:
            df[f'ma_{period}'] = talib.SMA(df['close'], timeperiod=period)

        # Keep the original SMA for backward compatibility
        df['sma'] = df['ma_20']  # 20-period MA

        # Calculate distance from moving averages
        df['ma_distance'] = (df['close'] - df['sma']) / df['sma'] * 100

        # Calculate distance from 200 MA (long-term trend)
        df['ma_200_distance'] = (df['close'] - df['ma_200']) / df['ma_200'] * 100

        return df

    def _analyze_reversion_signals(self, klines):
        """
        Analyze mean reversion signals.

        Args:
            klines (pd.DataFrame): DataFrame with indicators

        Returns:
            tuple: (reversion_signals, signal_strength, signal_details)
        """
        # Get the latest values
        latest = klines.iloc[-1]

        # Initialize signal details list
        signal_details = []

        # Check RSI for overbought/oversold conditions
        rsi_signal = 0
        if latest['rsi'] < self.rsi_oversold:
            rsi_signal = 1  # Oversold - bullish signal
            signal_details.append(f"RSI ({latest['rsi']:.2f}) below {self.rsi_oversold} - Oversold (Bullish)")
        elif latest['rsi'] > self.rsi_overbought:
            rsi_signal = -1  # Overbought - bearish signal
            signal_details.append(f"RSI ({latest['rsi']:.2f}) above {self.rsi_overbought} - Overbought (Bearish)")

        # Check Bollinger Bands
        bb_signal = 0
        if latest['close'] < latest['bb_lower']:
            bb_signal = 1  # Below lower band - bullish signal
            signal_details.append("Price below lower Bollinger Band - Potential buy signal")
        elif latest['close'] > latest['bb_upper']:
            bb_signal = -1  # Above upper band - bearish signal
            signal_details.append("Price above upper Bollinger Band - Potential sell signal")

        # Check Bollinger Band %B
        bb_pct_b_signal = 0
        if latest['bb_pct_b'] < 0:
            bb_pct_b_signal = 1  # Below 0 - bullish signal
            signal_details.append("Bollinger %B below 0 - Extreme oversold (Bullish)")
        elif latest['bb_pct_b'] > 1:
            bb_pct_b_signal = -1  # Above 1 - bearish signal
            signal_details.append("Bollinger %B above 1 - Extreme overbought (Bearish)")

        # Check Stochastic Oscillator
        stoch_signal = 0
        if latest['stoch_k'] < self.stoch_oversold and latest['stoch_d'] < self.stoch_oversold:
            stoch_signal = 1  # Oversold - bullish signal
            signal_details.append(f"Stochastic below {self.stoch_oversold} - Oversold (Bullish)")
        elif latest['stoch_k'] > self.stoch_overbought and latest['stoch_d'] > self.stoch_overbought:
            stoch_signal = -1  # Overbought - bearish signal
            signal_details.append(f"Stochastic above {self.stoch_overbought} - Overbought (Bearish)")

        # Check for Stochastic crossover
        if klines['stoch_k'].iloc[-2] < klines['stoch_d'].iloc[-2] and latest['stoch_k'] > latest['stoch_d']:
            stoch_signal += 0.5  # Bullish crossover
            signal_details.append("Stochastic K crossed above D - Bullish signal")
        elif klines['stoch_k'].iloc[-2] > klines['stoch_d'].iloc[-2] and latest['stoch_k'] < latest['stoch_d']:
            stoch_signal -= 0.5  # Bearish crossover
            signal_details.append("Stochastic K crossed below D - Bearish signal")

        # Check CCI
        cci_signal = 0
        if latest['cci'] < self.cci_oversold:
            cci_signal = 1  # Oversold - bullish signal
            signal_details.append(f"CCI ({latest['cci']:.2f}) below {self.cci_oversold} - Oversold (Bullish)")
        elif latest['cci'] > self.cci_overbought:
            cci_signal = -1  # Overbought - bearish signal
            signal_details.append(f"CCI ({latest['cci']:.2f}) above {self.cci_overbought} - Overbought (Bearish)")

        # Check distance from moving average
        ma_signal = 0
        if latest['ma_distance'] < -5:
            ma_signal = 1  # Price significantly below MA - bullish signal
            signal_details.append(f"Price {abs(latest['ma_distance']):.2f}% below SMA - Potential buy signal")
        elif latest['ma_distance'] > 5:
            ma_signal = -1  # Price significantly above MA - bearish signal
            signal_details.append(f"Price {latest['ma_distance']:.2f}% above SMA - Potential sell signal")

        # Determine overall reversion signals
        reversion_scores = [rsi_signal, bb_signal, bb_pct_b_signal, stoch_signal, cci_signal, ma_signal]
        reversion_signals = sum(reversion_scores)

        # Determine signal strength (0-100)
        max_strength = len([s for s in reversion_scores if s != 0]) * 1  # Maximum possible score from non-zero signals
        if max_strength > 0:
            signal_strength = (abs(reversion_signals) / max_strength) * 100
        else:
            signal_strength = 0

        return reversion_signals, signal_strength, signal_details

    def _generate_recommendation(self, reversion_signals, signal_strength, signal_details, klines):
        """
        Generate a trading recommendation based on mean reversion signals.

        Args:
            reversion_signals (float): Sum of reversion signals
            signal_strength (float): Signal strength (0-100)
            signal_details (list): List of signal descriptions
            klines (pd.DataFrame): DataFrame with indicators

        Returns:
            tuple: (recommendation, confidence, reasoning)
        """
        # Get the latest values
        latest = klines.iloc[-1]

        # Initialize recommendation
        recommendation = "HOLD"
        confidence = 0
        reasoning = ""

        # Generate recommendation based on reversion signals
        if reversion_signals > 1 and signal_strength > 50:
            recommendation = "BUY"
            confidence = signal_strength
            reasoning = f"Mean reversion buy signal with {signal_strength:.2f}% confidence. "
        elif reversion_signals < -1 and signal_strength > 50:
            recommendation = "SELL"
            confidence = signal_strength
            reasoning = f"Mean reversion sell signal with {signal_strength:.2f}% confidence. "
        else:
            recommendation = "HOLD"
            confidence = 100 - signal_strength
            reasoning = f"No clear mean reversion signal or signal strength ({signal_strength:.2f}%) is below threshold. "

        # Add indicator details to reasoning
        reasoning += "Indicator analysis: "
        for detail in signal_details:
            reasoning += detail + ". "

        # Check for trend confirmation or contradiction using all MAs
        if recommendation == "BUY":
            # Count how many MAs the price is below (contrarian buy in downtrend)
            mas_below = 0
            for period in self.ma_periods:
                if latest['close'] < latest[f'ma_{period}']:
                    mas_below += 1

            # Adjust confidence based on trend context
            if mas_below >= 4:  # Price below most MAs (strong downtrend)
                confidence = min(confidence + 15, 100)
                reasoning += f"Price below {mas_below}/5 MAs - Strong mean reversion opportunity. "
            elif mas_below >= 3:  # Price below majority of MAs (moderate downtrend)
                confidence = min(confidence + 10, 100)
                reasoning += f"Price below {mas_below}/5 MAs - Good mean reversion opportunity. "
            elif mas_below <= 1:  # Price above most MAs (uptrend)
                confidence = max(confidence - 15, 0)
                reasoning += f"Warning: Price above most MAs, weak mean reversion opportunity. "

            # Check 200 MA for long-term trend context
            if latest['close'] < latest['ma_200']:
                reasoning += "Price below 200 MA - In long-term downtrend. "
            else:
                reasoning += "Price above 200 MA - Counter to long-term uptrend. "

            # Check if volume is increasing (good for reversal)
            if 'volume' in klines.columns and klines['volume'].iloc[-1] > klines['volume'].iloc[-5:].mean():
                confidence = min(confidence + 10, 100)
                reasoning += "Volume increasing, supporting reversal signal. "

        elif recommendation == "SELL":
            # Count how many MAs the price is above (contrarian sell in uptrend)
            mas_above = 0
            for period in self.ma_periods:
                if latest['close'] > latest[f'ma_{period}']:
                    mas_above += 1

            # Adjust confidence based on trend context
            if mas_above >= 4:  # Price above most MAs (strong uptrend)
                confidence = min(confidence + 15, 100)
                reasoning += f"Price above {mas_above}/5 MAs - Strong mean reversion opportunity. "
            elif mas_above >= 3:  # Price above majority of MAs (moderate uptrend)
                confidence = min(confidence + 10, 100)
                reasoning += f"Price above {mas_above}/5 MAs - Good mean reversion opportunity. "
            elif mas_above <= 1:  # Price below most MAs (downtrend)
                confidence = max(confidence - 15, 0)
                reasoning += f"Warning: Price below most MAs, weak mean reversion opportunity. "

            # Check 200 MA for long-term trend context
            if latest['close'] > latest['ma_200']:
                reasoning += "Price above 200 MA - In long-term uptrend. "
            else:
                reasoning += "Price below 200 MA - Counter to long-term downtrend. "

            # Check if volume is increasing (good for reversal)
            if 'volume' in klines.columns and klines['volume'].iloc[-1] > klines['volume'].iloc[-5:].mean():
                confidence = min(confidence + 10, 100)
                reasoning += "Volume increasing, supporting reversal signal. "

        return recommendation, confidence, reasoning

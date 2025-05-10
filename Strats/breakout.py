"""
Breakout Strategy Implementation
This module implements a breakout strategy based on support/resistance levels and volatility breakouts.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import talib
import src.trading.strategies TradingStrategy

class BreakoutStrategy(TradingStrategy):
    """
    A breakout strategy that identifies price breakouts from consolidation patterns,
    support/resistance levels, and volatility contractions.
    """

    def __init__(self, name="Breakout", description="Identifies price breakouts from key levels and patterns"):
        """
        Initialize the breakout strategy.

        Args:
            name (str): Name of the strategy
            description (str): Description of the strategy
        """
        super().__init__(name, description)

        # Strategy parameters
        self.atr_period = 14
        self.bb_period = 20
        self.bb_std_dev = 2.0
        self.keltner_period = 20
        self.keltner_atr_multiple = 1.5
        self.volume_threshold = 1.5  # Volume increase threshold for breakout confirmation
        self.lookback_periods = 20  # Periods to look back for support/resistance
        self.consolidation_threshold = 0.03  # Max price range for consolidation (3%)

        # Moving average periods (aligned with other strategies)
        self.ma_periods = [9, 20, 50, 100, 200]

    def generate_signal(self, symbol, klines, ticker=None):
        """
        Generate a trading signal based on breakout patterns.

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

            # Identify breakout patterns
            breakout_direction, breakout_strength, breakout_details = self._identify_breakouts(klines)

            # Generate recommendation based on breakout analysis
            recommendation, confidence, reasoning = self._generate_recommendation(
                breakout_direction, breakout_strength, breakout_details, klines
            )

            # Calculate support and resistance levels
            support_levels, resistance_levels = self._calculate_support_resistance(klines, num_levels=5)

            # Calculate stop loss and take profit levels
            if recommendation == 'BUY':
                stop_loss = self._calculate_stop_loss(current_price, 'LONG', klines)
                take_profit = self._calculate_take_profit(current_price, stop_loss, risk_reward_ratio=2.5)
            elif recommendation == 'SELL':
                stop_loss = self._calculate_stop_loss(current_price, 'SHORT', klines)
                take_profit = self._calculate_take_profit(current_price, stop_loss, risk_reward_ratio=2.5)
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
                key_indicators=breakout_details
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
        Calculate technical indicators for breakout analysis.

        Args:
            klines (pd.DataFrame): Historical price data

        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df = klines.copy()

        # Calculate ATR (Average True Range)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)

        # Calculate all Moving Averages
        for period in self.ma_periods:
            df[f'ma_{period}'] = talib.SMA(df['close'], timeperiod=period)

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

        # Calculate Keltner Channels
        df['keltner_middle'] = talib.EMA(df['close'], timeperiod=self.keltner_period)
        df['keltner_upper'] = df['keltner_middle'] + df['atr'] * self.keltner_atr_multiple
        df['keltner_lower'] = df['keltner_middle'] - df['atr'] * self.keltner_atr_multiple

        # Calculate Donchian Channels
        df['donchian_high'] = df['high'].rolling(window=self.lookback_periods).max()
        df['donchian_low'] = df['low'].rolling(window=self.lookback_periods).min()
        df['donchian_middle'] = (df['donchian_high'] + df['donchian_low']) / 2

        # Calculate price range as percentage
        df['price_range'] = (df['high'] - df['low']) / df['low'] * 100

        # Calculate volume change
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Calculate historical volatility
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=20).std() * np.sqrt(365)

        return df

    def _identify_breakouts(self, klines):
        """
        Identify potential breakout patterns.

        Args:
            klines (pd.DataFrame): DataFrame with indicators

        Returns:
            tuple: (breakout_direction, breakout_strength, breakout_details)
        """
        # Get the latest values
        latest = klines.iloc[-1]

        # Initialize breakout details list
        breakout_details = []

        # Check for Bollinger Band squeeze and breakout
        bb_squeeze = False
        bb_breakout = 0

        # Look for Bollinger Band squeeze (narrowing bands)
        recent_bb_width = klines['bb_width'].iloc[-5:]
        if recent_bb_width.iloc[-1] < recent_bb_width.iloc[:-1].mean() * 0.8:
            bb_squeeze = True
            breakout_details.append("Bollinger Band squeeze detected - Potential breakout setup")

        # Check for Bollinger Band breakout
        if latest['close'] > latest['bb_upper']:
            bb_breakout = 1
            breakout_details.append("Price broke above upper Bollinger Band - Bullish breakout signal")
        elif latest['close'] < latest['bb_lower']:
            bb_breakout = -1
            breakout_details.append("Price broke below lower Bollinger Band - Bearish breakout signal")

        # Check for Donchian Channel breakout
        donchian_breakout = 0
        if latest['close'] > latest['donchian_high']:
            donchian_breakout = 1
            breakout_details.append(f"Price broke above {self.lookback_periods}-period high - Strong bullish breakout")
        elif latest['close'] < latest['donchian_low']:
            donchian_breakout = -1
            breakout_details.append(f"Price broke below {self.lookback_periods}-period low - Strong bearish breakout")

        # Check for consolidation breakout
        consolidation_breakout = 0

        # Check if price was in consolidation recently
        recent_ranges = klines['price_range'].iloc[-6:-1]  # Last 5 bars excluding current
        if recent_ranges.mean() < self.consolidation_threshold * 100:
            breakout_details.append(f"Recent price consolidation detected (avg range: {recent_ranges.mean():.2f}%)")

            # Check if current bar is breaking out of consolidation
            if latest['price_range'] > recent_ranges.mean() * 1.5:
                # Determine breakout direction
                if latest['close'] > klines['close'].iloc[-2]:
                    consolidation_breakout = 1
                    breakout_details.append("Bullish breakout from consolidation")
                else:
                    consolidation_breakout = -1
                    breakout_details.append("Bearish breakout from consolidation")

        # Check for volume confirmation if volume data is available
        volume_confirmation = 0
        if 'volume' in klines.columns and 'volume_ratio' in klines.columns:
            if latest['volume_ratio'] > self.volume_threshold:
                volume_confirmation = 1 if latest['close'] > klines['close'].iloc[-2] else -1
                breakout_details.append(f"Volume spike ({latest['volume_ratio']:.2f}x average) confirms breakout direction")

        # Check for volatility breakout
        volatility_breakout = 0
        if 'volatility' in klines.columns:
            recent_volatility = klines['volatility'].iloc[-6:-1].mean()
            current_volatility = latest['volatility']

            if current_volatility > recent_volatility * 1.5:
                volatility_breakout = 1 if latest['close'] > klines['close'].iloc[-2] else -1
                breakout_details.append(f"Volatility breakout detected (current: {current_volatility:.4f}, avg: {recent_volatility:.4f})")

        # Determine overall breakout direction and strength
        breakout_scores = [bb_breakout * 2 if bb_squeeze else bb_breakout,
                          donchian_breakout * 3,  # Stronger weight for Donchian breakout
                          consolidation_breakout * 2,
                          volume_confirmation,
                          volatility_breakout]

        # Filter out zero scores
        non_zero_scores = [s for s in breakout_scores if s != 0]

        # Calculate breakout direction
        if non_zero_scores:
            breakout_direction_score = sum(non_zero_scores)

            if breakout_direction_score > 0:
                breakout_direction = "UP"
            elif breakout_direction_score < 0:
                breakout_direction = "DOWN"
            else:
                breakout_direction = "NONE"

            # Calculate breakout strength (0-100)
            max_possible_score = sum([abs(s) for s in breakout_scores if s != 0])
            if max_possible_score > 0:
                breakout_strength = (abs(breakout_direction_score) / max_possible_score) * 100
            else:
                breakout_strength = 0
        else:
            breakout_direction = "NONE"
            breakout_strength = 0

        return breakout_direction, breakout_strength, breakout_details

    def _generate_recommendation(self, breakout_direction, breakout_strength, breakout_details, klines):
        """
        Generate a trading recommendation based on breakout analysis.

        Args:
            breakout_direction (str): 'UP', 'DOWN', or 'NONE'
            breakout_strength (float): Breakout strength (0-100)
            breakout_details (list): List of breakout pattern descriptions
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

        # Generate recommendation based on breakout direction and strength
        if breakout_direction == "UP" and breakout_strength > 60:
            recommendation = "BUY"
            confidence = breakout_strength
            reasoning = f"Strong bullish breakout detected with {breakout_strength:.2f}% confidence. "
        elif breakout_direction == "DOWN" and breakout_strength > 60:
            recommendation = "SELL"
            confidence = breakout_strength
            reasoning = f"Strong bearish breakout detected with {breakout_strength:.2f}% confidence. "
        else:
            recommendation = "HOLD"
            confidence = 100 - breakout_strength
            reasoning = f"No clear breakout detected or breakout strength ({breakout_strength:.2f}%) is below threshold. "

        # Add breakout details to reasoning
        reasoning += "Breakout analysis: "
        for detail in breakout_details:
            reasoning += detail + ". "

        # Check for false breakout warning signs and add trend context
        if recommendation == "BUY":
            # Check if price is extended from moving average
            if 'bb_middle' in klines.columns and latest['close'] > latest['bb_middle'] * 1.05:
                confidence = max(confidence - 10, 0)
                reasoning += "Warning: Price extended from moving average, potential false breakout. "

            # Check for divergence with RSI
            if 'rsi' in klines.columns:
                rsi = klines['rsi']
                if latest['close'] > klines['close'].iloc[-2] and latest['rsi'] < rsi.iloc[-2]:
                    confidence = max(confidence - 15, 0)
                    reasoning += "Warning: RSI divergence detected, reducing confidence. "

            # Add trend context using moving averages
            # Count how many MAs the price is above (trend alignment)
            mas_above = 0
            for period in self.ma_periods:
                if latest['close'] > latest[f'ma_{period}']:
                    mas_above += 1

            # Adjust confidence based on trend context
            if mas_above >= 4:  # Price above most MAs (strong uptrend)
                confidence = min(confidence + 15, 100)
                reasoning += f"Price above {mas_above}/5 MAs - Breakout aligned with strong uptrend. "
            elif mas_above <= 1:  # Price below most MAs (downtrend)
                confidence = max(confidence - 15, 0)
                reasoning += f"Warning: Price below most MAs, breakout against the trend. "

            # Check 200 MA for long-term trend context
            if latest['close'] > latest['ma_200']:
                reasoning += "Price above 200 MA - Aligned with long-term uptrend. "
            else:
                reasoning += "Price below 200 MA - Counter to long-term downtrend. "

        elif recommendation == "SELL":
            # Check if price is extended from moving average
            if 'bb_middle' in klines.columns and latest['close'] < latest['bb_middle'] * 0.95:
                confidence = max(confidence - 10, 0)
                reasoning += "Warning: Price extended from moving average, potential false breakout. "

            # Check for divergence with RSI
            if 'rsi' in klines.columns:
                rsi = klines['rsi']
                if latest['close'] < klines['close'].iloc[-2] and latest['rsi'] > rsi.iloc[-2]:
                    confidence = max(confidence - 15, 0)
                    reasoning += "Warning: RSI divergence detected, reducing confidence. "

            # Add trend context using moving averages
            # Count how many MAs the price is below (trend alignment)
            mas_below = 0
            for period in self.ma_periods:
                if latest['close'] < latest[f'ma_{period}']:
                    mas_below += 1

            # Adjust confidence based on trend context
            if mas_below >= 4:  # Price below most MAs (strong downtrend)
                confidence = min(confidence + 15, 100)
                reasoning += f"Price below {mas_below}/5 MAs - Breakout aligned with strong downtrend. "
            elif mas_below <= 1:  # Price above most MAs (uptrend)
                confidence = max(confidence - 15, 0)
                reasoning += f"Warning: Price above most MAs, breakout against the trend. "

            # Check 200 MA for long-term trend context
            if latest['close'] < latest['ma_200']:
                reasoning += "Price below 200 MA - Aligned with long-term downtrend. "
            else:
                reasoning += "Price above 200 MA - Counter to long-term uptrend. "

        return recommendation, confidence, reasoning

"""
Trend Following Strategy Implementation
This module implements a trend following strategy based on moving averages and other trend indicators.
"""
# Import required libraries
from datetime import datetime
import talib
import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    """
    A trend following strategy that uses moving averages, MACD, and ADX to identify trends.
    """

    def __init__(self, name="Trend Following", description="Uses moving averages and trend indicators to follow market trends"):
        """
        Initialize the trend following strategy.

        Args:
            name (str): Name of the strategy
            description (str): Description of the strategy
        """
        super().__init__(name, description)

        # Strategy parameters
        self.ma_periods = [9, 20, 50, 100, 200]  # Multiple moving average periods
        self.fast_ma_period = 9  # Fastest MA for crossover analysis
        self.medium_ma_period = 20  # Medium MA for crossover analysis
        self.slow_ma_period = 50  # Slow MA for crossover analysis
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.adx_period = 14
        self.adx_threshold = 25  # ADX above this indicates a strong trend

    def analyze(self, data):
        """
        Analyze the market data and generate signals based on trend following indicators.

        Args:
            data (dict): Dictionary containing market data including:
                - symbol (str): Trading symbol (e.g., 'BTC-USDT')
                - klines (pd.DataFrame): Historical price data
                - ticker (dict, optional): Current ticker data

        Returns:
            dict: Analysis results including signal, confidence, and supporting data
        """
        symbol_name = data.get('symbol')
        klines = data.get('klines')
        ticker = data.get('ticker')
        try:
            # Ensure we have enough data for the longest MA (200) plus some buffer
            if len(klines) < 220:  # 200 + 20 buffer
                return {
                    'source': self.name,
                    'symbol': symbol_name,
                    'recommendation': 'HOLD',
                    'confidence': 0,
                    'reasoning': f"Insufficient data for {self.name} strategy (need at least 220 periods)",
                    'timestamp': datetime.now().isoformat()
                }

            # Get current price
            current_price = float(ticker['last']) if ticker else klines['close'].iloc[-1]

            # Calculate indicators
            klines = self._calculate_indicators(klines)

            # Determine trend direction and strength
            trend_direction, trend_strength, trend_indicators = self._analyze_trend(klines)

            # Generate recommendation based on trend
            recommendation, confidence, reasoning = self._generate_recommendation(
                trend_direction, trend_strength, trend_indicators, klines
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
                symbol=symbol_name,
                recommendation=recommendation,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                reasoning=reasoning,
                key_indicators=trend_indicators
            )

        except Exception as e:
            # Return a HOLD signal with error information
            return {
                'source': self.name,
                'symbol': symbol_name,
                'recommendation': 'HOLD',
                'confidence': 0,
                'reasoning': f"Error in {self.name} strategy: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_indicators(self, klines):
        """
        Calculate technical indicators for trend analysis.

        Args:
            klines (pd.DataFrame): Historical price data

        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df = klines.copy()

        # Calculate all moving averages
        for period in self.ma_periods:
            df[f'ma_{period}'] = talib.SMA(df['close'], timeperiod=period)

        # Set specific MAs for crossover analysis
        df['fast_ma'] = df['ma_9']  # 9-period MA
        df['medium_ma'] = df['ma_20']  # 20-period MA
        df['slow_ma'] = df['ma_50']  # 50-period MA

        # Calculate MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'],
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # Calculate ADX (Average Directional Index)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=self.adx_period)

        # Calculate +DI and -DI (Directional Indicators)
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=self.adx_period)
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=self.adx_period)

        # Calculate RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)

        # Calculate Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )

        return df

    def _analyze_trend(self, klines):
        """
        Analyze the trend direction and strength.

        Args:
            klines (pd.DataFrame): DataFrame with indicators

        Returns:
            tuple: (trend_direction, trend_strength, trend_indicators)
        """
        # Get the latest values
        latest = klines.iloc[-1]

        # Initialize trend indicators list
        trend_indicators = []

        # Check moving average crossovers
        ma_trend = 0

        # Check fast/medium MA crossover (9/20)
        if latest['fast_ma'] > latest['medium_ma']:
            ma_trend += 1
            trend_indicators.append(f"9 MA above 20 MA - Bullish")
        elif latest['fast_ma'] < latest['medium_ma']:
            ma_trend -= 1
            trend_indicators.append(f"9 MA below 20 MA - Bearish")

        # Check medium/slow MA crossover (20/50)
        if latest['medium_ma'] > latest['slow_ma']:
            ma_trend += 1
            trend_indicators.append(f"20 MA above 50 MA - Bullish")
        elif latest['medium_ma'] < latest['slow_ma']:
            ma_trend -= 1
            trend_indicators.append(f"20 MA below 50 MA - Bearish")

        # Check 50/100 MA crossover
        if latest['ma_50'] > latest['ma_100']:
            ma_trend += 1
            trend_indicators.append(f"50 MA above 100 MA - Bullish")
        elif latest['ma_50'] < latest['ma_100']:
            ma_trend -= 1
            trend_indicators.append(f"50 MA below 100 MA - Bearish")

        # Check 100/200 MA crossover (longer-term trend)
        if latest['ma_100'] > latest['ma_200']:
            ma_trend += 1
            trend_indicators.append(f"100 MA above 200 MA - Bullish")
        elif latest['ma_100'] < latest['ma_200']:
            ma_trend -= 1
            trend_indicators.append(f"100 MA below 200 MA - Bearish")

        # Check price relative to moving averages
        price_ma_trend = 0

        # Count how many MAs the price is above/below
        mas_above = 0
        mas_below = 0

        for period in self.ma_periods:
            if latest['close'] > latest[f'ma_{period}']:
                mas_above += 1
            else:
                mas_below += 1

        # Determine trend based on price position relative to MAs
        if mas_above >= 4:  # Price above 4 or 5 MAs
            price_ma_trend = 2
            trend_indicators.append(f"Price above {mas_above}/5 MAs - Strong Bullish")
        elif mas_above >= 3:  # Price above 3 MAs
            price_ma_trend = 1
            trend_indicators.append(f"Price above {mas_above}/5 MAs - Bullish")
        elif mas_below >= 4:  # Price below 4 or 5 MAs
            price_ma_trend = -2
            trend_indicators.append(f"Price below {mas_below}/5 MAs - Strong Bearish")
        elif mas_below >= 3:  # Price below 3 MAs
            price_ma_trend = -1
            trend_indicators.append(f"Price below {mas_below}/5 MAs - Bearish")

        # Add specific MA analysis
        if latest['close'] > latest['ma_200']:
            trend_indicators.append("Price above 200 MA - Long-term bullish")
        else:
            trend_indicators.append("Price below 200 MA - Long-term bearish")

        # Check MACD
        macd_trend = 0
        if latest['macd'] > latest['macd_signal']:
            macd_trend = 1
            trend_indicators.append("MACD above Signal line - Bullish")
        elif latest['macd'] < latest['macd_signal']:
            macd_trend = -1
            trend_indicators.append("MACD below Signal line - Bearish")

        # Check ADX and directional indicators
        adx_trend = 0
        if latest['adx'] > self.adx_threshold:
            trend_indicators.append(f"ADX ({latest['adx']:.2f}) above threshold ({self.adx_threshold}) - Strong trend")
            if latest['plus_di'] > latest['minus_di']:
                adx_trend = 1
                trend_indicators.append("+DI above -DI - Bullish trend")
            else:
                adx_trend = -1
                trend_indicators.append("-DI above +DI - Bearish trend")
        else:
            trend_indicators.append(f"ADX ({latest['adx']:.2f}) below threshold ({self.adx_threshold}) - Weak trend")

        # Check RSI
        if latest['rsi'] > 70:
            trend_indicators.append(f"RSI ({latest['rsi']:.2f}) above 70 - Overbought")
        elif latest['rsi'] < 30:
            trend_indicators.append(f"RSI ({latest['rsi']:.2f}) below 30 - Oversold")

        # Determine overall trend direction
        # Note: ma_trend now ranges from -4 to 4 (4 crossover checks)
        # price_ma_trend now ranges from -2 to 2
        # Normalize ma_trend to have similar weight as other indicators
        normalized_ma_trend = ma_trend / 4 if ma_trend != 0 else 0

        trend_scores = [normalized_ma_trend, price_ma_trend / 2, macd_trend, adx_trend]
        trend_direction = sum(trend_scores)

        # Determine trend strength (0-100)
        max_strength = len(trend_scores)  # Maximum possible score
        trend_strength = (abs(trend_direction) / max_strength) * 100

        # Adjust trend strength based on ADX
        if latest['adx'] > self.adx_threshold:
            trend_strength = min(trend_strength * (latest['adx'] / self.adx_threshold), 100)

        # Determine final trend direction
        if trend_direction > 0:
            trend_direction = "UP"
        elif trend_direction < 0:
            trend_direction = "DOWN"
        else:
            trend_direction = "SIDEWAYS"

        return trend_direction, trend_strength, trend_indicators

    def _generate_recommendation(self, trend_direction, trend_strength, trend_indicators, klines):
        """
        Generate a trading recommendation based on trend analysis.

        Args:
            trend_direction (str): 'UP', 'DOWN', or 'SIDEWAYS'
            trend_strength (float): Trend strength (0-100)
            trend_indicators (list): List of trend indicator descriptions
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

        # Generate recommendation based on trend
        if trend_direction == "UP" and trend_strength > 50:
            recommendation = "BUY"
            confidence = trend_strength
            reasoning = f"Strong uptrend detected with {trend_strength:.2f}% confidence. "
        elif trend_direction == "DOWN" and trend_strength > 50:
            recommendation = "SELL"
            confidence = trend_strength
            reasoning = f"Strong downtrend detected with {trend_strength:.2f}% confidence. "
        else:
            recommendation = "HOLD"
            confidence = 100 - trend_strength
            reasoning = f"No clear trend detected or trend strength ({trend_strength:.2f}%) is below threshold. "

        # Add indicator details to reasoning
        reasoning += "Indicator analysis: "
        reasoning += ". ".join(trend_indicators) + "."

        # Check for potential reversal signals
        if recommendation == "BUY" and latest['rsi'] > 70:
            confidence = max(confidence - 20, 0)
            reasoning += "Warning: RSI indicates overbought conditions, reducing confidence. "
        elif recommendation == "SELL" and latest['rsi'] < 30:
            confidence = max(confidence - 20, 0)
            reasoning += "Warning: RSI indicates oversold conditions, reducing confidence. "

        # Check Bollinger Bands for volatility
        if latest['close'] > latest['bb_upper']:
            if recommendation == "BUY":
                confidence = max(confidence - 15, 0)
                reasoning += "Warning: Price above upper Bollinger Band, potential resistance. "
            elif recommendation == "SELL":
                confidence = min(confidence + 10, 100)
                reasoning += "Price above upper Bollinger Band, confirming bearish signal. "
        elif latest['close'] < latest['bb_lower']:
            if recommendation == "SELL":
                confidence = max(confidence - 15, 0)
                reasoning += "Warning: Price below lower Bollinger Band, potential support. "
            elif recommendation == "BUY":
                confidence = min(confidence + 10, 100)
                reasoning += "Price below lower Bollinger Band, confirming bullish signal. "

        return recommendation, confidence, reasoning

    def _calculate_support_resistance(self, klines):
        """
        Calculate support and resistance levels.

        Args:
            klines (pd.DataFrame): DataFrame with indicators

        Returns:
            tuple: (support_levels, resistance_levels)
        """
        # Get the latest close price
        latest_close = klines['close'].iloc[-1]

        # We'll use a simpler approach to find support and resistance levels

        # Find recent pivot points (last 20 candles)
        recent_data = klines.iloc[-30:]

        # Identify support levels (recent lows)
        support_levels = []
        for i in range(2, len(recent_data) - 2):
            if (recent_data['low'].iloc[i] <= recent_data['low'].iloc[i-1] and
                recent_data['low'].iloc[i] <= recent_data['low'].iloc[i-2] and
                recent_data['low'].iloc[i] <= recent_data['low'].iloc[i+1] and
                recent_data['low'].iloc[i] <= recent_data['low'].iloc[i+2]):
                support_levels.append(recent_data['low'].iloc[i])

        # Identify resistance levels (recent highs)
        resistance_levels = []
        for i in range(2, len(recent_data) - 2):
            if (recent_data['high'].iloc[i] >= recent_data['high'].iloc[i-1] and
                recent_data['high'].iloc[i] >= recent_data['high'].iloc[i-2] and
                recent_data['high'].iloc[i] >= recent_data['high'].iloc[i+1] and
                recent_data['high'].iloc[i] >= recent_data['high'].iloc[i+2]):
                resistance_levels.append(recent_data['high'].iloc[i])

        # Filter support levels below current price
        support_levels = [level for level in support_levels if level < latest_close]

        # Filter resistance levels above current price
        resistance_levels = [level for level in resistance_levels if level > latest_close]

        # Sort levels
        support_levels.sort(reverse=True)  # Highest support first
        resistance_levels.sort()  # Lowest resistance first

        # Limit to top 3 levels
        support_levels = support_levels[:3]
        resistance_levels = resistance_levels[:3]

        return support_levels, resistance_levels

    def _calculate_stop_loss(self, klines, position_type):
        """
        Calculate stop loss level based on recent volatility.

        Args:
            klines (pd.DataFrame): DataFrame with indicators
            position_type (str): 'LONG' or 'SHORT'

        Returns:
            float: Stop loss price
        """
        # Get the latest values
        latest_close = klines['close'].iloc[-1]

        # Calculate ATR (Average True Range)
        atr = talib.ATR(klines['high'], klines['low'], klines['close'], timeperiod=14).iloc[-1]

        # Calculate stop loss based on position type
        if position_type == 'LONG':
            # For long positions, stop loss is below current price
            stop_loss = latest_close - (atr * 2)

            # Check if there's a support level that can be used
            support_levels, _ = self._calculate_support_resistance(klines)
            if support_levels and support_levels[0] > stop_loss:
                # Use the highest support level as stop loss if it's better than ATR-based stop
                stop_loss = support_levels[0]
        else:
            # For short positions, stop loss is above current price
            stop_loss = latest_close + (atr * 2)

            # Check if there's a resistance level that can be used
            _, resistance_levels = self._calculate_support_resistance(klines)
            if resistance_levels and resistance_levels[0] < stop_loss:
                # Use the lowest resistance level as stop loss if it's better than ATR-based stop
                stop_loss = resistance_levels[0]

        return stop_loss

    def _calculate_take_profit(self, current_price, stop_loss):
        """
        Calculate take profit level based on risk-reward ratio.

        Args:
            current_price (float): Current price
            stop_loss (float): Stop loss price

        Returns:
            float: Take profit price
        """
        # Calculate risk (distance to stop loss)
        risk = abs(current_price - stop_loss)

        # Calculate take profit based on risk-reward ratio (default 2:1)
        risk_reward_ratio = 2.0

        # Calculate take profit based on position type (long if current_price > stop_loss)
        return current_price + (risk * risk_reward_ratio) if current_price > stop_loss else current_price - (risk * risk_reward_ratio)

    def get_required_indicators(self):
        """
        Get the list of indicators required by this strategy

        Returns:
            list: List of indicator names
        """
        return [
            'SMA',
            'MACD',
            'ADX',
            'PLUS_DI',
            'MINUS_DI',
            'RSI',
            'BBANDS',
            'ATR'
        ]

    def _format_signal(self, symbol, recommendation, confidence, entry_price, stop_loss, take_profit,
                      support_levels, resistance_levels, reasoning, key_indicators):
        """
        Format the trading signal.

        Args:
            symbol (str): Trading symbol
            recommendation (str): Trading recommendation (BUY, SELL, HOLD)
            confidence (float): Confidence level (0-100)
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            take_profit (float): Take profit price
            support_levels (list): Support levels
            resistance_levels (list): Resistance levels
            reasoning (str): Reasoning for the signal
            key_indicators (list): Key indicators used in the analysis

        Returns:
            dict: Formatted signal
        """
        return {
            'source': self.name,
            'symbol': symbol,
            'recommendation': recommendation,
            'confidence': confidence,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'key_indicators': key_indicators,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        }

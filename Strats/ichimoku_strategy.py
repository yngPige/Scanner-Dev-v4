"""
Ichimoku Cloud Strategy Implementation
This module implements a trading strategy based on the Ichimoku Cloud indicator.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import talib
from src.trading.strategies import TradingStrategy

class IchimokuStrategy(TradingStrategy):
    """
    A strategy that uses the Ichimoku Cloud indicator to identify trends, support/resistance,
    and potential entry/exit points.
    """

    def __init__(self, name="Ichimoku Cloud", description="Uses Ichimoku Cloud for trend analysis and support/resistance"):
        """
        Initialize the Ichimoku Cloud strategy.

        Args:
            name (str): Name of the strategy
            description (str): Description of the strategy
        """
        super().__init__(name, description)

        # Strategy parameters (traditional values)
        self.tenkan_period = 9  # Conversion Line (Tenkan-sen)
        self.kijun_period = 26  # Base Line (Kijun-sen)
        self.senkou_span_b_period = 52  # Leading Span B (Senkou Span B)
        self.displacement = 26  # Displacement (Chikou Span)

    def generate_signal(self, symbol, klines, ticker=None):
        """
        Generate a trading signal based on Ichimoku Cloud analysis.

        Args:
            symbol (str): Trading symbol (e.g., 'BTC-USDT')
            klines (pd.DataFrame): Historical price data
            ticker (dict, optional): Current ticker data

        Returns:
            dict: Signal with recommendation, confidence, etc.
        """
        try:
            # Ensure we have enough data
            min_periods = max(self.tenkan_period, self.kijun_period, self.senkou_span_b_period) + self.displacement + 10
            if len(klines) < min_periods:
                return {
                    'source': self.name,
                    'recommendation': 'HOLD',
                    'confidence': 0,
                    'reasoning': f"Insufficient data for {self.name} strategy",
                    'timestamp': datetime.now().isoformat()
                }

            # Extract symbol name without exchange suffix
            symbol_name = symbol.split(':')[-1] if ':' in symbol else symbol

            # Get current price
            current_price = float(ticker['last']) if ticker else klines['close'].iloc[-1]

            # Calculate Ichimoku Cloud components
            ichimoku = self._calculate_ichimoku(klines)

            # Analyze Ichimoku Cloud and generate signals
            recommendation, confidence, reasoning, ichimoku_indicators = self._analyze_ichimoku(
                current_price, ichimoku, klines
            )

            # Calculate support and resistance levels
            support_levels, resistance_levels = self._identify_support_resistance(ichimoku, current_price)

            # Calculate stop loss and take profit levels
            if recommendation == 'BUY':
                # For buy signals, use the Kijun-sen (Base Line) as stop loss
                stop_loss = min(ichimoku['kijun'].iloc[-1], support_levels[0] if support_levels else current_price * 0.95)
            elif recommendation == 'SELL':
                # For sell signals, use the Kijun-sen (Base Line) as stop loss
                stop_loss = max(ichimoku['kijun'].iloc[-1], resistance_levels[0] if resistance_levels else current_price * 1.05)
            else:
                # For hold signals, use a default stop loss
                stop_loss = current_price * 0.95 if current_price > ichimoku['kijun'].iloc[-1] else current_price * 1.05

            # Calculate take profit based on risk-reward ratio
            take_profit = self._calculate_take_profit(current_price, stop_loss, risk_reward_ratio=2.0)

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
                key_indicators=ichimoku_indicators
            )

        except Exception as e:
            # Return a HOLD signal with error information
            return {
                'source': self.name,
                'symbol': symbol_name if 'symbol_name' in locals() else symbol,
                'recommendation': 'HOLD',
                'confidence': 0,
                'reasoning': f"Error in {self.name} strategy: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_ichimoku(self, klines):
        """
        Calculate Ichimoku Cloud components.

        Args:
            klines (pd.DataFrame): Historical price data

        Returns:
            pd.DataFrame: DataFrame with Ichimoku components
        """
        # Extract price data
        high = klines['high']
        low = klines['low']
        close = klines['close']

        # Calculate Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=self.tenkan_period).max()
        tenkan_low = low.rolling(window=self.tenkan_period).min()
        tenkan = (tenkan_high + tenkan_low) / 2

        # Calculate Kijun-sen (Base Line)
        kijun_high = high.rolling(window=self.kijun_period).max()
        kijun_low = low.rolling(window=self.kijun_period).min()
        kijun = (kijun_high + kijun_low) / 2

        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan + kijun) / 2).shift(self.displacement)

        # Calculate Senkou Span B (Leading Span B)
        senkou_b_high = high.rolling(window=self.senkou_span_b_period).max()
        senkou_b_low = low.rolling(window=self.senkou_span_b_period).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(self.displacement)

        # Calculate Chikou Span (Lagging Span)
        chikou_span = close.shift(-self.displacement)

        # Combine into a DataFrame
        ichimoku = pd.DataFrame({
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_span_a,
            'senkou_b': senkou_span_b,
            'chikou': chikou_span
        })

        return ichimoku

    def _analyze_ichimoku(self, current_price, ichimoku, klines):
        """
        Analyze Ichimoku Cloud and generate trading signals.

        Args:
            current_price (float): Current price
            ichimoku (pd.DataFrame): Ichimoku Cloud data
            klines (pd.DataFrame): Historical price data

        Returns:
            tuple: (recommendation, confidence, reasoning, ichimoku_indicators)
        """
        # Initialize variables
        ichimoku_indicators = []
        
        # Get the latest values
        latest = ichimoku.iloc[-1]
        tenkan = latest['tenkan']
        kijun = latest['kijun']
        senkou_a = latest['senkou_a']
        senkou_b = latest['senkou_b']
        
        # We can't use the latest Chikou Span as it's in the future
        # Instead, use the value from displacement periods ago
        chikou = ichimoku['chikou'].iloc[-self.displacement-1] if len(ichimoku) > self.displacement else None
        
        # Add indicators to the list
        ichimoku_indicators.append(f"Tenkan-sen (Conversion Line): {tenkan:.2f}")
        ichimoku_indicators.append(f"Kijun-sen (Base Line): {kijun:.2f}")
        ichimoku_indicators.append(f"Senkou Span A (Leading Span A): {senkou_a:.2f}")
        ichimoku_indicators.append(f"Senkou Span B (Leading Span B): {senkou_b:.2f}")
        if chikou is not None:
            price_at_chikou = klines['close'].iloc[-self.displacement-1]
            ichimoku_indicators.append(f"Chikou Span (Lagging Span): {chikou:.2f} vs Price: {price_at_chikou:.2f}")
        
        # Determine cloud color (bullish or bearish)
        cloud_bullish = senkou_a > senkou_b
        if cloud_bullish:
            ichimoku_indicators.append("Cloud is bullish (green)")
        else:
            ichimoku_indicators.append("Cloud is bearish (red)")
        
        # Check price position relative to the cloud
        above_cloud = current_price > max(senkou_a, senkou_b)
        below_cloud = current_price < min(senkou_a, senkou_b)
        in_cloud = not above_cloud and not below_cloud
        
        if above_cloud:
            ichimoku_indicators.append("Price is above the cloud (bullish)")
        elif below_cloud:
            ichimoku_indicators.append("Price is below the cloud (bearish)")
        else:
            ichimoku_indicators.append("Price is inside the cloud (neutral)")
        
        # Check Tenkan/Kijun cross (TK cross)
        tk_cross_bullish = tenkan > kijun and ichimoku['tenkan'].iloc[-2] <= ichimoku['kijun'].iloc[-2]
        tk_cross_bearish = tenkan < kijun and ichimoku['tenkan'].iloc[-2] >= ichimoku['kijun'].iloc[-2]
        
        if tk_cross_bullish:
            ichimoku_indicators.append("Bullish TK Cross (Tenkan crossed above Kijun)")
        elif tk_cross_bearish:
            ichimoku_indicators.append("Bearish TK Cross (Tenkan crossed below Kijun)")
        
        # Check Chikou Span position
        chikou_bullish = chikou is not None and chikou > klines['close'].iloc[-self.displacement-1]
        chikou_bearish = chikou is not None and chikou < klines['close'].iloc[-self.displacement-1]
        
        if chikou_bullish:
            ichimoku_indicators.append("Chikou Span is above price (bullish)")
        elif chikou_bearish:
            ichimoku_indicators.append("Chikou Span is below price (bearish)")
        
        # Generate recommendation based on Ichimoku analysis
        # Strong buy signal: Price above cloud, bullish TK cross, Chikou above price
        if above_cloud and (tk_cross_bullish or tenkan > kijun) and chikou_bullish:
            recommendation = "BUY"
            confidence = 80
            reasoning = "Strong bullish signal: Price above cloud, Tenkan above Kijun, Chikou above price."
            
        # Strong sell signal: Price below cloud, bearish TK cross, Chikou below price
        elif below_cloud and (tk_cross_bearish or tenkan < kijun) and chikou_bearish:
            recommendation = "SELL"
            confidence = 80
            reasoning = "Strong bearish signal: Price below cloud, Tenkan below Kijun, Chikou below price."
            
        # Moderate buy signal: Price above cloud, other indicators mixed
        elif above_cloud and (tenkan > kijun or chikou_bullish):
            recommendation = "BUY"
            confidence = 65
            reasoning = "Moderate bullish signal: Price above cloud with some bullish indicators."
            
        # Moderate sell signal: Price below cloud, other indicators mixed
        elif below_cloud and (tenkan < kijun or chikou_bearish):
            recommendation = "SELL"
            confidence = 65
            reasoning = "Moderate bearish signal: Price below cloud with some bearish indicators."
            
        # Weak buy signal: Price just crossed above cloud or bullish TK cross
        elif (current_price > senkou_a > senkou_b and klines['close'].iloc[-2] <= senkou_a) or tk_cross_bullish:
            recommendation = "BUY"
            confidence = 55
            reasoning = "Weak bullish signal: Price crossing above cloud or bullish TK cross."
            
        # Weak sell signal: Price just crossed below cloud or bearish TK cross
        elif (current_price < senkou_b < senkou_a and klines['close'].iloc[-2] >= senkou_b) or tk_cross_bearish:
            recommendation = "SELL"
            confidence = 55
            reasoning = "Weak bearish signal: Price crossing below cloud or bearish TK cross."
            
        # No clear signal
        else:
            recommendation = "HOLD"
            confidence = 50
            reasoning = "No clear Ichimoku signal."
        
        # Adjust confidence based on cloud color and price momentum
        if recommendation == "BUY" and cloud_bullish:
            confidence = min(confidence + 10, 100)
            reasoning += " Bullish cloud confirms the signal."
        elif recommendation == "SELL" and not cloud_bullish:
            confidence = min(confidence + 10, 100)
            reasoning += " Bearish cloud confirms the signal."
        elif recommendation == "BUY" and not cloud_bullish:
            confidence = max(confidence - 10, 0)
            reasoning += " Bearish cloud contradicts the signal."
        elif recommendation == "SELL" and cloud_bullish:
            confidence = max(confidence - 10, 0)
            reasoning += " Bullish cloud contradicts the signal."
        
        # Check price momentum
        price_momentum = klines['close'].pct_change(5).iloc[-1] * 100  # 5-period momentum
        
        if recommendation == "BUY" and price_momentum > 1:
            confidence = min(confidence + 5, 100)
            reasoning += f" Strong upward momentum ({price_momentum:.2f}%)."
        elif recommendation == "SELL" and price_momentum < -1:
            confidence = min(confidence + 5, 100)
            reasoning += f" Strong downward momentum ({price_momentum:.2f}%)."
        
        return recommendation, confidence, reasoning, ichimoku_indicators

    def _identify_support_resistance(self, ichimoku, current_price):
        """
        Identify support and resistance levels from Ichimoku Cloud.

        Args:
            ichimoku (pd.DataFrame): Ichimoku Cloud data
            current_price (float): Current price

        Returns:
            tuple: (support_levels, resistance_levels)
        """
        # Get the latest values
        latest = ichimoku.iloc[-1]
        tenkan = latest['tenkan']
        kijun = latest['kijun']
        senkou_a = latest['senkou_a']
        senkou_b = latest['senkou_b']
        
        # Initialize support and resistance levels
        support_levels = []
        resistance_levels = []
        
        # Add Ichimoku components as support/resistance levels
        if tenkan < current_price:
            support_levels.append(tenkan)
        else:
            resistance_levels.append(tenkan)
            
        if kijun < current_price:
            support_levels.append(kijun)
        else:
            resistance_levels.append(kijun)
            
        # Add cloud boundaries as support/resistance
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        if cloud_top < current_price:
            support_levels.append(cloud_top)
        else:
            resistance_levels.append(cloud_top)
            
        if cloud_bottom < current_price:
            support_levels.append(cloud_bottom)
        else:
            resistance_levels.append(cloud_bottom)
        
        # Sort levels
        support_levels = sorted(support_levels, reverse=True)
        resistance_levels = sorted(resistance_levels)
        
        return support_levels, resistance_levels

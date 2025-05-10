"""
Harmonic Pattern Strategy Implementation
This module implements a trading strategy based on harmonic price patterns.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from ta import ta
from src.trading.strategies import TradingStrategy

class HarmonicPatternStrategy(TradingStrategy):
    """
    A strategy that identifies harmonic price patterns for potential reversal points.
    Harmonic patterns are geometric price patterns that use Fibonacci ratios to define
    precise turning points.
    """

    def __init__(self, name="Harmonic Patterns", description="Identifies harmonic price patterns for potential reversals"):
        """
        Initialize the harmonic pattern strategy.

        Args:
            name (str): Name of the strategy
            description (str): Description of the strategy
        """
        super().__init__(name, description)

        # Strategy parameters
        self.lookback_periods = 100  # Number of periods to look back for pattern detection
        self.fibonacci_tolerance = 0.1  # Tolerance for Fibonacci ratio matching
        self.pattern_completion_tolerance = 0.03  # Tolerance for pattern completion point

    def generate_signal(self, symbol, klines, ticker=None):
        """
        Generate a trading signal based on harmonic pattern analysis.

        Args:
            symbol (str): Trading symbol (e.g., 'BTC-USDT')
            klines (pd.DataFrame): Historical price data
            ticker (dict, optional): Current ticker data

        Returns:
            dict: Signal with recommendation, confidence, etc.
        """
        try:
            # Ensure we have enough data
            if len(klines) < self.lookback_periods + 10:
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

            # Identify swing points (pivots)
            pivots = self._identify_pivots(klines)

            # Detect harmonic patterns
            patterns = self._detect_harmonic_patterns(pivots, klines, current_price)

            # Analyze patterns and generate signals
            recommendation, confidence, reasoning, pattern_indicators = self._analyze_patterns(
                patterns, current_price, klines
            )

            # Calculate support and resistance levels
            support_levels, resistance_levels = self._identify_support_resistance(patterns, pivots, current_price)

            # Calculate stop loss and take profit levels
            if recommendation == 'BUY':
                # For buy signals, use the most recent swing low as stop loss
                recent_lows = [p['price'] for p in pivots if p['type'] == 'low' and p['price'] < current_price]
                stop_loss = max(recent_lows) if recent_lows else current_price * 0.95
            elif recommendation == 'SELL':
                # For sell signals, use the most recent swing high as stop loss
                recent_highs = [p['price'] for p in pivots if p['type'] == 'high' and p['price'] > current_price]
                stop_loss = min(recent_highs) if recent_highs else current_price * 1.05
            else:
                # For hold signals, use a default stop loss
                stop_loss = current_price * 0.95 if current_price > klines['close'].mean() else current_price * 1.05

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
                key_indicators=pattern_indicators
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

    def _identify_pivots(self, klines):
        """
        Identify swing highs and lows (pivots) in the price data.

        Args:
            klines (pd.DataFrame): Historical price data

        Returns:
            list: List of pivot points with price, index, and type
        """
        # Use recent data
        recent_data = klines.iloc[-self.lookback_periods:]

        # Initialize pivot points list
        pivots = []

        # Window size for pivot detection
        window = 5

        # Detect swing highs
        for i in range(window, len(recent_data) - window):
            if all(recent_data['high'].iloc[i] > recent_data['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(recent_data['high'].iloc[i] > recent_data['high'].iloc[i+j] for j in range(1, window+1)):
                pivots.append({
                    'index': i + len(klines) - len(recent_data),  # Adjust index to original data
                    'price': recent_data['high'].iloc[i],
                    'type': 'high',
                    'date': recent_data.index[i] if hasattr(recent_data, 'index') else i
                })

        # Detect swing lows
        for i in range(window, len(recent_data) - window):
            if all(recent_data['low'].iloc[i] < recent_data['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(recent_data['low'].iloc[i] < recent_data['low'].iloc[i+j] for j in range(1, window+1)):
                pivots.append({
                    'index': i + len(klines) - len(recent_data),  # Adjust index to original data
                    'price': recent_data['low'].iloc[i],
                    'type': 'low',
                    'date': recent_data.index[i] if hasattr(recent_data, 'index') else i
                })

        # Sort pivots by index
        pivots.sort(key=lambda x: x['index'])

        return pivots

    def _detect_harmonic_patterns(self, pivots, klines, current_price):
        """
        Detect harmonic patterns in the price data.

        Args:
            pivots (list): List of pivot points
            klines (pd.DataFrame): Historical price data
            current_price (float): Current price

        Returns:
            list: List of detected patterns
        """
        # Initialize patterns list
        patterns = []

        # Need at least 5 pivot points to detect patterns
        if len(pivots) < 5:
            return patterns

        # Check each sequence of 5 pivot points
        for i in range(len(pivots) - 4):
            # Get 5 consecutive pivot points (X, A, B, C, D)
            points = pivots[i:i+5]

            # Ensure alternating high/low pattern
            if not all(points[j]['type'] != points[j+1]['type'] for j in range(4)):
                continue

            # Extract prices
            X, A, B, C, D = [p['price'] for p in points]

            # Calculate price movements
            XA = A - X
            AB = B - A
            BC = C - B
            CD = D - C

            # Calculate Fibonacci ratios
            if XA != 0 and AB != 0 and BC != 0:
                AB_XA_ratio = abs(AB / XA)
                BC_AB_ratio = abs(BC / AB)
                CD_BC_ratio = abs(CD / BC)

                # Check for Gartley pattern
                if self._is_gartley_pattern(AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, points):
                    pattern_type = "Bullish Gartley" if points[0]['type'] == 'low' else "Bearish Gartley"
                    patterns.append({
                        'type': pattern_type,
                        'points': points,
                        'completion_price': D,
                        'confidence': self._calculate_pattern_confidence(
                            pattern_type, AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, current_price, D
                        )
                    })

                # Check for Butterfly pattern
                if self._is_butterfly_pattern(AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, points):
                    pattern_type = "Bullish Butterfly" if points[0]['type'] == 'low' else "Bearish Butterfly"
                    patterns.append({
                        'type': pattern_type,
                        'points': points,
                        'completion_price': D,
                        'confidence': self._calculate_pattern_confidence(
                            pattern_type, AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, current_price, D
                        )
                    })

                # Check for Bat pattern
                if self._is_bat_pattern(AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, points):
                    pattern_type = "Bullish Bat" if points[0]['type'] == 'low' else "Bearish Bat"
                    patterns.append({
                        'type': pattern_type,
                        'points': points,
                        'completion_price': D,
                        'confidence': self._calculate_pattern_confidence(
                            pattern_type, AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, current_price, D
                        )
                    })

                # Check for Crab pattern
                if self._is_crab_pattern(AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, points):
                    pattern_type = "Bullish Crab" if points[0]['type'] == 'low' else "Bearish Crab"
                    patterns.append({
                        'type': pattern_type,
                        'points': points,
                        'completion_price': D,
                        'confidence': self._calculate_pattern_confidence(
                            pattern_type, AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, current_price, D
                        )
                    })

        # Sort patterns by confidence
        patterns.sort(key=lambda x: x['confidence'], reverse=True)

        return patterns

    def _is_gartley_pattern(self, AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, points):
        """
        Check if the given ratios form a Gartley pattern.

        Args:
            AB_XA_ratio (float): AB to XA ratio
            BC_AB_ratio (float): BC to AB ratio
            CD_BC_ratio (float): CD to BC ratio
            points (list): List of pivot points

        Returns:
            bool: True if pattern matches Gartley, False otherwise
        """
        # Gartley pattern ratios
        # AB = 0.618 of XA
        # BC = 0.382 or 0.886 of AB
        # CD = 1.272 or 1.618 of BC

        tolerance = self.fibonacci_tolerance

        AB_XA_valid = abs(AB_XA_ratio - 0.618) <= tolerance
        BC_AB_valid = abs(BC_AB_ratio - 0.382) <= tolerance or abs(BC_AB_ratio - 0.886) <= tolerance
        CD_BC_valid = abs(CD_BC_ratio - 1.272) <= tolerance or abs(CD_BC_ratio - 1.618) <= tolerance

        # Check if D point is near the 0.786 retracement of XA
        X, A = points[0]['price'], points[1]['price']
        D = points[4]['price']
        XA = A - X
        D_target = X + 0.786 * XA
        D_valid = abs(D - D_target) / abs(XA) <= self.pattern_completion_tolerance

        return AB_XA_valid and BC_AB_valid and CD_BC_valid and D_valid

    def _is_butterfly_pattern(self, AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, points):
        """
        Check if the given ratios form a Butterfly pattern.

        Args:
            AB_XA_ratio (float): AB to XA ratio
            BC_AB_ratio (float): BC to AB ratio
            CD_BC_ratio (float): CD to BC ratio
            points (list): List of pivot points

        Returns:
            bool: True if pattern matches Butterfly, False otherwise
        """
        # Butterfly pattern ratios
        # AB = 0.786 of XA
        # BC = 0.382 or 0.886 of AB
        # CD = 1.618 or 2.618 of BC

        tolerance = self.fibonacci_tolerance

        AB_XA_valid = abs(AB_XA_ratio - 0.786) <= tolerance
        BC_AB_valid = abs(BC_AB_ratio - 0.382) <= tolerance or abs(BC_AB_ratio - 0.886) <= tolerance
        CD_BC_valid = abs(CD_BC_ratio - 1.618) <= tolerance or abs(CD_BC_ratio - 2.618) <= tolerance

        # Check if D point is near the 1.27 or 1.618 extension of XA
        X, A = points[0]['price'], points[1]['price']
        D = points[4]['price']
        XA = A - X
        D_target1 = X - 1.27 * XA if XA > 0 else X - 1.27 * XA
        D_target2 = X - 1.618 * XA if XA > 0 else X - 1.618 * XA
        D_valid = abs(D - D_target1) / abs(XA) <= self.pattern_completion_tolerance or \
                  abs(D - D_target2) / abs(XA) <= self.pattern_completion_tolerance

        return AB_XA_valid and BC_AB_valid and CD_BC_valid and D_valid

    def _is_bat_pattern(self, AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, points):
        """
        Check if the given ratios form a Bat pattern.

        Args:
            AB_XA_ratio (float): AB to XA ratio
            BC_AB_ratio (float): BC to AB ratio
            CD_BC_ratio (float): CD to BC ratio
            points (list): List of pivot points

        Returns:
            bool: True if pattern matches Bat, False otherwise
        """
        # Bat pattern ratios
        # AB = 0.382 or 0.5 of XA
        # BC = 0.382 or 0.886 of AB
        # CD = 1.618 or 2.618 of BC

        tolerance = self.fibonacci_tolerance

        AB_XA_valid = abs(AB_XA_ratio - 0.382) <= tolerance or abs(AB_XA_ratio - 0.5) <= tolerance
        BC_AB_valid = abs(BC_AB_ratio - 0.382) <= tolerance or abs(BC_AB_ratio - 0.886) <= tolerance
        CD_BC_valid = abs(CD_BC_ratio - 1.618) <= tolerance or abs(CD_BC_ratio - 2.618) <= tolerance

        # Check if D point is near the 0.886 retracement of XA
        X, A = points[0]['price'], points[1]['price']
        D = points[4]['price']
        XA = A - X
        D_target = X + 0.886 * XA
        D_valid = abs(D - D_target) / abs(XA) <= self.pattern_completion_tolerance

        return AB_XA_valid and BC_AB_valid and CD_BC_valid and D_valid

    def _is_crab_pattern(self, AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, points):
        """
        Check if the given ratios form a Crab pattern.

        Args:
            AB_XA_ratio (float): AB to XA ratio
            BC_AB_ratio (float): BC to AB ratio
            CD_BC_ratio (float): CD to BC ratio
            points (list): List of pivot points

        Returns:
            bool: True if pattern matches Crab, False otherwise
        """
        # Crab pattern ratios
        # AB = 0.382 or 0.618 of XA
        # BC = 0.382 or 0.886 of AB
        # CD = 2.618 or 3.618 of BC

        tolerance = self.fibonacci_tolerance

        AB_XA_valid = abs(AB_XA_ratio - 0.382) <= tolerance or abs(AB_XA_ratio - 0.618) <= tolerance
        BC_AB_valid = abs(BC_AB_ratio - 0.382) <= tolerance or abs(BC_AB_ratio - 0.886) <= tolerance
        CD_BC_valid = abs(CD_BC_ratio - 2.618) <= tolerance or abs(CD_BC_ratio - 3.618) <= tolerance

        # Check if D point is near the 1.618 extension of XA
        X, A = points[0]['price'], points[1]['price']
        D = points[4]['price']
        XA = A - X
        D_target = X - 1.618 * XA if XA > 0 else X - 1.618 * XA
        D_valid = abs(D - D_target) / abs(XA) <= self.pattern_completion_tolerance

        return AB_XA_valid and BC_AB_valid and CD_BC_valid and D_valid

    def _calculate_pattern_confidence(self, pattern_type, AB_XA_ratio, BC_AB_ratio, CD_BC_ratio, current_price, completion_price):
        """
        Calculate confidence level for a detected pattern.

        Args:
            pattern_type (str): Type of harmonic pattern
            AB_XA_ratio (float): AB to XA ratio
            BC_AB_ratio (float): BC to AB ratio
            CD_BC_ratio (float): CD to BC ratio
            current_price (float): Current price
            completion_price (float): Pattern completion price (D point)

        Returns:
            float: Confidence level (0-100)
        """
        # Base confidence
        confidence = 70

        # Adjust confidence based on ratio precision
        if "Gartley" in pattern_type:
            confidence += (1 - abs(AB_XA_ratio - 0.618) / 0.618) * 10
        elif "Butterfly" in pattern_type:
            confidence += (1 - abs(AB_XA_ratio - 0.786) / 0.786) * 10
        elif "Bat" in pattern_type:
            confidence += (1 - min(abs(AB_XA_ratio - 0.382), abs(AB_XA_ratio - 0.5)) / 0.5) * 10
        elif "Crab" in pattern_type:
            confidence += (1 - min(abs(AB_XA_ratio - 0.382), abs(AB_XA_ratio - 0.618)) / 0.618) * 10

        # Adjust confidence based on pattern completion
        price_diff_pct = abs(current_price - completion_price) / completion_price

        if price_diff_pct < 0.01:
            # Price is very close to completion point - highest confidence
            confidence += 15
        elif price_diff_pct < 0.03:
            # Price is close to completion point
            confidence += 10
        elif price_diff_pct > 0.05:
            # Price is far from completion point
            confidence -= 15

        # Ensure confidence is within bounds
        confidence = max(0, min(confidence, 100))

        return confidence

    def _analyze_patterns(self, patterns, current_price, klines):
        """
        Analyze detected patterns and generate trading signals.

        Args:
            patterns (list): List of detected patterns
            current_price (float): Current price
            klines (pd.DataFrame): Historical price data

        Returns:
            tuple: (recommendation, confidence, reasoning, pattern_indicators)
        """
        # Initialize variables
        pattern_indicators = []

        # If no patterns detected, return HOLD
        if not patterns:
            return "HOLD", 50, "No harmonic patterns detected.", pattern_indicators

        # Add detected patterns to indicators
        for pattern in patterns:
            pattern_indicators.append(f"{pattern['type']} detected with {pattern['confidence']:.2f}% confidence")

            # Add completion price
            completion_price = pattern['completion_price']
            price_diff_pct = (current_price - completion_price) / completion_price * 100
            pattern_indicators.append(f"Completion price: {completion_price:.2f} (Current: {current_price:.2f}, Diff: {price_diff_pct:.2f}%)")

        # Get the highest confidence pattern
        best_pattern = patterns[0]
        pattern_type = best_pattern['type']
        confidence = best_pattern['confidence']

        # Generate recommendation based on pattern type and price position
        if "Bullish" in pattern_type and current_price >= best_pattern['completion_price']:
            recommendation = "BUY"
            reasoning = f"{pattern_type} detected with {confidence:.2f}% confidence. Price has reached or exceeded the pattern completion point, indicating a potential bullish reversal."
        elif "Bearish" in pattern_type and current_price <= best_pattern['completion_price']:
            recommendation = "SELL"
            reasoning = f"{pattern_type} detected with {confidence:.2f}% confidence. Price has reached or fallen below the pattern completion point, indicating a potential bearish reversal."
        else:
            # Pattern detected but price hasn't reached completion point
            recommendation = "HOLD"
            confidence = max(confidence - 20, 50)  # Reduce confidence for incomplete patterns
            reasoning = f"{pattern_type} detected but price hasn't reached the optimal entry point. Wait for price to reach {best_pattern['completion_price']:.2f} before taking action."

        # Check for confirmation signals
        if recommendation in ["BUY", "SELL"]:
            # Check RSI for confirmation
            rsi = talib.RSI(klines['close'], timeperiod=14)
            latest_rsi = rsi.iloc[-1]

            if recommendation == "BUY" and latest_rsi < 30:
                confidence = min(confidence + 10, 100)
                reasoning += f" RSI is oversold ({latest_rsi:.2f}), confirming the bullish signal."
                pattern_indicators.append(f"RSI: {latest_rsi:.2f} (oversold)")
            elif recommendation == "SELL" and latest_rsi > 70:
                confidence = min(confidence + 10, 100)
                reasoning += f" RSI is overbought ({latest_rsi:.2f}), confirming the bearish signal."
                pattern_indicators.append(f"RSI: {latest_rsi:.2f} (overbought)")
            elif (recommendation == "BUY" and latest_rsi > 70) or (recommendation == "SELL" and latest_rsi < 30):
                confidence = max(confidence - 10, 0)
                reasoning += f" RSI ({latest_rsi:.2f}) contradicts the pattern signal, reducing confidence."
                pattern_indicators.append(f"RSI: {latest_rsi:.2f} (contradicts pattern)")

            # Check volume for confirmation
            recent_volume = klines['volume'].iloc[-5:].mean()
            historical_volume = klines['volume'].iloc[-20:-5].mean()

            if recent_volume > historical_volume * 1.5:
                confidence = min(confidence + 5, 100)
                reasoning += " Increased volume supports the pattern signal."
                pattern_indicators.append("Volume is increasing (confirmation)")

        return recommendation, confidence, reasoning, pattern_indicators

    def _identify_support_resistance(self, patterns, pivots, current_price):
        """
        Identify support and resistance levels from harmonic patterns and pivots.

        Args:
            patterns (list): List of detected patterns
            pivots (list): List of pivot points
            current_price (float): Current price

        Returns:
            tuple: (support_levels, resistance_levels)
        """
        # Initialize support and resistance levels
        support_levels = []
        resistance_levels = []

        # Add pattern completion points as support/resistance
        for pattern in patterns:
            completion_price = pattern['completion_price']
            if "Bullish" in pattern['type'] and completion_price < current_price:
                support_levels.append(completion_price)
            elif "Bearish" in pattern['type'] and completion_price > current_price:
                resistance_levels.append(completion_price)

        # Add recent pivot points as support/resistance
        recent_pivots = pivots[-10:] if len(pivots) > 10 else pivots

        for pivot in recent_pivots:
            if pivot['type'] == 'low' and pivot['price'] < current_price:
                support_levels.append(pivot['price'])
            elif pivot['type'] == 'high' and pivot['price'] > current_price:
                resistance_levels.append(pivot['price'])

        # Sort levels
        support_levels = sorted(list(set(support_levels)), reverse=True)
        resistance_levels = sorted(list(set(resistance_levels)))

        return support_levels, resistance_levels

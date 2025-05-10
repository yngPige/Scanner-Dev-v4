"""
Volume Profile Strategy Implementation
This module implements a trading strategy based on volume profile analysis.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import talib
import TradingStrategy

class VolumeProfileStrategy(TradingStrategy):
    """
    A strategy that uses volume profile analysis to identify key price levels and potential breakouts.
    Volume profile helps identify areas of high trading activity and potential support/resistance.
    """

    def __init__(self, name="Volume Profile", description="Uses volume profile analysis to identify key price levels"):
        """
        Initialize the volume profile strategy.

        Args:
            name (str): Name of the strategy
            description (str): Description of the strategy
        """
        super().__init__(name, description)

        # Strategy parameters
        self.num_bins = 20  # Number of price bins for volume profile
        self.lookback_periods = 100  # Number of periods to look back for volume profile
        self.vwap_period = 20  # Period for VWAP calculation
        self.volume_threshold = 0.8  # Threshold for high volume nodes (percentile)
        self.poc_proximity = 0.02  # Proximity to Point of Control (as percentage of price)

    def generate_signal(self, symbol, klines, ticker=None):
        """
        Generate a trading signal based on volume profile analysis.

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

            # Calculate volume profile
            volume_profile, poc, value_area = self._calculate_volume_profile(klines)

            # Calculate VWAP
            vwap = self._calculate_vwap(klines)

            # Analyze volume profile and generate signals
            recommendation, confidence, reasoning, profile_indicators = self._analyze_volume_profile(
                current_price, volume_profile, poc, value_area, vwap, klines
            )

            # Calculate support and resistance levels
            support_levels, resistance_levels = self._identify_support_resistance(volume_profile, current_price)

            # Calculate stop loss and take profit levels
            if recommendation == 'BUY':
                # For buy signals, use the nearest support as stop loss
                stop_loss = max([level for level in support_levels if level < current_price], default=current_price * 0.95)
            elif recommendation == 'SELL':
                # For sell signals, use the nearest resistance as stop loss
                stop_loss = min([level for level in resistance_levels if level > current_price], default=current_price * 1.05)
            else:
                # For hold signals, use a default stop loss
                stop_loss = current_price * 0.95 if current_price > poc else current_price * 1.05

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
                key_indicators=profile_indicators
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

    def _calculate_volume_profile(self, klines):
        """
        Calculate volume profile from historical data.

        Args:
            klines (pd.DataFrame): Historical price data

        Returns:
            tuple: (volume_profile, point_of_control, value_area)
        """
        # Use recent data for volume profile
        recent_data = klines.iloc[-self.lookback_periods:]

        # Calculate price range
        price_min = recent_data['low'].min()
        price_max = recent_data['high'].max()
        price_range = price_max - price_min

        # Create price bins
        bin_size = price_range / self.num_bins
        bins = np.linspace(price_min, price_max, self.num_bins + 1)

        # Initialize volume profile
        volume_profile = []

        # Calculate volume for each price bin
        for i in range(len(bins) - 1):
            bin_low = bins[i]
            bin_high = bins[i + 1]
            bin_mid = (bin_low + bin_high) / 2

            # Calculate volume in this price range
            mask = (recent_data['low'] <= bin_high) & (recent_data['high'] >= bin_low)
            bin_volume = recent_data.loc[mask, 'volume'].sum()

            volume_profile.append({
                'price_low': bin_low,
                'price_high': bin_high,
                'price_mid': bin_mid,
                'volume': bin_volume
            })

        # Convert to DataFrame
        volume_profile_df = pd.DataFrame(volume_profile)

        # Find Point of Control (POC) - price level with highest volume
        poc_idx = volume_profile_df['volume'].idxmax()
        poc = volume_profile_df.iloc[poc_idx]['price_mid']

        # Calculate Value Area (70% of total volume)
        total_volume = volume_profile_df['volume'].sum()
        value_area_target = total_volume * 0.7

        # Sort by volume in descending order
        sorted_profile = volume_profile_df.sort_values('volume', ascending=False)
        cumulative_volume = 0
        value_area_bins = []

        for _, row in sorted_profile.iterrows():
            cumulative_volume += row['volume']
            value_area_bins.append((row['price_low'], row['price_high']))
            if cumulative_volume >= value_area_target:
                break

        # Find value area high and low
        value_area_low = min([low for low, _ in value_area_bins])
        value_area_high = max([high for _, high in value_area_bins])
        value_area = (value_area_low, value_area_high)

        return volume_profile_df, poc, value_area

    def _calculate_vwap(self, klines):
        """
        Calculate Volume Weighted Average Price (VWAP).

        Args:
            klines (pd.DataFrame): Historical price data

        Returns:
            float: VWAP value
        """
        # Use recent data for VWAP
        recent_data = klines.iloc[-self.vwap_period:]

        # Calculate typical price
        typical_price = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3

        # Calculate VWAP
        vwap = (typical_price * recent_data['volume']).sum() / recent_data['volume'].sum()

        return vwap

    def _analyze_volume_profile(self, current_price, volume_profile, poc, value_area, vwap, klines):
        """
        Analyze volume profile and generate trading signals.

        Args:
            current_price (float): Current price
            volume_profile (pd.DataFrame): Volume profile data
            poc (float): Point of Control
            value_area (tuple): Value Area (low, high)
            vwap (float): Volume Weighted Average Price
            klines (pd.DataFrame): Historical price data

        Returns:
            tuple: (recommendation, confidence, reasoning, profile_indicators)
        """
        # Initialize variables
        profile_indicators = []
        value_area_low, value_area_high = value_area

        # Add POC and Value Area to indicators
        profile_indicators.append(f"Point of Control: {poc:.2f}")
        profile_indicators.append(f"Value Area: {value_area_low:.2f} - {value_area_high:.2f}")
        profile_indicators.append(f"VWAP: {vwap:.2f}")

        # Check if price is near POC
        poc_proximity_pct = self.poc_proximity
        near_poc = abs(current_price - poc) / current_price < poc_proximity_pct
        if near_poc:
            profile_indicators.append(f"Price is near Point of Control ({poc:.2f})")

        # Check if price is inside or outside Value Area
        inside_value_area = value_area_low <= current_price <= value_area_high
        if inside_value_area:
            profile_indicators.append("Price is inside Value Area (balanced market)")
        else:
            profile_indicators.append("Price is outside Value Area (potential imbalance)")

        # Check relationship with VWAP
        above_vwap = current_price > vwap
        if above_vwap:
            profile_indicators.append("Price is above VWAP (bullish)")
        else:
            profile_indicators.append("Price is below VWAP (bearish)")

        # Find high volume nodes (HVN) and low volume nodes (LVN)
        volume_threshold = volume_profile['volume'].quantile(self.volume_threshold)
        hvn_prices = volume_profile[volume_profile['volume'] >= volume_threshold]['price_mid'].tolist()
        lvn_prices = volume_profile[volume_profile['volume'] < volume_threshold * 0.2]['price_mid'].tolist()

        # Check if price is breaking through a high volume node
        breaking_hvn = False
        for hvn in hvn_prices:
            # Check if price recently crossed this HVN
            if (klines['close'].iloc[-2] < hvn < current_price) or (klines['close'].iloc[-2] > hvn > current_price):
                breaking_hvn = True
                profile_indicators.append(f"Price breaking through high volume node at {hvn:.2f}")
                break

        # Generate recommendation based on volume profile analysis
        if current_price > value_area_high and above_vwap:
            # Price above value area and VWAP - bullish
            recommendation = "BUY"
            confidence = 70
            reasoning = "Price is above Value Area High and VWAP, indicating bullish momentum."
            
            # Increase confidence if breaking through HVN
            if breaking_hvn and current_price > klines['close'].iloc[-2]:
                confidence += 15
                reasoning += " Price is breaking through a high volume node with increasing price."
                
        elif current_price < value_area_low and not above_vwap:
            # Price below value area and VWAP - bearish
            recommendation = "SELL"
            confidence = 70
            reasoning = "Price is below Value Area Low and VWAP, indicating bearish momentum."
            
            # Increase confidence if breaking through HVN
            if breaking_hvn and current_price < klines['close'].iloc[-2]:
                confidence += 15
                reasoning += " Price is breaking through a high volume node with decreasing price."
                
        elif near_poc:
            # Price near POC - potential reversal or continuation
            if above_vwap:
                recommendation = "BUY"
                confidence = 60
                reasoning = "Price is near Point of Control and above VWAP, potential bullish continuation."
            else:
                recommendation = "SELL"
                confidence = 60
                reasoning = "Price is near Point of Control and below VWAP, potential bearish continuation."
        else:
            # No clear signal
            recommendation = "HOLD"
            confidence = 50
            reasoning = "No clear volume profile signal."

        # Check for volume confirmation
        recent_volume = klines['volume'].iloc[-5:].mean()
        historical_volume = klines['volume'].iloc[-20:-5].mean()
        
        if recent_volume > historical_volume * 1.5:
            confidence = min(confidence + 10, 100)
            reasoning += " Increased volume confirms the signal."
            profile_indicators.append("Volume is increasing (confirmation)")
        elif recent_volume < historical_volume * 0.7:
            confidence = max(confidence - 10, 0)
            reasoning += " Decreased volume weakens the signal."
            profile_indicators.append("Volume is decreasing (caution)")

        return recommendation, confidence, reasoning, profile_indicators

    def _identify_support_resistance(self, volume_profile, current_price):
        """
        Identify support and resistance levels from volume profile.

        Args:
            volume_profile (pd.DataFrame): Volume profile data
            current_price (float): Current price

        Returns:
            tuple: (support_levels, resistance_levels)
        """
        # Find high volume nodes
        volume_threshold = volume_profile['volume'].quantile(self.volume_threshold)
        high_volume_nodes = volume_profile[volume_profile['volume'] >= volume_threshold]['price_mid'].tolist()
        
        # Separate into support and resistance
        support_levels = [price for price in high_volume_nodes if price < current_price]
        resistance_levels = [price for price in high_volume_nodes if price > current_price]
        
        # Sort levels
        support_levels = sorted(support_levels, reverse=True)
        resistance_levels = sorted(resistance_levels)
        
        return support_levels, resistance_levels

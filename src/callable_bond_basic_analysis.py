src/callable_bond_basic_analysis.py
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 19:47:05 2025

@author: amits
"""

# -*- coding: utf-8 -*-
"""
OAD, KRD and OAC calculations
"""
import QuantLib as ql
import numpy as np
import pandas as pd

# [Previous setup code remains the same through base_price calculation]
calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

issue_date = ql.Date(15, 12, 2022)
maturity_date = ql.Date(15, 12, 2032)
call_price = 100.0
call_date = ql.Date(15, 12, 2027)
schedule = ql.Schedule(issue_date, maturity_date, ql.Period(ql.Annual), calendar,
                       ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)

callability_schedule = ql.CallabilitySchedule()
callability_schedule.append(
    ql.Callability(
        ql.BondPrice(call_price, ql.BondPrice.Clean),
        ql.Callability.Call,
        call_date
    )
)

settlement_days = 1
face_value = 100
coupons = [0.05]
bond = ql.CallableFixedRateBond(settlement_days, face_value, schedule, coupons,
                                ql.ActualActual(ql.ActualActual.Bond), ql.Following, 100.0, issue_date, callability_schedule)

# Create dates starting exactly 1 year from today
def create_term_structure_dates(start_date, num_tenors=8):
    """
    Create term structure dates starting exactly 1 year from start_date
    Returns 8 yearly tenors: 1Y, 2Y, 3Y, ..., 8Y
    """
    dates = []
    for i in range(1, num_tenors + 1):  # 1 to 8 years
        # Add exactly i years to today
        tenor_date = start_date + ql.Period(i, ql.Years)
        dates.append(tenor_date)
    return dates

# Your zero rates (unchanged)
zero_rates = [0.0395, 0.0378, 0.0378, 0.03845, 0.0391, 0.0395, 0.0408, 0.0428]

# Generate dates dynamically
zero_dates = create_term_structure_dates(today, 8)

def build_curve_from_zeros(dates, rates):
    bond_helpers = []
    for d, r in zip(dates, rates):
        price = 100 * np.exp(-r * ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, d))
        # Create schedule with only maturity payment (no intermediate coupons)
        helper_schedule = ql.Schedule(today, d, ql.Period(ql.Once), calendar, 
                                    ql.Unadjusted, ql.Unadjusted, 
                                    ql.DateGeneration.Backward, False)
        
        helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(price)),
            settlement_days, 
            face_value, 
            helper_schedule, 
            [0.0],  # Zero coupon rate
            ql.ActualActual(ql.ActualActual.Bond)
        )
        bond_helpers.append(helper)
    
    curve = ql.PiecewiseLinearZero(today, bond_helpers, ql.ActualActual(ql.ActualActual.Bond))
    return ql.YieldTermStructureHandle(curve)

base_curve_handle = build_curve_from_zeros(zero_dates, zero_rates)
hw_model = ql.HullWhite(base_curve_handle, a=0.03, sigma=0.015)
engine = ql.TreeCallableFixedRateBondEngine(hw_model, 500)
bond.setPricingEngine(engine)
base_price = bond.cleanPrice()

# Calculate OAD (parallel shift method)
def shifted_curve(base_handle, shift_bps):
    shifted_curve = ql.ZeroSpreadedTermStructure(base_handle,
                                                 ql.QuoteHandle(ql.SimpleQuote(shift_bps)))
    return ql.YieldTermStructureHandle(shifted_curve)

shift = 0.0001  # 1 bp

hw_model_plus = ql.HullWhite(shifted_curve(base_curve_handle, shift), a=0.03, sigma=0.015)
bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_plus, 500))
price_plus = bond.cleanPrice()

hw_model_minus = ql.HullWhite(shifted_curve(base_curve_handle, -shift), a=0.03, sigma=0.015)
bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_minus, 500))
price_minus = bond.cleanPrice()

oad = (price_minus - price_plus) / (2 * base_price * shift)

# Calculate Convexity (second derivative)
# Convexity = (P+ + P- - 2*P0) / (P0 * (Δy)²)
convexity = (price_minus + price_plus - 2 * base_price) / (base_price * shift**2)

# Alternative KRD Method 1: Using ZeroSpreadedTermStructure with key rate interpolation
def calculate_krd_method1(key_rate_index, shift_size=0.0001):
    """
    KRD using interpolated key rate shifts - more consistent with OAD approach
    """
    # Calculate time to each key rate maturity
    key_times = [ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, d) for d in zero_dates]
    target_time = key_times[key_rate_index]
    
    # Create a bump function that affects primarily the target maturity
    # with linear decay to neighboring points
    def create_bump_function(target_time, key_times, bump_size):
        # Simple approach: bump only the exact key rate
        bump_vector = [0.0] * len(key_times)
        bump_vector[key_rate_index] = bump_size
        return bump_vector
    
    # Calculate price with positive bump
    bump_up = create_bump_function(target_time, key_times, shift_size)
    rates_up = [r + b for r, b in zip(zero_rates, bump_up)]
    curve_up = build_curve_from_zeros(zero_dates, rates_up)
    
    hw_model_up = ql.HullWhite(curve_up, a=0.03, sigma=0.015)
    bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_up, 500))
    price_up = bond.cleanPrice()
    
    # Calculate price with negative bump
    bump_down = create_bump_function(target_time, key_times, -shift_size)
    rates_down = [r + b for r, b in zip(zero_rates, bump_down)]
    curve_down = build_curve_from_zeros(zero_dates, rates_down)
    
    hw_model_down = ql.HullWhite(curve_down, a=0.03, sigma=0.015)
    bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_down, 500))
    price_down = bond.cleanPrice()
    
    # Calculate KRD
    krd = (price_down - price_up) / (2 * base_price * shift_size)
    # Calculate Key Rate Convexity (second derivative)
    key_rate_convexity = (price_up + price_down - 2 * base_price) / (base_price * shift_size**2)
    
    return krd, price_up, price_down, key_rate_convexity

print("="*80)
print("DEBUGGING KRD CALCULATIONS")
print("="*80)
print(f"Base Price: {base_price:.4f}")
print(f"OAD: {oad:.4f}")
print()

# Method 1 results
krd_results_m1 = []
debug_info = []

for i, (date, rate) in enumerate(zip(zero_dates, zero_rates)):
    tenor_years = ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, date)
    krd_val, price_up, price_down, key_rate_convexity = calculate_krd_method1(i, shift)
    krd_results_m1.append(krd_val)
    
    debug_info.append({
        'Index': i,
        'Tenor': f"{tenor_years:.1f}Y",
        'Base_Rate_%': rate*100,
        'Price_Up': price_up,
        'Price_Down': price_down,
        'Price_Change_Up': price_up - base_price,
        'Price_Change_Down': price_down - base_price,
        'KRD': krd_val,
        'Key_Rate_Convexity': key_rate_convexity
    })

# Create debug DataFrame
debug_df = pd.DataFrame(debug_info)
print("DETAILED KRD CALCULATION DEBUG:")
print("="*80)
print(debug_df.to_string(index=False, float_format='%.4f'))

total_krd_m1 = sum(krd_results_m1)
total_krc = sum(debug_df['Key_Rate_Convexity'])

print(f"\nSUMMARY:")
print(f"OAD (parallel shift): {oad:.4f}")
print(f"Option-Adjusted Convexity: {convexity:.4f}")
print(f"Total KRD (method 1): {total_krd_m1:.4f}")
print(f"Difference: {abs(total_krd_m1 - oad):.4f}")
print(f"Relative Error: {abs(total_krd_m1 - oad)/oad*100:.2f}%")

# Analyze the components
print(f"\nKRD BREAKDOWN:")
print("-" * 40)
for i, (tenor, krd_val) in enumerate(zip([f"{ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, d):.1f}Y" for d in zero_dates], krd_results_m1)):
    contribution = abs(krd_val) / sum([abs(k) for k in krd_results_m1]) * 100
    print(f"{tenor:>8}: {krd_val:>8.4f} ({contribution:>5.1f}%)")

# Check for potential issues
print(f"\nDIAGNOSTIC CHECKS:")
print(f"1. Any negative KRDs? {any(k < 0 for k in krd_results_m1)}")
print(f"2. Largest absolute KRD: {max([abs(k) for k in krd_results_m1]):.4f}")
print(f"3. Sum of absolute KRDs: {sum([abs(k) for k in krd_results_m1]):.4f}")
print(f"4. Bond callable from: {call_date}")

# Reset bond engine
hw_model_base = ql.HullWhite(base_curve_handle, a=0.03, sigma=0.015)
bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_base, 500))

# ----------------------------------Option Value analysis-------------------------------
# Create identical bond without call option
option_free_bond = ql.FixedRateBond(settlement_days, face_value, schedule, coupons,
                                   ql.ActualActual(ql.ActualActual.Bond))
option_free_bond.setPricingEngine(ql.DiscountingBondEngine(base_curve_handle))
option_free_price = option_free_bond.cleanPrice()

print(f"\nCall Option price valuation:")
print("-" * 40)
print(f"Option-free bond price: {option_free_price:.2f}")
print(f"Callable bond price: {base_price:.2f}")
print(f"Call option value: {option_free_price - base_price:.2f}")

# -------------------Convexity analysis----------------------------------------------------
# Try larger shifts to see convexity behavior
large_shift = 0.005  # 50bp instead of 1bp

# Recalculate convexity with larger shifts
hw_model_plus_large = ql.HullWhite(shifted_curve(base_curve_handle, -large_shift), a=0.03, sigma=0.015)
bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_plus_large, 500))
price_plus_large = bond.cleanPrice()

hw_model_minus_large = ql.HullWhite(shifted_curve(base_curve_handle, -large_shift), a=0.03, sigma=0.015)
bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_minus_large, 500))
price_minus_large = bond.cleanPrice()

convexity_large = (price_plus_large + price_minus_large - 2 * base_price) / (base_price * large_shift**2)

print(f"\nConvexity with large shifts:")
print("-" * 40)
print(f"Convexity with 50bp shifts: {convexity_large:.4f}")

# --------------------Price Yield Curve Plot--------------------------------------


import matplotlib.pyplot as plt


# Add this to your existing code after calculating base_price

def plot_yield_price_curve(zero_dates, zero_rates, base_curve_handle, bond, hw_params, call_price):
    """
    Enhanced plotting with wider range and better tree resolution
    """
    # Much wider shift range to capture convexity
    # yield_shifts = np.linspace(-0.05, 0.05, 51)  # -500bp to +500bp in 20bp increments
    yield_shifts = np.linspace(0, 0.04, 51)  # -500bp to +500bp in 20bp increments
    
    callable_prices = []
    option_free_prices = []
    
    # Create option-free version of the bond for comparison
    option_free_bond = ql.FixedRateBond(
        settlement_days, face_value, schedule, coupons,
        ql.ActualActual(ql.ActualActual.Bond)
    )
    
    print("Calculating price-yield relationship with wider range...")
    print("Shift (bp)|Callable | Option-Free|Difference|Call Value")
    print("-" * 65)
    
    for i, shift in enumerate(yield_shifts):
        try:
            # Use ZeroSpreadedTermStructure but with wider shifts
            shifted_curve = ql.ZeroSpreadedTermStructure(
                base_curve_handle, 
                ql.QuoteHandle(ql.SimpleQuote(shift))
            )
            shifted_curve_handle = ql.YieldTermStructureHandle(shifted_curve)
            
            # Price callable bond with MORE tree steps for accuracy
            hw_model_shifted = ql.HullWhite(shifted_curve_handle, a=hw_params['a'], sigma=hw_params['sigma'])
            callable_engine = ql.TreeCallableFixedRateBondEngine(hw_model_shifted, 1000)  # More steps
            bond.setPricingEngine(callable_engine)
            callable_price = bond.cleanPrice()
            callable_prices.append(callable_price)
            
            # Price option-free bond
            option_free_engine = ql.DiscountingBondEngine(shifted_curve_handle)
            option_free_bond.setPricingEngine(option_free_engine)
            option_free_price = option_free_bond.cleanPrice()
            option_free_prices.append(option_free_price)
            
            # Calculate embedded call option value
            call_option_value = option_free_price - callable_price
            
            # Print every 5th point
            if i % 5 == 0:
                print(f"{shift*10000:>8.0f} | {callable_price:>8.2f} | {option_free_price:>11.2f} | {callable_price - option_free_price:>+10.2f} | {call_option_value:>8.2f}")
        
        except Exception as e:
            print(f"Error at shift {shift*10000:.0f}bp: {e}")
            # Use interpolation for failed points
            if len(callable_prices) >= 2:
                callable_prices.append(callable_prices[-1])
                option_free_prices.append(option_free_prices[-1])
            else:
                callable_prices.append(20)  # Fallback price
                option_free_prices.append(100)
    
    # Create the plot with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Price vs Yield
    yield_shifts_bp = yield_shifts * 10000
    valid_indices = [i for i, p in enumerate(callable_prices) if p > 0]
    
    if valid_indices:
        valid_shifts = [yield_shifts_bp[i] for i in valid_indices]
        valid_callable = [callable_prices[i] for i in valid_indices]
        valid_option_free = [option_free_prices[i] for i in valid_indices]
        
        ax1.plot(valid_shifts, valid_option_free, 'b-', linewidth=3, label='Option-Free Bond', alpha=0.8)
        ax1.plot(valid_shifts, valid_callable, 'r-', linewidth=3, label='Callable Bond', alpha=0.8)
        
        # Add call price line
        ax1.axhline(y=call_price, color='gray', linestyle='--', alpha=0.7, linewidth=2, label=f'Call Price ({call_price})')
        
        # Mark current point (zero shift)
        zero_idx = np.argmin(np.abs(np.array(valid_shifts)))
        if zero_idx < len(valid_callable):
            current_callable = valid_callable[zero_idx]
            current_option_free = valid_option_free[zero_idx]
            ax1.plot(valid_shifts[zero_idx], current_callable, 'ro', markersize=10, label='Current Callable Price')
            ax1.plot(valid_shifts[zero_idx], current_option_free, 'bo', markersize=10, label='Current Option-Free Price')
        
        # Enhanced formatting
        ax1.set_xlabel('Yield Shift (basis points)', fontsize=12)
        ax1.set_ylabel('Bond Price', fontsize=12)
        ax1.set_title('Callable vs Option-Free Bond: Price-Yield Relationship', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Add curvature annotations
        ax1.annotate('Positive Convexity\n(upward curve)', 
                    xy=(valid_shifts[5], valid_option_free[5]), 
                    xytext=(valid_shifts[5]-100, valid_option_free[5]+20),
                    arrowprops=dict(arrowstyle='->', color='blue', alpha=0.8),
                    fontsize=10, ha='center', color='blue', weight='bold')
        
        ax1.annotate('Negative Convexity\n(flattened curve)', 
                    xy=(valid_shifts[5], valid_callable[5]), 
                    xytext=(valid_shifts[5]-100, valid_callable[5]-20),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.8),
                    fontsize=10, ha='center', color='red', weight='bold')
    
    # Plot 2: Call Option Value vs Yield
    if valid_indices:
        call_option_values = [valid_option_free[i] - valid_callable[i] for i in range(len(valid_callable))]
        ax2.plot(valid_shifts, call_option_values, 'g-', linewidth=3, label='Embedded Call Option Value', alpha=0.8)
        ax2.fill_between(valid_shifts, 0, call_option_values, alpha=0.3, color='green')
        
        ax2.set_xlabel('Yield Shift (basis points)', fontsize=12)
        ax2.set_ylabel('Call Option Value', fontsize=12)
        ax2.set_title('Embedded Call Option Value vs Yield Changes', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        # Mark current option value
        current_option_value = call_option_values[zero_idx] if zero_idx < len(call_option_values) else 0
        ax2.plot(valid_shifts[zero_idx], current_option_value, 'go', markersize=10, label='Current Option Value')
        
        # Add annotation about option behavior
        max_option_value = max(call_option_values)
        max_idx = call_option_values.index(max_option_value)
        ax2.annotate(f'Max Option Value: {max_option_value:.1f}\n(when rates fall)', 
                    xy=(valid_shifts[max_idx], max_option_value), 
                    xytext=(valid_shifts[max_idx]+100, max_option_value-10),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.8),
                    fontsize=10, ha='center', color='green', weight='bold')
    
    plt.tight_layout()
    
    # Calculate convexity at different shift sizes
    print(f"\nCONVEXITY ANALYSIS WITH WIDER SHIFTS:")
    print("-" * 50)
    
    if len(callable_prices) > 10:
        base_idx = len(callable_prices) // 2
        base_price_calc = callable_prices[base_idx]
        
        for shift_size in [0.001, 0.005, 0.01, 0.02]:  # 10bp, 50bp, 100bp, 200bp
            shift_bp = shift_size * 10000
            
            # Find nearest indices
            up_target = shift_size
            down_target = -shift_size
            
            up_idx = min(range(len(yield_shifts)), key=lambda i: abs(yield_shifts[i] - up_target))
            down_idx = min(range(len(yield_shifts)), key=lambda i: abs(yield_shifts[i] - down_target))
            
            if up_idx < len(callable_prices) and down_idx < len(callable_prices):
                price_up = callable_prices[up_idx]
                price_down = callable_prices[down_idx]
                
                convexity = (price_up + price_down - 2 * base_price_calc) / (base_price_calc * shift_size**2)
                
                print(f"Convexity with ±{shift_bp:3.0f}bp shifts: {convexity:>12.1f}")
    
    plt.show()
    
    return yield_shifts_bp, callable_prices, option_free_prices

# Call the function with your parameters
hw_params = {'a': 0.03, 'sigma': 0.015}  # Your Hull-White parameters
yield_shifts, callable_prices, option_free_prices = plot_yield_price_curve(
    zero_dates, zero_rates, base_curve_handle, bond, hw_params, call_price
)

src/callable_bond_ytw_oas_analysis.py
# -*- coding: utf-8 -*-
"""
Complete Callable Bond Analysis: Yield-to-Worst (YTW) and Option-Adjusted Spread (OAS)
Author: Amit
Date: May 2025
"""

import QuantLib as ql
import numpy as np
import pandas as pd


print("="*80)
print("CALLABLE BOND: YIELD-TO-WORST AND OPTION-ADJUSTED SPREAD ANALYSIS")
print("="*80)

# ============================================================================
# 1. BASIC SETUP
# ============================================================================

# Calendar and evaluation date
calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
today = ql.Date.todaysDate()  # Use actual today's date
ql.Settings.instance().evaluationDate = today

print(f"Evaluation Date: {today}")
print()

# ============================================================================
# 2. BOND SPECIFICATIONS
# ============================================================================

# Bond basic terms - adjusted to ensure all dates are within curve limits
max_bond_maturity = today + ql.Period(int(9), ql.Years)

issue_date = ql.Date(15, 6, 2023)  # June 2023 (before today)
maturity_date = min(ql.Date(15, 6, 2033), max_bond_maturity)  # Constrain to curve limit
first_call_date = today + ql.Period(2, ql.Years)  # 2 years from today (safe)
call_price = 95.0
settlement_days = 1
face_value = 100
coupon_rate = 0.05  # 5% annual coupon
price_today = 85

print("BOND SPECIFICATIONS:")
print(f"Issue Date: {issue_date}")
print(f"Maturity Date: {maturity_date}")
print(f"First Call Date: {first_call_date}")
print(f"Call Price: ${call_price}")
print(f"Coupon Rate: {coupon_rate*100}%")
print()

# ============================================================================
# 3. YIELD CURVE CONSTRUCTION
# ============================================================================
def create_term_structure_dates(start_date, num_tenors=8):
    """Create term structure dates with proper spacing to ensure 8+ years coverage"""
    dates = []
    for i in range(1, num_tenors + 1):
        # Create dates at 1Y, 2Y, 3Y, ... 8Y from start_date
        tenor_date = start_date + ql.Period(i, ql.Years)
        dates.append(tenor_date)
    return dates


# Realistic zero rates
zero_rates = [0.045, 0.048, 0.050, 0.051, 0.052, 0.053, 0.054, 0.055, 0.055]
zero_dates = create_term_structure_dates(today, 9)

# Validate all dates and calculate actual curve coverage
print("YIELD CURVE VALIDATION:")
valid_dates = []
valid_rates = []

for i, (date, rate) in enumerate(zip(zero_dates, zero_rates)):
    years = ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, date)
    if years > 0:
        valid_dates.append(date)
        valid_rates.append(rate)
        print(f"  {years:.1f}Y: {rate*100:.1f}% ✓")
    else:
        print(f"  {date}: SKIPPED (in the past)")

# Use only valid dates
zero_dates = valid_dates
zero_rates = valid_rates

if len(zero_dates) == 0:
    print("ERROR: No valid future dates found!")
    exit()

# Calculate ACTUAL maximum curve coverage
max_curve_years = max([ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, d) for d in zero_dates])
print(f"Using {len(zero_dates)} valid tenor points")
print(f"Actual curve coverage: 0Y to {max_curve_years:.2f}Y")

# Add some buffer for interpolation/extrapolation
curve_buffer = 0.5  # 6 months buffer
effective_max_time = max_curve_years - curve_buffer
print(f"Safe calculation range: 0Y to {effective_max_time:.2f}Y")
print()

# Build yield curve using direct construction
def build_yield_curve(dates, rates):
    """Build yield curve directly from zero rates"""
    # return ql.YieldTermStructureHandle(
    #     ql.ZeroCurve(dates, rates, ql.ActualActual(ql.ActualActual.Bond))
    # )
    bond_helpers = []
    for d, r in zip(dates, rates):
        price = 100 * np.exp(-r * ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, d))
        helper_schedule = ql.Schedule(today, d, ql.Period(ql.Once), calendar, 
                                    ql.Unadjusted, ql.Unadjusted, 
                                    ql.DateGeneration.Backward, False)
        
        helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(price)),
            settlement_days, face_value, helper_schedule, [0.0],
            ql.ActualActual(ql.ActualActual.Bond)
        )
        bond_helpers.append(helper)
    
    curve = ql.PiecewiseLinearZero(today, bond_helpers, ql.ActualActual(ql.ActualActual.Bond))
    return ql.YieldTermStructureHandle(curve)


base_yield_curve = build_yield_curve(zero_dates, zero_rates)

# ============================================================================
# 4. CREATE CALL SCHEDULE (AMERICAN-STYLE)
# ============================================================================

def create_call_schedule(first_call_date, maturity_date, call_price, frequency='Quarterly'):
    """Create American-style call schedule with multiple call dates"""
    callability_schedule = ql.CallabilitySchedule()
    call_dates = []
    
    # Frequency mapping
    freq_map = {
        'Monthly': ql.Period(1, ql.Months),
        'Quarterly': ql.Period(3, ql.Months),
        'SemiAnnual': ql.Period(6, ql.Months),
        'Annual': ql.Period(1, ql.Years)
    }
    
    period = freq_map.get(frequency, ql.Period(3, ql.Months))
    current_date = first_call_date
    
    while current_date <= maturity_date:
        # Ensure call date is in the future AND within curve limits
        years_to_call = ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, current_date)
        
        if current_date > today and years_to_call <= max_curve_years:
            adjusted_date = calendar.adjust(current_date, ql.Following)
            
            callability_schedule.append(
                ql.Callability(
                    ql.BondPrice(call_price, ql.BondPrice.Clean),
                    ql.Callability.Call,
                    adjusted_date
                )
            )
            call_dates.append(adjusted_date)
        
        current_date = current_date + period
    
    return callability_schedule, call_dates

# Create call schedule
callability_schedule, call_dates = create_call_schedule(
    first_call_date, maturity_date, call_price, 'Quarterly'
)

print(f"CALL SCHEDULE:")
print(f"Number of call opportunities: {len(call_dates)}")
print(f"First 5 call dates:")
for i, date in enumerate(call_dates[:5]):
    years = ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, date)
    print(f"  {i+1}. {date} ({years:.2f}Y)")
print(f"... and {len(call_dates)-5} more quarterly opportunities")
print()

# ============================================================================
# 5. CREATE CALLABLE BOND
# ============================================================================

# Bond payment schedule
bond_schedule = ql.Schedule(
    issue_date, maturity_date, ql.Period(ql.Annual), calendar,
    ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False
)

# Create callable bond
callable_bond = ql.CallableFixedRateBond(
    settlement_days, face_value, bond_schedule, [coupon_rate],
    ql.ActualActual(ql.ActualActual.Bond), ql.Following, face_value,
    issue_date, callability_schedule
)

# ============================================================================
# 6. PRICE CALLABLE BOND USING HULL-WHITE MODEL
# ============================================================================

# Hull-White model parameters (realistic)
hw_volatility = 0.015  # 1.5%
hw_mean_reversion = 0.03  # 3%
tree_steps = 1000

# Create Hull-White model and pricing engine
hw_model = ql.HullWhite(base_yield_curve, a=hw_mean_reversion, sigma=hw_volatility)
pricing_engine = ql.TreeCallableFixedRateBondEngine(hw_model, tree_steps)
callable_bond.setPricingEngine(pricing_engine)

# Price the callable bond
callable_bond_price = callable_bond.cleanPrice()

# Price option-free equivalent for comparison
option_free_bond = ql.FixedRateBond(
    settlement_days, face_value, bond_schedule, [coupon_rate],
    ql.ActualActual(ql.ActualActual.Bond)
)
option_free_engine = ql.DiscountingBondEngine(base_yield_curve)
option_free_bond.setPricingEngine(option_free_engine)
option_free_price = option_free_bond.cleanPrice()

# Calculate embedded option value
embedded_option_value = option_free_price - callable_bond_price

print("PRICING RESULTS:")
print(f"Option-Free Bond Price: ${option_free_price:.4f}")
print(f"Callable Bond Price: ${callable_bond_price:.4f}")
print(f"Embedded Option Value: ${embedded_option_value:.4f}")
print(f"Hull-White Volatility: {hw_volatility*100:.1f}%")
print()

# ============================================================================
# 7. YIELD-TO-WORST CALCULATION USING QUANTLIB
# ============================================================================

def calculate_yield_to_worst(market_price, call_dates, call_price, maturity_date, coupon_rate, face_value):
    """
    Calculate Yield-to-Worst using QuantLib's built-in yield calculations
    Creates separate bond objects for each call scenario
    """
    yields = []
    scenarios = []
    day_counter = ql.ActualActual(ql.ActualActual.Bond)
    
    print("YIELD-TO-WORST CALCULATION:")
    print("-" * 40)
    
    # Scenario 1: Yield-to-Maturity (bond runs to full maturity)
    try:
        # ytm = callable_bond.yield_(market_price, day_counter, ql.Compounded, ql.Annual)
        # ytm = option_free_bond.bondYield(market_price, day_counter, ql.Compounded, ql.Annual)
        ytm = option_free_bond.bondYield((93.0778/face_value)*100, day_counter, ql.Compounded, ql.Annual)
        years_to_maturity = day_counter.yearFraction(today, maturity_date)
        
        yields.append(ytm)
        scenarios.append({
            'Type': 'Maturity',
            'Date': maturity_date,
            'Years': years_to_maturity,
            'Price': face_value,
            'Yield': ytm
        })
        
        print(f"Yield-to-Maturity: {ytm*100:.4f}% ({years_to_maturity:.2f} years)")
        
    except Exception as e:
        print(f"YTM calculation failed: {e}")
    
    # Scenario 2+: Yield-to-Call for each call date
    print("\nYield-to-Call scenarios:")
    
    for i, call_date in enumerate(call_dates[:8]):  # Limit to first 8 for display
        try:
            years_to_call = day_counter.yearFraction(today, call_date)
            
            # Additional safety check - ensure within curve bounds
            if years_to_call > max_curve_years:
                print(f"  Call {call_date}: SKIPPED (beyond curve limit)")
                continue
            
            # Create new bond that "matures" on call date with call price
            call_schedule = ql.Schedule(
                issue_date, call_date, ql.Period(ql.Annual), calendar,
                ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False
            )
            
            # Bond that matures on call date with call price as face value
            call_scenario_bond = ql.FixedRateBond(
                settlement_days, call_price, call_schedule, [coupon_rate],
                day_counter
            )
            
            # Calculate yield for this call scenario
            # ytc = call_scenario_bond.yield_(market_price, day_counter, ql.Compounded, ql.Annual)
            # ytc = call_scenario_bond.bondYield(market_price, day_counter, ql.Compounded, ql.Annual)
            ytc = call_scenario_bond.bondYield((93.0778/call_price)*100, day_counter, ql.Compounded, ql.Annual)
            
            yields.append(ytc)
            scenarios.append({
                'Type': 'Call',
                'Date': call_date,
                'Years': years_to_call,
                'Price': call_price,
                'Yield': ytc
            })
            
            print(f"  Call {call_date}: {ytc*100:.4f}% ({years_to_call:.2f} years)")
            
        except Exception as e:
            print(f"  YTC calculation failed for {call_date}: {e}")
    
    # Calculate Yield-to-Worst (minimum yield)
    if yields:
        ytw = min(yields)
        worst_scenario = min(scenarios, key=lambda x: x['Yield'])
        return scenarios, ytw, worst_scenario
    else:
        return [], None, None

# Execute YTW calculation
scenarios, ytw, worst_scenario = calculate_yield_to_worst(
    callable_bond_price, call_dates, call_price, maturity_date, coupon_rate, face_value
)

print("\n" + "="*50)
print("YIELD-TO-WORST RESULTS:")
print("="*50)

if ytw:
    print(f"Yield-to-Worst: {ytw*100:.4f}%")
    print(f"Worst case scenario: {worst_scenario['Type']} on {worst_scenario['Date']}")
    print(f"Time to worst case: {worst_scenario['Years']:.2f} years")
    print(f"Expected final payment: ${worst_scenario['Price']:.2f}")
    
    # Determine most likely outcome
    if worst_scenario['Type'] == 'Call':
        print(f"→ Bond likely to be CALLED early")
        print(f"→ Expected life: {worst_scenario['Years']:.2f} years (not full {ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, maturity_date):.2f} years)")
    else:
        print(f"→ Bond likely to run to MATURITY")
else:
    print("Could not calculate Yield-to-Worst")

print()

# ============================================================================
# 8. OPTION-ADJUSTED SPREAD (OAS) CALCULATION
# ============================================================================

def calculate_oas(callable_bond_price, base_yield_curve, hw_model, tree_steps=1000):
    """
    Calculate Option-Adjusted Spread (OAS)
    OAS is the spread that when added to the yield curve makes the model price = market price
    """
    
    print("OPTION-ADJUSTED SPREAD CALCULATION:")
    print("-" * 40)
    
    target_price = callable_bond_price
    
    def price_with_spread(spread):
        """Price callable bond with given spread added to yield curve"""
        try:
            # Create spread curve
            spread_curve = ql.ZeroSpreadedTermStructure(
                base_yield_curve, ql.QuoteHandle(ql.SimpleQuote(spread))
            )
            spread_curve_handle = ql.YieldTermStructureHandle(spread_curve)
            
            # Create Hull-White model with spread curve
            hw_with_spread = ql.HullWhite(spread_curve_handle, a=hw_mean_reversion, sigma=hw_volatility)
            engine_with_spread = ql.TreeCallableFixedRateBondEngine(hw_with_spread, tree_steps)
            
            # Price the bond
            callable_bond.setPricingEngine(engine_with_spread)
            return callable_bond.cleanPrice()
            
        except Exception as e:
            print(f"Pricing error with spread {spread*10000:.1f}bp: {e}")
            return target_price + 1000  # Return a bad price to continue search
    
    # Use bisection method to find OAS
    def solve_for_oas():
        low_spread = -0.02   # -100bp
        high_spread = 0.02   # +100bp
        tolerance = 1e-6
        max_iterations = 500
        
        for iteration in range(max_iterations):
            mid_spread = (low_spread + high_spread) / 2
            mid_price = price_with_spread(mid_spread)
            
            price_diff = mid_price - target_price
            
            if abs(price_diff) < tolerance:
                return mid_spread
            
            if price_diff > 0:  # Model price too high, need higher spread
                low_spread = mid_spread
            else:  # Model price too low, need lower spread
                high_spread = mid_spread
            
            if iteration % 20 == 0:
                print(f"  Iteration {iteration:2d}: Spread = {mid_spread*10000:+6.1f}bp, Price = {mid_price:.4f}, Target = {target_price:.4f}")
        
        return (low_spread + high_spread) / 2
    
    try:
        oas = solve_for_oas()
        
        # Verify the result
        verification_price = price_with_spread(oas)
        
        print(f"\nOAS RESULTS:")
        print(f"Option-Adjusted Spread: {oas*10000:+.1f} basis points")
        print(f"Verification - Model price with OAS: ${verification_price:.4f}")
        print(f"Target price: ${target_price:.4f}")
        print(f"Pricing error: ${abs(verification_price - target_price):.6f}")
        
        return oas
        
    except Exception as e:
        print(f"OAS calculation failed: {e}")
        return None

# Calculate OAS
# oas = calculate_oas(callable_bond_price, base_yield_curve, hw_model, tree_steps)
oas = calculate_oas(price_today, base_yield_curve, hw_model, tree_steps)

print()

# ============================================================================
# 9. COMPREHENSIVE RESULTS SUMMARY
# ============================================================================

print("="*80)
print("COMPREHENSIVE ANALYSIS RESULTS")
print("="*80)

# Create summary table
summary_data = {
    'Metric': [
        'Option-Free Bond Price',
        'Callable Bond Price', 
        'Embedded Option Value',
        'Yield-to-Maturity',
        'Yield-to-Worst',
        'Yield Pickup (Callable)',
        'Option-Adjusted Spread',
        'Expected Bond Life',
        'Call Probability'
    ],
    'Value': [
        f"${option_free_price:.2f}",
        f"${callable_bond_price:.2f}",
        f"${embedded_option_value:.2f}",
        f"{scenarios[0]['Yield']*100:.2f}%" if scenarios else "N/A",
        f"{ytw*100:.2f}%" if ytw else "N/A", 
        f"{(scenarios[0]['Yield'] - option_free_bond.bondYield(option_free_price, ql.ActualActual(ql.ActualActual.Bond), ql.Compounded, ql.Annual))*10000:.0f} bps" if scenarios else "N/A",
        f"{oas*10000:+.0f} bps" if oas else "N/A",
        f"{worst_scenario['Years']:.1f} years" if worst_scenario else "N/A",
        "High" if worst_scenario and worst_scenario['Type'] == 'Call' else "Low"
    ],
    'Interpretation': [
        'Theoretical value without call option',
        'Market value with embedded call option',
        'Value of call option to issuer',
        'Yield if bond runs to maturity',
        'Worst-case yield across all scenarios',
        'Extra yield for accepting call risk',
        'Credit spread over risk-free curve',
        'Expected time until call/maturity',
        'Likelihood of early call'
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print()

# ============================================================================
# 10. INVESTMENT ANALYSIS AND RECOMMENDATIONS
# ============================================================================

print("INVESTMENT ANALYSIS:")
print("-" * 30)

# Risk assessment
if embedded_option_value > 5:
    risk_level = "HIGH"
elif embedded_option_value > 2:
    risk_level = "MODERATE" 
else:
    risk_level = "LOW"

print(f"Call Risk Level: {risk_level}")
print(f"  - Embedded option value: ${embedded_option_value:.2f}")

if worst_scenario and worst_scenario['Type'] == 'Call':
    print(f"  - Likely to be called in {worst_scenario['Years']:.1f} years")
    print(f"  - Reinvestment risk at call date")
else:
    print(f"  - Likely to run to full maturity")
    print(f"  - Limited reinvestment risk")

# Yield analysis
if ytw and scenarios:
    ytm = scenarios[0]['Yield']
    if ytw < ytm:
        print(f"\nYield Impact: NEGATIVE")
        print(f"  - YTW ({ytw*100:.2f}%) < YTM ({ytm*100:.2f}%)")
        print(f"  - Early call hurts investor returns")
    else:
        print(f"\nYield Impact: NEUTRAL")
        print(f"  - YTW ≈ YTM")
        print(f"  - Limited early call impact")

# OAS interpretation
if oas:
    if oas > 0.005:  # > 50bp
        print(f"\nCredit Assessment: ATTRACTIVE")
        print(f"  - OAS of {oas*10000:+.0f}bp indicates good value")
    elif oas > 0:
        print(f"\nCredit Assessment: FAIR")
        print(f"  - OAS of {oas*10000:+.0f}bp shows reasonable pricing")
    else:
        print(f"\nCredit Assessment: EXPENSIVE")
        print(f"  - Negative OAS of {oas*10000:+.0f}bp suggests overpricing")

print()

# ============================================================================
# 11. RESET PRICING ENGINE
# ============================================================================

# Reset to base pricing engine for any further calculations
callable_bond.setPricingEngine(pricing_engine)

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()
print("Key Findings:")
print(f"• Callable bond trades at ${callable_bond_price:.2f} vs ${option_free_price:.2f} option-free")
print(f"• Embedded call option worth ${embedded_option_value:.2f} to issuer")
print(f"• Yield-to-Worst: {ytw*100:.2f}%" if ytw else "• YTW calculation failed")
print(f"• Option-Adjusted Spread: {oas*10000:+.0f} basis points" if oas else "• OAS calculation failed")
print(f"• Expected to be {'called early' if worst_scenario and worst_scenario['Type'] == 'Call' else 'held to maturity'}")
print()
print("This analysis provides comprehensive callable bond risk assessment")
print("suitable for portfolio management and investment decision-making.")

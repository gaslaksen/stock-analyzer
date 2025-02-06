import streamlit as st
import matplotlib.pyplot as plt

def calculate_fees(aum, fee_percentage):
    current_fees = (aum * fee_percentage) / 100
    flat_fee = 10000.0  # Fixed flat fee
    savings = current_fees - flat_fee
    return current_fees, flat_fee, savings

# Streamlit UI
st.title("Financial Advisor Fee Comparison")
st.write("Compare your current advisory fees with a flat-fee model over time.")

# User Inputs
aum = st.number_input("Enter your total assets under management (AUM) in USD:", min_value=0.0, step=1000.0)
fee_percentage = st.number_input("Enter your current advisory fee percentage (%):", min_value=0.0, step=0.01)

# Calculate and Display Fees
if st.button("Calculate Savings"):
    if aum > 0 and fee_percentage > 0:
        current_fees, flat_fee, savings = calculate_fees(aum, fee_percentage)
        
        st.write(f"### Your Current Annual Fees: ${current_fees:,.2f}")
        st.write(f"### Gugle Flat Fee: ${flat_fee:,.2f}")
        if savings > 0:
            st.write(f"### Potential Annual Savings: ${savings:,.2f}")
        else:
            st.write("You are already paying less than the flat-fee model.")
        
        # Line Graph Comparison Over 10 Years
        years = list(range(1, 11))
        current_fees_over_time = [current_fees * year for year in years]
        flat_fee_over_time = [flat_fee * year for year in years]
        
        plt.figure(figsize=(8, 5))
        plt.plot(years, current_fees_over_time, label="Current Fee", marker='o')
        plt.plot(years, flat_fee_over_time, label="Flat Fee", marker='s', linestyle='dashed')
        plt.xlabel("Years")
        plt.ylabel("Total Fees Paid (USD)")
        plt.title("Fee Comparison Over 10 Years")
        plt.legend()
        plt.grid()
        
        st.pyplot(plt)
    else:
        st.warning("Please enter valid values for AUM and advisory fee percentage.")

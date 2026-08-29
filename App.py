import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Invoice Intelligence System",
    page_icon="🧾",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_models():

    # Invoice Classification Model
    invoice_model = joblib.load(
        os.path.join(BASE_DIR, "invoice_model.pkl")
    )

    invoice_scaler = joblib.load(
        os.path.join(BASE_DIR, "invoice_scaler.pkl")
    )

    invoice_columns = joblib.load(
        os.path.join(BASE_DIR, "invoice_columns.pkl")
    )

    # Freight Regression Model
    freight_model = joblib.load(
        os.path.join(BASE_DIR, "freight_model.pkl")
    )

    freight_columns = joblib.load(
        os.path.join(BASE_DIR, "freight_columns.pkl")
    )

    return (
        invoice_model,
        invoice_scaler,
        invoice_columns,
        freight_model,
        freight_columns
    )


try:

    (
        invoice_model,
        invoice_scaler,
        invoice_columns,
        freight_model,
        freight_columns
    ) = load_models()

except Exception as e:

    st.error(f"Error loading model files: {e}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧾 Invoice Intelligence")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Invoice Risk Detection",
        "Freight Cost Prediction"
    ]
)


st.sidebar.divider()

st.sidebar.write(
    """
    **Invoice Intelligence System**

    Machine Learning based system for:

    • Invoice risk detection  
    • Manual review prioritization  
    • Freight cost estimation
    """
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("🧾 Invoice Intelligence System")

    st.write(
        """
        This system uses machine learning to analyze vendor invoices
        and identify potential financial and operational risks.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🚨 Invoice Risk Detection")

        st.write(
            """
            Analyzes invoice and purchase information and predicts
            whether an invoice should be sent for manual review.
            """
        )

        st.info(
            "Model: Random Forest Classifier"
        )

    with col2:

        st.subheader("🚚 Freight Cost Prediction")

        st.write(
            """
            Predicts the expected freight cost using quantity
            and invoice dollar amount.
            """
        )

        st.info(
            "Model: Random Forest Regressor"
        )

    st.divider()

    st.subheader("System Architecture")

    st.code(
        """
Invoice Intelligence System

        |
        +------------------------+
        |                        |
        v                        v

Invoice Risk Detection      Freight Prediction

RandomForestClassifier      RandomForestRegressor

        |                        |
        v                        v

Normal / Review          Expected Freight Cost
        """
    )


# ============================================================
# INVOICE RISK DETECTION
# ============================================================

elif page == "Invoice Risk Detection":

    st.title("🚨 Invoice Risk Detection")

    st.write(
        """
        Enter invoice and purchase information to determine
        whether the invoice should be manually reviewed.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        total_invoice_quantity = st.number_input(
            "Total Invoice Quantity",
            min_value=0.0,
            value=100.0
        )

    with col2:

        total_invoice_dollars = st.number_input(
            "Total Invoice Dollars ($)",
            min_value=0.0,
            value=1000.0
        )

    with col3:

        freight = st.number_input(
            "Freight Cost ($)",
            min_value=0.0,
            value=50.0
        )


    col4, col5 = st.columns(2)

    with col4:

        days_to_pay = st.number_input(
            "Days to Pay",
            min_value=0.0,
            value=10.0
        )

    with col5:

        avg_receiving_delay = st.number_input(
            "Average Receiving Delay",
            min_value=0.0,
            value=5.0
        )


    col6, col7 = st.columns(2)

    with col6:

        total_item_quantity = st.number_input(
            "Purchase Record Quantity",
            min_value=0.0,
            value=100.0
        )

    with col7:

        total_item_dollars = st.number_input(
            "Purchase Record Dollars ($)",
            min_value=0.0,
            value=1000.0
        )


    # ========================================================
    # BASIC COMPARISON
    # ========================================================

    st.divider()

    st.subheader("📊 Invoice Comparison")

    amount_difference = abs(
        total_invoice_dollars - total_item_dollars
    )

    quantity_difference = abs(
        total_invoice_quantity - total_item_quantity
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Amount Difference",
        f"${amount_difference:,.2f}"
    )

    c2.metric(
        "Quantity Difference",
        f"{quantity_difference:,.0f}"
    )

    c3.metric(
        "Receiving Delay",
        f"{avg_receiving_delay:.1f} Days"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    st.divider()

    if st.button(
        "🔍 Analyze Invoice",
        type="primary",
        use_container_width=True
    ):

        # -----------------------------------------------
        # Create dataframe
        # -----------------------------------------------

        input_data = pd.DataFrame(
            [[
                total_invoice_quantity,
                total_invoice_dollars,
                freight,
                days_to_pay,
                total_item_quantity,
                total_item_dollars,
                avg_receiving_delay
            ]],
            columns=[
                "total_invoice_quantity",
                "total_invoice_dollars",
                "Freight",
                "days_to_pay",
                "total_item_quantity",
                "total_item_dollars",
                "avg_receiving_delay"
            ]
        )


        # -----------------------------------------------
        # Correct training column order
        # -----------------------------------------------

        try:

            input_data = input_data[invoice_columns]

        except Exception as e:

            st.error(
                "Input columns do not match model training columns."
            )

            st.write("Expected:")
            st.write(invoice_columns)

            st.write("Received:")
            st.write(input_data.columns.tolist())

            st.stop()


        # -----------------------------------------------
        # Scale
        # -----------------------------------------------

        try:

            input_scaled = invoice_scaler.transform(
                input_data
            )

        except Exception as e:

            st.error(
                f"Scaling failed: {e}"
            )

            st.stop()


        # -----------------------------------------------
        # Predict
        # -----------------------------------------------

        try:

            prediction = invoice_model.predict(
                input_scaled
            )[0]

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

            st.stop()


        # ====================================================
        # RESULT
        # ====================================================

        st.subheader("🤖 Prediction Result")

        if prediction == 1:

            st.error(
                "🚨 FLAGGED INVOICE — Manual Review Recommended"
            )

        else:

            st.success(
                "✅ NORMAL INVOICE — No Immediate Review Required"
            )


        # ====================================================
        # PROBABILITY
        # ====================================================

        if hasattr(invoice_model, "predict_proba"):

            probabilities = invoice_model.predict_proba(
                input_scaled
            )[0]

            classes = list(invoice_model.classes_)

            if 1 in classes:

                risk_index = classes.index(1)

                risk_probability = probabilities[
                    risk_index
                ]

                st.subheader("Risk Probability")

                st.progress(
                    float(risk_probability)
                )

                st.write(
                    f"""
                    Probability of requiring manual review:
                    **{risk_probability * 100:.2f}%**
                    """
                )


        # ====================================================
        # RISK REASONS
        # ====================================================

        st.divider()

        st.subheader("🔎 Risk Indicators")

        reasons = []

        if amount_difference > 5:

            reasons.append(
                f"Invoice amount differs from purchase "
                f"records by ${amount_difference:,.2f}."
            )

        if avg_receiving_delay > 10:

            reasons.append(
                f"Average receiving delay is "
                f"{avg_receiving_delay:.1f} days."
            )

        if reasons:

            for reason in reasons:

                st.warning("⚠️ " + reason)

        else:

            st.success(
                "No major rule-based risk indicators detected."
            )


        # ====================================================
        # INPUT DATA
        # ====================================================

        with st.expander(
            "View Model Input"
        ):

            st.dataframe(
                input_data,
                use_container_width=True
            )


# ============================================================
# FREIGHT COST PREDICTION
# ============================================================

elif page == "Freight Cost Prediction":

    st.title("🚚 Freight Cost Prediction")

    st.write(
        """
        Enter the purchase quantity and dollar amount to
        estimate the expected freight cost.
        """
    )

    st.divider()


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        quantity = st.number_input(
            "Quantity",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

    with col2:

        dollars = st.number_input(
            "Dollar Amount ($)",
            min_value=0.0,
            value=1000.0,
            step=10.0
        )


    st.divider()


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🚚 Predict Freight Cost",
        type="primary",
        use_container_width=True
    ):

        # Create dataframe
        freight_input = pd.DataFrame(
            [[
                quantity,
                dollars
            ]],
            columns=[
                "Quantity",
                "Dollars"
            ]
        )


        # ----------------------------------------------------
        # Make same column order as training
        # ----------------------------------------------------

        try:

            freight_input = freight_input[
                freight_columns
            ]

        except Exception:

            st.error(
                "Freight input columns don't match "
                "the model training columns."
            )

            st.write(
                "Expected:",
                freight_columns
            )

            st.stop()


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        try:

            predicted_freight = freight_model.predict(
                freight_input
            )[0]

        except Exception as e:

            st.error(
                f"Freight prediction failed: {e}"
            )

            st.stop()


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader(
            "📊 Estimated Freight Cost"
        )

        st.metric(
            "Predicted Freight",
            f"${predicted_freight:,.2f}"
        )

        st.success(
            f"""
            Based on a quantity of **{quantity:,.0f}**
            and purchase value of **${dollars:,.2f}**,
            the model estimates a freight cost of
            **${predicted_freight:,.2f}**.
            """
        )


        # ----------------------------------------------------
        # Freight percentage
        # ----------------------------------------------------

        if dollars > 0:

            freight_percentage = (
                predicted_freight / dollars
            ) * 100

            st.metric(
                "Freight % of Purchase Value",
                f"{freight_percentage:.2f}%"
            )


        with st.expander(
            "View Model Input"
        ):

            st.dataframe(
                freight_input,
                use_container_width=True
            )
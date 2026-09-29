from pathlib import Path
import json

import streamlit as st

from src.inference import (
    load_model,
    load_routing_config,
    predict_ticket,
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Customer Ticket Routing System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_system_metrics.json"
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 750;
        margin-bottom: 0.4rem;
        line-height: 1.15;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    .decision-auto {
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.10);
        border: 1px solid rgba(34, 197, 94, 0.35);
        margin-top: 1rem;
        margin-bottom: 1.4rem;
    }

    .decision-review {
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        background: rgba(245, 158, 11, 0.10);
        border: 1px solid rgba(245, 158, 11, 0.40);
        margin-top: 1rem;
        margin-bottom: 1.4rem;
    }

    .decision-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .decision-text {
        color: #6b7280;
        margin: 0;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Load resources
# ============================================================

@st.cache_resource
def load_resources():

    config = load_routing_config()

    tokenizer, model, device = (
        load_model()
    )

    return (
        config,
        tokenizer,
        model,
        device,
    )


@st.cache_data
def load_system_metrics():

    if METRICS_PATH.exists():

        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    return {}


config, tokenizer, model, device = (
    load_resources()
)

system_metrics = (
    load_system_metrics()
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header(
        "System Information"
    )

    st.write(
        "**Model**"
    )

    st.write(
        "DistilBERT"
    )

    st.write(
        "**Intent Classes**"
    )

    st.write(
        "77"
    )

    st.write(
        "**Routing Threshold**"
    )

    st.write(
        f"{config['confidence_threshold']:.2%}"
    )

    st.divider()

    st.subheader(
        "Final Test Results"
    )

    st.metric(
        "Raw Accuracy",
        "92.40%",
    )

    st.metric(
        "Macro-F1",
        "92.58%",
    )

    st.metric(
        "Auto-routing Coverage",
        "94.34%",
    )

    st.metric(
        "Auto-routed Accuracy",
        "95.41%",
    )

    st.metric(
        "Manual Review Rate",
        "5.66%",
    )

    st.divider()

    st.caption(
        "The confidence threshold was selected "
        "using the validation set with a target "
        "auto-routing accuracy of at least 95%."
    )


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="main-title">
        Intelligent Customer Ticket Routing System
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        77-class banking intent classification with
        confidence-aware automatic routing and manual review.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# System overview
# ============================================================

overview_col1, overview_col2, overview_col3 = (
    st.columns(3)
)

with overview_col1:

    st.metric(
        "Intent Classes",
        "77",
    )

with overview_col2:

    st.metric(
        "Test Macro-F1",
        "92.58%",
    )

with overview_col3:

    st.metric(
        "Auto-route Accuracy",
        "95.41%",
    )


st.divider()


# ============================================================
# Example tickets
# ============================================================

EXAMPLE_TICKETS = {

    "Custom ticket":
        "",

    "Card not working":
        "My card has stopped working.",

    "Pending transfer":
        "When will my transfer go through?",

    "Recipient has not received transfer":
        "The recipient still has not received my money.",

    "Unrecognised cash withdrawal":
        "I do not recognise this cash withdrawal.",

    "Change personal details":
        "I need to change my phone number.",

    "Identity verification":
        "What documents do I need to verify my identity?",

    "Ambiguous transfer issue":
        "My transfer has a problem.",
}


st.markdown(
    '<div class="section-title">Analyse a customer ticket</div>',
    unsafe_allow_html=True,
)

example_name = st.selectbox(
    "Example Ticket",
    options=list(
        EXAMPLE_TICKETS.keys()
    ),
)

example_text = (
    EXAMPLE_TICKETS[
        example_name
    ]
)


ticket_text = st.text_area(
    "Customer Ticket",
    value=example_text,
    placeholder=(
        "Enter a banking customer support request..."
    ),
    height=150,
)


analyse_button = st.button(
    "Analyse Ticket",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Prediction
# ============================================================

if analyse_button:

    if not ticket_text.strip():

        st.warning(
            "Please enter a customer ticket before running the model."
        )

    else:

        try:

            with st.spinner(
                "Analysing customer request..."
            ):

                result = predict_ticket(
                    text=ticket_text,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    config=config,
                )


            st.divider()


            # ====================================================
            # Decision banner
            # ====================================================

            if result["manual_review"]:

                st.markdown(
                    """
                    <div class="decision-review">
                        <div class="decision-title">
                            Manual Review Required
                        </div>
                        <p class="decision-text">
                            Model confidence is below the
                            automatic-routing threshold.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    f"""
                    <div class="decision-auto">
                        <div class="decision-title">
                            Ticket Automatically Routed
                        </div>
                        <p class="decision-text">
                            Assigned to:
                            <strong>{result["department"]}</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            # ====================================================
            # Main prediction cards
            # ====================================================

            st.markdown(
                '<div class="section-title">Routing Result</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Predicted Intent",
                    result[
                        "predicted_intent"
                    ],
                )

            with col2:

                st.metric(
                    "Department",
                    result[
                        "department"
                    ],
                )

            with col3:

                st.metric(
                    "Confidence",
                    f"{result['confidence']:.2%}",
                )

            with col4:

                st.metric(
                    "Decision",
                    (
                        "Manual Review"
                        if result["manual_review"]
                        else "Auto Route"
                    ),
                )


            # ====================================================
            # Confidence
            # ====================================================

            st.markdown(
                '<div class="section-title">Confidence Analysis</div>',
                unsafe_allow_html=True,
            )

            confidence_col, threshold_col = (
                st.columns(2)
            )

            with confidence_col:

                st.write(
                    "**Model confidence**"
                )

                st.progress(
                    min(
                        float(
                            result[
                                "confidence"
                            ]
                        ),
                        1.0,
                    )
                )

                st.write(
                    f"{result['confidence']:.2%}"
                )

            with threshold_col:

                st.write(
                    "**Automatic-routing threshold**"
                )

                st.progress(
                    min(
                        float(
                            result[
                                "confidence_threshold"
                            ]
                        ),
                        1.0,
                    )
                )

                st.write(
                    f"{result['confidence_threshold']:.2%}"
                )


            # ====================================================
            # Alternative prediction
            # ====================================================

            st.markdown(
                '<div class="section-title">Alternative Prediction</div>',
                unsafe_allow_html=True,
            )

            alternative_col1, alternative_col2, alternative_col3 = (
                st.columns(3)
            )

            with alternative_col1:

                st.metric(
                    "Second-best Intent",
                    result[
                        "alternative_intent"
                    ],
                )

            with alternative_col2:

                st.metric(
                    "Alternative Confidence",
                    f"{result['alternative_confidence']:.2%}",
                )

            with alternative_col3:

                st.metric(
                    "Confidence Margin",
                    f"{result['confidence_margin']:.2%}",
                )


            # ====================================================
            # Explanation
            # ====================================================

            if result["manual_review"]:

                st.info(
                    "The model produced a prediction, "
                    "but its confidence was below the "
                    "validated routing threshold. "
                    "The ticket should therefore be reviewed "
                    "by a human agent before routing."
                )

            else:

                st.info(
                    "The model confidence exceeded the "
                    "validated routing threshold, so the "
                    "ticket can be automatically routed "
                    "to the predicted department."
                )


            # ====================================================
            # Technical information
            # ====================================================

            with st.expander(
                "Technical Details"
            ):

                st.write(
                    "**Input text**"
                )

                st.code(
                    result["text"]
                )

                st.write(
                    "**Raw inference output**"
                )

                st.json(
                    result
                )


        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# Architecture explanation
# ============================================================

st.divider()

with st.expander(
    "How the routing system works"
):

    st.markdown(
        """
        **1. Customer ticket**

        The user enters a banking support request.

        **2. DistilBERT intent classification**

        The fine-tuned Transformer predicts one of
        77 BANKING77 intent classes.

        **3. Confidence estimation**

        Softmax scores are used to obtain the
        top prediction and an alternative prediction.

        **4. Confidence gate**

        Predictions with confidence of at least
        **75.29%** are automatically routed.
        Lower-confidence predictions are sent for
        manual review.

        **5. Department routing**

        The predicted fine-grained intent is mapped
        to the appropriate business department.
        """
    )
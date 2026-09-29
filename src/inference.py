from pathlib import Path
import json

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "distilbert_final"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "routing_config.json"
)


# ============================================================
# Load routing configuration
# ============================================================

def load_routing_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(file)

    return config


# ============================================================
# Load trained model
# ============================================================

def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_DIR
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_DIR
        )
    )

    model = model.to(device)

    model.eval()

    return (
        tokenizer,
        model,
        device,
    )


# ============================================================
# Ticket prediction
# ============================================================

def predict_ticket(
    text,
    tokenizer,
    model,
    device,
    config,
):

    if not isinstance(text, str):

        raise TypeError(
            "Ticket text must be a string."
        )

    text = text.strip()

    if not text:

        raise ValueError(
            "Ticket text cannot be empty."
        )


    max_length = int(
        config["max_length"]
    )

    confidence_threshold = float(
        config["confidence_threshold"]
    )

    department_mapping = (
        config["department_mapping"]
    )


    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    encoded = {
        key: value.to(device)
        for key, value
        in encoded.items()
    }


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **encoded
        )

        probabilities = F.softmax(
            outputs.logits,
            dim=-1,
        )


    # --------------------------------------------------------
    # Top-2 predictions
    # --------------------------------------------------------

    top2_probs, top2_ids = torch.topk(
        probabilities,
        k=2,
        dim=-1,
    )


    top1_id = int(
        top2_ids[0, 0].item()
    )

    top2_id = int(
        top2_ids[0, 1].item()
    )


    confidence = float(
        top2_probs[0, 0].item()
    )

    alternative_confidence = float(
        top2_probs[0, 1].item()
    )


    # --------------------------------------------------------
    # Convert IDs back to labels
    # --------------------------------------------------------

    predicted_intent = (
        model.config.id2label[
            top1_id
        ]
    )

    alternative_intent = (
        model.config.id2label[
            top2_id
        ]
    )


    # --------------------------------------------------------
    # Business routing
    # --------------------------------------------------------

    department = (
        department_mapping[
            predicted_intent
        ]
    )

    manual_review = (
        confidence
        <
        confidence_threshold
    )

    routing_decision = (
        "MANUAL_REVIEW"
        if manual_review
        else "AUTO_ROUTE"
    )


    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        "text":
            text,

        "predicted_intent":
            predicted_intent,

        "department":
            department,

        "confidence":
            confidence,

        "alternative_intent":
            alternative_intent,

        "alternative_confidence":
            alternative_confidence,

        "confidence_margin":
            (
                confidence
                - alternative_confidence
            ),

        "confidence_threshold":
            confidence_threshold,

        "routing_decision":
            routing_decision,

        "manual_review":
            manual_review,
    }
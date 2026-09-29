# Intelligent Customer Ticket Classification and Routing System

An end-to-end NLP project for classifying banking customer-support tickets into 77 fine-grained intents and routing them to service departments with confidence-aware manual review.

The project compares traditional machine-learning baselines, neural text classifiers, and a pretrained Transformer. The final system uses a fine-tuned DistilBERT model together with a validation-selected confidence threshold to decide whether a ticket should be routed automatically or sent for human review.

## System Overview

```text
Customer Ticket
      ↓
Fine-tuned DistilBERT
      ↓
77-class Intent Prediction
      ↓
Softmax Confidence
      ↓
Confidence ≥ 0.7529?
     /              \
   Yes               No
    ↓                 ↓
AUTO_ROUTE       MANUAL_REVIEW
    ↓
Department Mapping
```

The final routing layer maps fine-grained BANKING77 intents to operational groups such as:

- Card Services
- Card Payments
- Transfers
- Top Up & Deposits
- Cash & ATM
- Currency & Exchange
- Account & Identity
- Digital Wallet & General Banking

## Dataset

This project uses **BANKING77**, a banking customer-support intent classification dataset with 77 fine-grained intent categories.

After preprocessing and deduplication, the project uses a fixed stratified split:

| Split | Samples |
|---|---:|
| Train | 9,149 |
| Validation | 1,961 |
| Test | 1,961 |

All 77 intent categories are represented across the frozen splits.

### Experimental protocol

- Training-only statistics and vocabularies are fitted on the training split.
- Model development and threshold selection use the validation split.
- The test split is used for final evaluation.
- All model comparisons use the same frozen data split.

> **Important:** this project uses a custom 70/15/15 stratified split after preprocessing rather than the official BANKING77 benchmark split. Therefore, the reported metrics should not be treated as directly comparable to papers using the official split.

## Models

The project evaluates several approaches:

### Traditional machine learning

- TF-IDF + Logistic Regression
- TF-IDF + Multinomial Naive Bayes
- TF-IDF + Linear SVM

### Neural baselines

- TextCNN
- BiLSTM

### Transformer

- Fine-tuned DistilBERT

## Model Comparison

| Model | Validation Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|
| Multinomial Naive Bayes | 82.26% | — | — |
| Logistic Regression | 87.02% | — | — |
| BiLSTM | 86.72% | 85.87% | 86.14% |
| Linear SVM | 88.76% | 88.53% | — |
| TextCNN | 89.26% | 88.22% | 88.43% |
| **DistilBERT** | **91.57%** | **92.40%** | **92.58%** |

The experiments show that model complexity alone does not guarantee better performance. BiLSTM underperformed both the Linear SVM and TextCNN baselines, while pretrained contextual representations from DistilBERT produced the strongest improvement.

## Final DistilBERT Results

The best DistilBERT checkpoint was selected using validation Macro-F1.

### Validation

- Accuracy: **91.59%**
- Macro-F1: **91.57%**

### Test

- Accuracy: **92.40%**
- Macro Precision: **92.95%**
- Macro Recall: **92.62%**
- Macro-F1: **92.58%**

Compared with TextCNN, DistilBERT improved test Macro-F1 by approximately **4.15 percentage points**.

## Confidence-aware Routing

A practical routing system should not automatically execute every prediction. The final system therefore uses the maximum DistilBERT softmax score as a confidence signal.

The decision rule is:

```text
confidence >= threshold
→ AUTO_ROUTE

confidence < threshold
→ MANUAL_REVIEW
```

The threshold was selected **only on the validation set**. The optimisation objective was:

```text
Auto-routed Accuracy >= 95%
while maximising Auto-routing Coverage
```

Selected threshold:

```text
0.7528857589
```

### Validation routing performance

| Metric | Result |
|---|---:|
| Auto-routing Coverage | 93.32% |
| Manual Review Rate | 6.68% |
| Auto-routed Accuracy | 95.03% |
| Error Capture Rate | 44.85% |

### Final test routing performance

| Metric | Result |
|---|---:|
| Raw Accuracy | **92.40%** |
| Auto-routing Coverage | **94.34%** |
| Manual Review Rate | **5.66%** |
| Auto-routed Accuracy | **95.41%** |
| Error Capture Rate | **42.95%** |

The confidence gate raises the reliability of automatically executed routing decisions while sending only a small fraction of tickets to manual review.

> Softmax confidence is treated as a model confidence score, not as a perfectly calibrated probability of correctness.

## Example Predictions

### Automatic routing

```text
Customer ticket:
"The beneficiary is not allowed."

Predicted intent:
beneficiary_not_allowed

Department:
Transfers

Confidence:
92.11%

Decision:
AUTO_ROUTE
```

### Manual review

```text
Customer ticket:
"help me with my transfer"

Predicted intent:
failed_transfer

Confidence:
75.00%

Routing threshold:
75.29%

Decision:
MANUAL_REVIEW
```

The second example is intentionally underspecified: the text does not clearly indicate whether the transfer is failed, pending, delayed, or not received by the recipient. The system therefore abstains from automatic routing.

## Error Analysis

The project includes sample-level comparisons across Linear SVM, TextCNN, and DistilBERT.

DistilBERT:

- corrected **111** SVM errors;
- introduced **35** new errors relative to SVM;
- produced a net gain of **76** correct predictions;

and:

- corrected **122** TextCNN errors;
- introduced **40** new errors relative to TextCNN;
- produced a net gain of **82** correct predictions.

Across SVM, TextCNN, and DistilBERT:

- **1,627** test samples were correctly classified by all three models;
- **95** samples were misclassified by all three models;
- **48** samples were correctly classified only by DistilBERT.

Qualitative analysis suggests that DistilBERT is particularly useful for:

- paraphrased and non-standard expressions;
- contextual disambiguation;
- object-action-state relationships;
- fine-grained intent distinctions.

Remaining shared errors are concentrated around:

- pending vs failed vs declined vs reverted transaction states;
- fine-grained transfer states;
- payment-channel distinctions;
- identity-verification intents;
- very short or underspecified customer messages.

These cases suggest that part of the remaining error is caused by insufficient textual context and fine-grained label ambiguity rather than model capacity alone.

## Demo

A Streamlit interface is included in `app.py`.

The interface displays:

- predicted intent;
- routed department;
- model confidence;
- alternative intent;
- confidence margin;
- automatic-routing threshold;
- `AUTO_ROUTE` or `MANUAL_REVIEW` decision.

Run it with:

```bash
python -m streamlit run app.py
```

## Project Structure

```text
customer-ticket-routing/
│
├── app.py
├── config/
│   └── routing_config.json
│
├── data/
│   └── processed/
│       ├── train_77.csv
│       ├── validation_77.csv
│       ├── test_77.csv
│       ├── label_to_id.json
│       └── id_to_label.json
│
├── models/
│   ├── distilbert_baseline/
│   ├── distilbert_final/
│   └── ml_baseline/
│
├── notebook/
│   ├── mlbaseline ver.1.ipynb
│   ├── CNN_deep_learning_baseline.ipynb
│   ├── bilstm_baseline.ipynb
│   ├── transformer_baseline.ipynb
│   ├── model_comparison.ipynb
│   └── final_routing_system.ipynb
│
├── reports/
├── results/
│   ├── final_system_metrics.json
│   ├── final_routing_test_results.csv
│   ├── distilbert_test_predictions.csv
│   ├── distilbert_test_errors.csv
│   ├── distilbert_per_class_metrics.csv
│   └── distilbert_confusion_pairs.csv
│
├── src/
│   └── inference.py
│
├── .gitattributes
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/shdbdud/customer-ticket-routing.git
cd customer-ticket-routing
```

### 2. Download Git LFS model artifacts

The repository stores large model artifacts using Git LFS.

```bash
git lfs install
git lfs pull
```

### 3. Create an environment

For example:

```bash
conda create -n ticket-routing python=3.11
conda activate ticket-routing
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit demo

```bash
python -m streamlit run app.py
```

## Programmatic Inference

The inference pipeline is implemented in `src/inference.py`.

```python
from src.inference import (
    load_model,
    load_routing_config,
    predict_ticket,
)

config = load_routing_config()
tokenizer, model, device = load_model()

result = predict_ticket(
    text="My card has stopped working.",
    tokenizer=tokenizer,
    model=model,
    device=device,
    config=config,
)

print(result)
```

The returned object includes the predicted intent, department, confidence score, alternative intent, confidence margin, and routing decision.

## Git LFS

Large model artifacts are tracked with Git LFS:

```text
*.safetensors
*.pt
*.pth
*.bin
*.joblib
```

After cloning, run:

```bash
git lfs pull
```

to download the full model files.

## Main Technologies

- Python
- PyTorch
- Hugging Face Transformers
- scikit-learn
- pandas
- NumPy
- Streamlit
- Git LFS

## Future Work

Potential extensions include:

- temperature scaling or other probability-calibration methods;
- class-specific routing thresholds;
- hierarchical intent classification;
- transaction metadata such as payment channel and transaction status;
- retrieval of similar historical tickets;
- human-feedback collection and active learning;
- REST API deployment;
- monitoring confidence and performance drift after deployment.

A particularly useful extension would be moving from text-only classification to context-aware routing by incorporating transaction status, channel, sender/recipient role, and account metadata.

## Disclaimer

The department-routing layer is an engineering design for this project and does not represent the internal organisational structure of any real financial institution.

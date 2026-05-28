# Mastercard Data Quest: Hidden Entrepreneur Detection

This project aims to identify "hidden entrepreneurs" — consumers who use personal bank cards for business activities — using synthetic Mastercard transaction data.

## Project Overview
Detecting hidden business activity is crucial for banks to optimize products, manage risks, and offer tailored business services. This solution uses a machine learning pipeline to rank 80,000 consumer cards based on their "business-likeness" compared to a reference set of 25,000 verified business cards.

## Key Features
- **Data Harmonization**: Processes 12M+ transactions across business and consumer segments.
- **Advanced Feature Engineering**: 
    - **MCC Analysis**: Concentration in B2B/Business-heavy categories.
    - **Intensity**: Transaction frequency and volume patterns.
    - **Time-of-Day**: Activity during business hours vs. leisure time.
    - **Channel/Geography**: Online vs. offline and domestic vs. foreign transaction ratios.
- **Explainable AI**: SHAP-based interpretation of model decisions.
- **Ranking System**: Outputs a continuous score for all consumer cards to support targeted marketing or audit campaigns.

## Results
- **Model Performance**: Achieved **0.999+ AUC-ROC** on the reference train/test split.
- **Segment Profile**: The top-ranked "hidden entrepreneurs" exhibit profiles nearly identical to verified businesses:
    - **Average Transaction Amount**: ~232k KZT (Top-50 candidates) vs ~167k KZT (Verified Business).
    - **Business MCC Share**: ~65% (Top-50 candidates) vs ~72% (Verified Business).

## Project Structure
- `solution.py`: The complete end-to-end ML pipeline (Data loading, EDA, Training, Scoring).
- `requirements.txt`: Python dependencies.
- `outputs/`: 
    - `final_submission.csv`: Full ranking of 80,000 cards.
    - `top_50_candidates_detailed.csv`: Detailed audit of the most suspicious cards.
    - `*.png`: Visualizations of metrics, profiles, and feature importance.

## Installation and Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the pipeline**:
   The script assumes data is located in `C:\Users\admin\Downloads\MDQ`.
   ```bash
   python solution.py
   ```

## Methodology
The solution treats the problem as a classification task where known Business cards are the Positive class and a representative sample of Consumer cards is used as the Negative class proxy (One-Class/PU Learning approach). The LightGBM model is tuned specifically for the high imbalance and perfectly separable nature of the synthetic dataset.
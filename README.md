\# Predictive Maintenance: RUL Prediction on NASA CMAPSS Dataset



Predicting the Remaining Useful Life (RUL) of jet engines using an LSTM neural network with advanced regularization techniques on the NASA CMAPSS turbofan engine degradation dataset.



\---



\## Results



| Model | RMSE (cycles) | MAE (cycles) | R² | NASA Score |
|---|---|---|---|---|
| **LSTM** | **14.20** | **10.63** | **0.8744** | **376.59** |
| Random Forest | 17.44 | 12.33 | 0.8105 | 1085.04 |
| Linear Regression | 20.66 | 16.42 | 0.7342 | 1077.23 |
| Mean Baseline | 40.42 | 35.17 | -0.0175 | 16915.99 |



The LSTM outperforms all baselines across every metric. NASA Score is the official asymmetric scoring function from the CMAPSS benchmark paper — it penalizes late predictions more heavily than early ones, reflecting the real-world cost of missing an engine failure.



\---



\## Key Features



\### Architecture

\- 2-layer LSTM with hidden size 64

\- Layer Normalization after each LSTM layer for training stability

\- Variational (Locked) Dropout — same dropout mask applied across all timesteps, following Merity et al. (2017) AWD-LSTM paper

\- Fully connected regression head with dropout regularization



\### Regularization (addressing overfitting)

\- Layer Normalization between LSTM layers

\- Locked/Variational Dropout (p=0.3) on inputs and between layers

\- Standard dropout (p=0.3) in the regression head

\- Weight decay (L2 regularization, λ=1e-4) in Adam optimizer

\- Gradient clipping (max norm=1.0)

\- Early stopping with patience=10 (stopped at epoch 17/50)

\- Learning rate scheduling with ReduceLROnPlateau (factor=0.5, patience=5)



\### Uncertainty Quantification

\- Monte Carlo Dropout: 50 stochastic forward passes at inference time with dropout kept active, following Gal \& Ghahramani (2016)

\- Produces per-engine confidence intervals on RUL predictions

\- Average prediction uncertainty: ±16.4 cycles

\- 64/100 test engines flagged as high uncertainty (std > 15 cycles)

\- Uncertainty-error correlation computed to assess calibration quality



\### Feature Engineering

\- 14 informative sensors selected (7 low-variance sensors dropped)

\- Rolling mean and std over 5-cycle windows per sensor (42 total features)

\- RUL clipped at 125 cycles (standard CMAPSS practice — early-life sensor readings carry no degradation signal)

\- MinMax normalization fitted on training set only (no data leakage)



\---



\## Plots



\### Predicted vs Actual RUL

!\[Predicted vs Actual](results/plots/predicted\_vs\_actual.png)



\### Error Distribution

!\[Error Distribution](results/plots/error\_distribution.png)



\### MC Dropout Uncertainty Analysis

!\[Uncertainty](results/plots/uncertainty\_analysis.png)



\### Model Comparison

!\[Model Comparison](results/plots/model\_comparison.png)



\---



\## Project Structure

nasa-rul-prediction/

├── data/                        # Raw and processed data files

├── src/

│   ├── data\_pipeline.py         # Data loading and RUL computation

│   ├── features.py              # Feature engineering pipeline

│   ├── model.py                 # LSTM architecture with locked dropout

│   ├── train.py                 # Training loop, evaluation, baselines

│   └── evaluate.py              # Plot generation

├── results/

│   ├── plots/                   # Generated visualizations

│   ├── metrics\_comparison.csv   # Metrics table for all models

│   └── uncertainty\_estimates.csv # Per-engine MC Dropout uncertainty

└── README.md



\## Setup and Usage



\### 1. Clone the repo

git clone https://github.com/yourusername/nasa-rul-prediction.git

cd nasa-rul-prediction



\### 2. Create virtual environment

python -m venv venv

venv\\Scripts\\activate       # Windows

source venv/bin/activate    # Linux/Mac



\### 3. Install dependencies

pip install torch pandas numpy scikit-learn matplotlib seaborn



\### 4. Download dataset

cd data

curl -L -o train\_FD001.txt "https://raw.githubusercontent.com/schwxd/LSTM-Keras-CMAPSS/master/C-MAPSS-Data/train\_FD001.txt"

curl -L -o test\_FD001.txt "https://raw.githubusercontent.com/schwxd/LSTM-Keras-CMAPSS/master/C-MAPSS-Data/test\_FD001.txt"

curl -L -o RUL\_FD001.txt "https://raw.githubusercontent.com/schwxd/LSTM-Keras-CMAPSS/master/C-MAPSS-Data/RUL\_FD001.txt"

cd ..

\### 5. Run pipeline

cd src

python data\_pipeline.py    # Load and process raw data

python features.py         # Feature engineering

python train.py            # Train LSTM and evaluate all models

python evaluate.py         # Generate plots

\---



\## References



\- Saxena, A. et al. (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. NASA CMAPSS Dataset.

\- Merity, S. et al. (2017). Regularizing and Optimizing LSTM Language Models. (AWD-LSTM — Locked Dropout)

\- Gal, Y. \& Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. (MC Dropout)


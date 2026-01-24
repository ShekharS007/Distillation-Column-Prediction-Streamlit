# ⚗️ Ethanol Purity Predictor

A real-time machine learning application for predicting ethanol purity in distillation columns. Built with Streamlit and trained on 2,500+ experimental datasets.

**Author:** Narendra Kumar

---

## 📋 Overview

This app predicts ethanol purity in distillation columns using machine learning models (XGBoost/Random Forest) trained on real distillation data. It takes 7 core operating parameters and auto-calculates 14 derived features to predict product quality in real-time.

### Key Features
- ✅ Real-time purity prediction
- ✅ 21 engineered features for accurate modeling
- ✅ Input validation with warnings
- ✅ Interactive UI with Streamlit
- ✅ 92% accuracy (R² = 0.92)
- ✅ Fallback demo model included

---

## 🚀 Quick Start

### Requirements
```bash
Python 3.8+
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/ethanol-purity-predictor.git
cd ethanol-purity-predictor
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Train your model in Colab (or use provided training script)

4. Export model files to local folder:
```python
import pickle

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names
with open('features_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
```

5. Place the 3 files in project folder:
```
project_folder/
├── app_ethanol_final.py
├── model.pkl
├── scaler.pkl
└── features_names.pkl
```

6. Run the app
```bash
streamlit run app_ethanol_final.py
```

Open browser to `http://localhost:8501`

---

## 📊 How It Works

### Input Parameters (7 Core)
- **Pressure (bar):** Column operating pressure
- **T1 (°C):** Top tray temperature
- **L (kmol/hr):** Reflux flow rate
- **V (kmol/hr):** Vapor flow rate
- **D (kmol/hr):** Distillate product flow
- **B (kmol/hr):** Bottoms product flow
- **F (kmol/hr):** Feed flow rate

### Auto-Calculated Features (14 Derived)
The app automatically calculates:
- **6 Derived:** Reflux Ratio, Reboiler Intensity, Condenser Load, Feed Normalized, Distillate Withdrawal, Bottoms Withdrawal
- **5 Interactions:** Reflux×Temperature, Reboiler×Temperature, Feed interactions
- **3 Efficiency:** Column Load, Separation Duty, Column Efficiency

### Output
Real-time prediction of ethanol purity (0.68 - 0.95 mole fraction) with confidence level

---

## 🎯 Best Operating Parameters (For Maximum Purity)

To get highest ethanol purity (~0.92-0.95), use these parameters:

| Parameter | Optimal Value | Range |
|-----------|---------------|-------|
| Pressure | 1.0 bar | 0.5-2.0 |
| T1 | 78-80 °C | 70-85 |
| L (Reflux) | 450-500 kmol/hr | 350-600 |
| V (Vapor) | 350-380 kmol/hr | 300-450 |
| D (Distillate) | 120-150 kmol/hr | 80-200 |
| B (Bottoms) | 30-50 kmol/hr | 20-80 |
| F (Feed) | 160-200 kmol/hr | 100-300 |

**Pro Tip:** Higher reflux ratio (L/V > 1.3) gives better separation and higher purity.

---

## 📈 Model Performance

- **Algorithm:** XGBoost / Random Forest
- **R² Score:** 0.92 (explains 92% of variance)
- **RMSE:** 0.0155
- **MAE:** 0.0122
- **Accuracy:** ±1.55%
- **Training Samples:** 2,500+

---

## ⚠️ About the Demo Model

The app includes a **fallback demo prediction model** based on distillation physics principles.

### Why Is It There?

When you first use the app or if the actual model files (model.pkl, scaler.pkl) are not found in the folder, the app switches to this demo model automatically instead of crashing.

### Why I Built It This Way

Honestly, I added the demo model for these reasons:

1. **Error Prevention:** If someone forgets to download model files or puts them in wrong folder, the app doesn't break completely. It still works, just with demo predictions.

2. **Testing During Development:** While building the app, I could test the UI and features without needing the actual model files every time.

3. **User Experience:** Instead of showing an error message and stopping, the app gives you *something* to work with. At least the user knows something is running and can see how the UI works.

4. **Production Safety:** In real production, if model loading fails for any reason (corrupt file, missing dependency, etc.), the app doesn't crash - it falls back to demo mode and logs the error. This is better than everything stopping.

### How It Works

The demo model is a simple rule-based system that mimics distillation behavior:
- Higher reflux ratio = higher purity
- Higher temperature = higher purity (with limits)
- Higher reboiler intensity = higher purity
- Predictions stay between 0.68-0.95 (realistic range)

It's **not accurate**, but it shows the concept and prevents crashes.

### Real vs Demo Mode

When you run the app:
- ✅ **Green checkmark** = Using your actual trained model
- ⚠️ **Warning icon** = Using demo model (files not found)

So you always know which one is running.

---

## 📁 Project Structure

```
ethanol-purity-predictor/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── app_ethanol_final.py         # Main Streamlit app
├── model.pkl                    # Your trained model
├── scaler.pkl                   # Feature scaler
├── features_names.pkl           # Feature names list
├── training_colab_code.py       # Colab training script
└── sample_data/
    └── dataset_distill.csv      # Example training data
```

---

## 🎓 Model Training (Colab)

To train your own model:

1. Upload your distillation data to Colab
2. Run the training script with:
   - Feature engineering (21 total features)
   - Train/validation/test split (60/20/20)
   - 5-fold cross-validation
   - XGBoost + Random Forest comparison
   - Automatic model selection (best one chosen)

3. Export the 3 files (model.pkl, scaler.pkl, features_names.pkl)

See `training_colab_code.py` for full training pipeline.

---

## 💡 Usage Tips

### For Best Results:
1. Use **steady-state** operating data (not transient)
2. Keep **mass balance** reasonable (D + B ≈ F)
3. Ensure T1 sensor is **calibrated**
4. Input parameters in **typical ranges**
5. Cross-check predictions with **lab analysis** when possible

### Typical Predictions:
- **0.92-0.95:** Excellent separation (high reflux)
- **0.85-0.90:** Normal operation (good conditions)
- **0.75-0.85:** Acceptable (standard operation)
- **<0.75:** Poor separation (increase reflux)

---

## 🔧 Configuration

You can adjust these in the code:

```python
# Input ranges (modify as needed for your column)
Pressure:  0.5 - 3.0 bar
T1:        30 - 120 °C
Flows:     0 - 5000 kmol/hr

# Temperature approximation
temp_bottom = t1 - 5  # Approximate bottom temp (adjust if needed)
```

---

## ⚡ Performance Metrics

The model was validated on:
- 500 test samples
- Diverse operating conditions
- Various feed compositions

### Results:
```
Test R²:    0.92
Test RMSE:  0.0155
Test MAE:   0.0122
CV Score:   0.91 ± 0.02
```

---

## 🐛 Troubleshooting

### "Model files not found" error
- Check files are named exactly: `model.pkl`, `scaler.pkl`, `features_names.pkl`
- Put them in same folder as `app_ethanol_final.py`
- On Linux/Mac, filenames are case-sensitive

### Predictions seem wrong
- Verify input parameters are in typical ranges
- Check mass balance (D + B should be close to F)
- Ensure column is at steady state
- Compare T1 reading with actual temperature

### App is slow
- First load caches the model (normal)
- Subsequent predictions should be instant
- Check internet speed if using cloud deployment

---

## 📊 Data Requirements

For retraining with your own data:
- **Minimum samples:** 500+
- **Features:** 14 temperature sensors (or simplified to T1 only)
- **Target:** Ethanol purity (0.60 - 1.00)
- **Format:** CSV with column headers

---

## 📚 References

- **Libraries:** Streamlit, XGBoost, scikit-learn, pandas
- **Data:** 2,500+ real distillation column experiments
- **Methods:** Feature engineering, cross-validation, regularization
- **Validation:** Train/test split with 5-fold CV

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Narendra Kumar**
- Chemical R&D Scientist
- Expertise: ML in chemical manufacturing, process optimization
- Location: Maharashtra, India

---

## 🤝 Contributing

Found a bug or have suggestions? Feel free to:
1. Open an issue
2. Submit a pull request
3. Provide feedback

---

## 📞 Contact

For questions or collaboration:
- 📧 Email: [your email]
- 💼 LinkedIn: [your profile]
- 🐙 GitHub: [@yourname]

---

**Last Updated:** January 24, 2026  
**App Version:** 1.0  
**Status:** Production Ready ✅

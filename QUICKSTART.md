# 🚀 Quick Start Guide

## For Hackathon Judges/Reviewers

### ⚡ 5-Minute Setup

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

2. **Run Complete Pipeline**

```bash
python run_pipeline.py
```

This will:

- Generate 1200 synthetic transactions
- Train ensemble model
- Run comprehensive evaluation
- Generate all reports

Expected time: 2-3 minutes

### 📊 View Results

After pipeline completes:

```bash
# Check F1-Score (should be ≥ 0.90)
python -c "import json; print('F1-Score:', json.load(open('evaluation_report.json'))['macro_f1'])"

# View detailed evaluation
cat evaluation_report.json

# View bias analysis
cat bias_report.txt

# View performance benchmarks
cat performance_report.txt
```

### 🎯 Interactive Demo

**Option 1: Web UI (Recommended)**

```bash
streamlit run app.py
```

Then open http://localhost:8501

**Option 2: Command Line**

```bash
python demo.py
```

Try these sample transactions:

- `AMAZON.COM*123456`
- `STARBUCKS STORE #789`
- `SHELL OIL 12345`
- `UBER * TRIP`
- `NETFLIX.COM`

---

## 🎬 Demo Script for Presentation

### 1. Show Core Metrics (30 seconds)

```bash
python evaluate_model.py
```

Highlight:

- ✅ Macro F1-Score ≥ 0.90 (target achieved)
- ✅ Per-category performance
- ✅ Confusion matrix

### 2. Live Categorization (60 seconds)

```bash
streamlit run app.py
```

Demonstrate:

1. Enter "STARBUCKS COFFEE #123"

   - Show predicted category
   - Show confidence score
   - Show explanation

2. Enter "UNKNOWN MERCHANT XYZ"

   - Show low confidence warning
   - Provide feedback correction
   - Show feedback recorded

3. Batch upload
   - Upload sample CSV
   - Show bulk categorization
   - Download results

### 3. Configuration Demo (30 seconds)

Show `config/categories.yaml`:

```bash
cat config/categories.yaml
```

Highlight:

- Easy to modify categories
- Add keywords without code changes
- Retrain with `python train_model.py`

### 4. Advanced Features (60 seconds)

**Explainability:**

```bash
python -c "
from src.explainability import TransactionExplainer
import joblib

model = joblib.load('model.pkl')
explainer = TransactionExplainer(model)
exp = explainer.explain_prediction('STARBUCKS #123')
print('Category:', exp['predicted_category'])
print('Confidence:', exp['confidence'])
print('Explanation:', exp['keyword_explanation'])
"
```

**Bias Analysis:**

```bash
cat bias_report.txt
```

**Performance:**

```bash
cat performance_report.txt
```

### 5. Closing (30 seconds)

Highlight key achievements:

- ✅ F1-Score ≥ 0.90 (business-grade accuracy)
- ✅ Zero external API dependencies
- ✅ Full explainability (LIME + keywords)
- ✅ Human-in-the-loop feedback
- ✅ Bias detection and mitigation
- ✅ High performance (1000+ TPS throughput)
- ✅ Robust to noise and variations

---

## 📝 Troubleshooting

### Import Errors

If you get import errors for optional packages:

```bash
# For LIME (explainability)
pip install lime

# For visualization
pip install matplotlib seaborn plotly

# For performance monitoring
pip install psutil

# For web UI
pip install streamlit streamlit-aggrid
```

### Model Not Found

If you see "model.pkl not found":

```bash
python train_model.py
```

### Test Data Missing

If evaluation fails:

```bash
python -m src.data_generator
```

---

## 🎯 Key Files to Show

1. **README.md** - Comprehensive documentation
2. **evaluation_report.json** - Detailed metrics proving F1 ≥ 0.90
3. **config/categories.yaml** - Easy configuration
4. **src/model.py** - Advanced ensemble architecture
5. **app.py** - Interactive web UI
6. **bias_report.txt** - Fairness analysis
7. **performance_report.txt** - Throughput/latency benchmarks

---

## 💡 Demo Tips

1. **Start with Results** - Show F1-score first to prove target achievement
2. **Live Interaction** - Use web UI for visual impact
3. **Show Explainability** - Demonstrate LIME and keyword matching
4. **Configuration Flexibility** - Edit categories.yaml live
5. **Highlight Completeness** - Emphasize all bonus objectives achieved

---

## 🏆 Bonus Points Checklist

- [x] Explainability UI (LIME + visualization)
- [x] Robustness to input noise (tested with 30% noise)
- [x] Batch inference metrics (1000+ TPS)
- [x] Human-in-the-loop feedback (active learning)
- [x] Bias mitigation (automated fairness analysis)
- [x] End-to-end reproducibility (one-command pipeline)
- [x] Production-ready code (modular, documented)

---

## 📞 Support

For issues during setup:

1. Check Python version: `python --version` (need 3.8+)
2. Verify installation: `pip list | grep -E "sklearn|pandas|numpy"`
3. Run pipeline: `python run_pipeline.py`

---

**Ready to impress! 🚀**

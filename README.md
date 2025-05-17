# Market Master 💹

**Market Master** is a visually immersive, multi-theme Streamlit app for financial data analysis, prediction, and visualization. It guides users through a step-by-step pipeline, offering a unique experience for each theme: anime-inspired, futuristic fintech, or classic old-money style.

---

## 🚀 Features

- **Three Unique Themes:**
  - **Financial Shinobi:** Anime-inspired, dark, energetic, and bold.
  - **Techno Exchange Protocol:** Futuristic, neon, cyber/AI, and modern.
  - **Imperial Wealth Club:** Classic, gold, vintage, and elegant.
- **Step-by-step Pipeline:**
  1. Welcome & theme selection
  2. Data upload (CSV/Excel) or fetch from Yahoo Finance
  3. Data cleaning & preprocessing
  4. Feature engineering (moving averages, volatility, etc.)
  5. Train/test split
  6. Model training (Linear Regression, Logistic Regression, K-Means Clustering)
  7. Model evaluation (metrics, graphs, interpretation)
  8. Results visualization & download
- **Theme-specific UI:**
  - Custom GIFs, CSS, icons, graph titles, axis labels, and interpretation blocks for each theme
  - Sidebar progress and navigation
- **Modern, responsive design**
- **Downloadable results**
- **No coding required for users**

---

## 🎨 Theme Showcase
| Financial Shinobi | Techno Exchange | Imperial Wealth Club |
|:----------------:|:--------------:|:-------------------:|
| ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQjp3muwfBgyjebUOVbOerdEEm8SP9yRaAgTg&s) | ![](https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif) | ![](https://media.giphy.com/media/3o6Zt6ML6BklcajjsA/giphy.gif) |

---

---

## 📚 Usage Guide

1. **Launch the app** and select your preferred theme on the landing page.
2. **Follow the sidebar steps:**
   - **Upload** your own CSV/Excel file or fetch data from Yahoo Finance.
   - **Preprocess** and clean your data (missing values, outliers).
   - **Engineer features** (moving averages, volatility, returns, etc.).
   - **Split** your data into training and testing sets.
   - **Train** models (Linear Regression, Logistic Regression, K-Means Clustering).
   - **Evaluate** model performance with custom, theme-specific graphs and metrics.
   - **Visualize and download** your results.
3. **Interpret** results with theme-specific explanations and download predictions for further analysis.

---

## 🔄 Pipeline Steps

- **Step 1:** Welcome & Theme Selection
- **Step 2:** Data Upload (CSV/Excel or Yahoo Finance)
- **Step 3:** Data Preprocessing (missing values, outlier handling)
- **Step 4:** Feature Engineering (moving averages, volatility, returns)
- **Step 5:** Train/Test Split
- **Step 6:** Model Training (choose one or more models)
- **Step 7:** Model Evaluation (metrics, actual vs. predicted, residuals, histograms)
- **Step 8:** Results Visualization & Download

---

## ⚙️ Customization
- **Easily add new themes** by extending the theme dictionaries in `app.py`.
- **Modify pipeline steps** or add new models as needed.
- **All user-facing text, graphs, and UI elements** are theme-specific and can be customized.

---

## 🧩 Troubleshooting
- If you encounter issues with data upload, ensure your file contains at least `Date` and `Close` columns.
- For Yahoo Finance, use valid stock symbols (e.g., `AAPL`, `TSLA`).
- If you see errors about missing packages, run `pip install -r requirements.txt` again.
- For best experience, use the latest version of Chrome or Firefox.

---

## 🙏 Credits
- Built with [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/python/), [scikit-learn](https://scikit-learn.org/), and [yfinance](https://github.com/ranaroussi/yfinance).
- Theme GIFs and icons are for demonstration purposes only.
- Inspired by anime, fintech, and classic finance aesthetics.

---

**Enjoy your financial adventure!** 

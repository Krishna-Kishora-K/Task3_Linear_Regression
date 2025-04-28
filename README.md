# Linear Regression - House Price Prediction

## Objective:
The objective of this task is to implement **Linear Regression** to predict housing prices using features like area, bedrooms, age, etc.

---

## Tools and Libraries Used:
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---
-Output plot is attached

## Steps Performed:

1. **Loading the Dataset**
   - Loaded the Housing CSV file using `pandas.read_csv()`.

2. **Data Preprocessing**
   - Checked for missing values.

3. **Feature Selection**
   - Selected independent variables (X) and dependent variable (y).

4. **Splitting Dataset**
   - Divided into Train and Test sets (80%-20%).

5. **Model Building**
   - Built a Linear Regression model using `sklearn.linear_model`.

6. **Model Evaluation**
   - Calculated MAE, MSE, and R² score.

7. **Visualization**
   - Plotted regression line for feature "area" vs "price".

---

## Evaluation Metrics:

- **Mean Absolute Error (MAE)**: Measures average magnitude of errors.
- **Mean Squared Error (MSE)**: Penalizes large errors.
- **R² Score**: Measures goodness of fit of the model.

---

## Observations:

- Higher area leads to higher price (positive coefficient).
- Model shows decent R² score on test data.
- Some features may need scaling or polynomial features for better performance.

---

## How to Run:

1. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn

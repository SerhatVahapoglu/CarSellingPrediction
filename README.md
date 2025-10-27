# 🚗 Car Selling Price Prediction

This project predicts the **selling price of used cars** using **Linear Regression**.  
It includes complete data preprocessing, feature encoding, scaling, and model evaluation.

---

## 📊 Project Overview

- **Algorithm:** Linear Regression  
- **Dataset:** Car data (301 entries, 9 features)  
- **Goal:** Predict `Selling_Price` based on car features such as present price, age, kilometers driven, and fuel type.  

---

## 🧠 Features

| Feature | Description |
|----------|-------------|
| `Present_Price` | Current market price of the car |
| `Kms_Driven` | Distance driven by the car |
| `Owner` | Number of previous owners |
| `Age` | Age of the car (calculated from Year) |
| `Fuel_Type_Diesel`, `Fuel_Type_Petrol` | Encoded categorical fuel types |
| `Seller_Type_Individual` | Indicates if the seller is an individual |
| `Transmission_Manual` | Indicates if the car has manual transmission |

---

## ⚙️ Model Pipeline

1. **Data Cleaning** – Removed irrelevant columns (`Car_Name`, `Year`, etc.)  
2. **Feature Encoding** – One-hot encoded categorical features  
3. **Feature Scaling** – Applied `StandardScaler` to numeric features  
4. **Model Training** – Used `LinearRegression` from `sklearn`  
5. **Evaluation Metrics:**
   - Mean Absolute Error (MAE): `1.41`
   - Mean Squared Error (MSE): `5.25`
   - R² Score: `0.84`

---

## 🔍 Example Prediction

```python
sample = pd.DataFrame({
    'Present_Price': [9.85],
    'Kms_Driven': [6900],
    'Owner': [0],
    'Age': [8],
    'Fuel_Type_Diesel': [0],
    'Seller_Type_Individual': [0],
    'Transmission_Manual': [1]
})
sample_scaled = scaler.transform(sample)
predicted_price = linear.predict(sample_scaled)
print("Predicted Selling Price:", predicted_price[0])

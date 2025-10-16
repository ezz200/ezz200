import streamlit as st
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ===============================
# 🎨 تصميم الواجهة الاحترافية
# ===============================
st.set_page_config(page_title="💰 Profit Prediction Dashboard", layout="wide")

st.markdown("""
<style>
.main-title {
    background: linear-gradient(90deg, #11998e, #38ef7d);
    color: white;
    padding: 25px;
    text-align: center;
    border-radius: 15px;
    font-size: 38px;
    font-weight: 800;
    letter-spacing: 1px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    margin-bottom: 25px;
    text-transform: uppercase;
    font-family: 'Segoe UI', sans-serif;
}
.signature {
    text-align: center;
    color: #FFD700;
    font-weight: 600;
    font-size: 18px;
    margin-top: -10px;
    letter-spacing: 1px;
    text-shadow: 0 0 10px #ffcc00, 0 0 20px #ffcc00;
}
.stApp {
    background: linear-gradient(120deg, #1f1c2c, #928dab);
    font-family: 'Segoe UI', sans-serif;
}
h2, h3, h4 {
    color: #f8f9fa !important;
    text-align: center;
}
.stButton>button {
    background: linear-gradient(45deg, #11998e, #38ef7d);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(45deg, #38ef7d, #11998e);
    transform: scale(1.05);
}
div[data-baseweb="input"] label, .stNumberInput label {
    color: #66ccff !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    text-shadow: 0 0 5px #66ccff;
}
</style>

<div class="main-title">💰 Profit Prediction Dashboard</div>
<div class="signature">by Eng. Ezz Afify</div>
""", unsafe_allow_html=True)

# ===============================
# 🔗 الاتصال بقاعدة البيانات
# ===============================
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["RetailCustomerDB"]
    collection = db["Transactions Collection"]
except Exception as e:
    st.error(f"❌ فشل الاتصال بقاعدة البيانات: {e}")
    st.stop()

# ===============================
# 📊 تحميل البيانات
# ===============================
data = list(collection.find())
df = pd.json_normalize(data)

if df.empty:
    st.error("⚠️ قاعدة البيانات فارغة! أضف بيانات أولاً.")
    st.stop()

# ===============================
# 🧮 إنشاء عمود الربح وتدريب الموديل
# ===============================
df["Profit"] = df["SaleTotalPrice"] - (df["ProductUnitPrice"] * df["Quantity"])

features = ["SaleTotalPrice", "ProductUnitPrice", "Quantity", "TotalAmount"]
X = df[features]
y = df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)


# ===============================
# 🎯 إدخال القيم الجديدة
# ===============================
st.subheader("🔮 أدخل القيم لحساب الربح المتوقع")

col1, col2 = st.columns(2)
with col1:
    sale_total = st.number_input("Sale Total Price", min_value=0.0, value=1000.0)
    product_price = st.number_input("Product Unit Price", min_value=0.0, value=600.0)
with col2:
    quantity = st.number_input("Quantity", min_value=1, value=2)
    total_amount = st.number_input("Total Amount", min_value=0.0, value=1200.0)

if st.button("🚀 احسب الربح المتوقع"):
    new_data = pd.DataFrame({
        "SaleTotalPrice": [sale_total],
        "ProductUnitPrice": [product_price],
        "Quantity": [quantity],
        "TotalAmount": [total_amount]
    })
    
    predicted_profit = model.predict(new_data)[0]
    st.success(f"💵 **الربح المتوقع:** {predicted_profit:.2f} EGP")

# ===============================
# ✨ التوقيع النهائي
# ===============================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align:center; color:#dfe6e9;'>🧠 Developed with ❤️ by Eng. Ezz Afify</h6>", unsafe_allow_html=True)

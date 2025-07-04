import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Fluffy Bakes Dashboard", layout="wide")

# ✅ White background theme
st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
        }
        .section {
            background-color: #f9f9f9;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Load data
df = pd.read_excel("clean_bakery_data.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month_name()
df['Day'] = df['Date'].dt.day_name()

# ✅ Title section
st.markdown("""
<div class="section">
    <h1 style='font-size: 40px; color: #d63384;'>🎂 Fluffy Bakes - Business Analytics Dashboard</h1>
    <p style='font-size: 20px; color: #555;'>Grow your bakery business with insights that are as sweet as your treats! 🍰</p>
</div>
""", unsafe_allow_html=True)

# ✅ Top Selling Items
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("🍪 Top 5 Best-Selling Items")
top_items = df["Item_Name"].value_counts().head(5)
fig1, ax1 = plt.subplots()
pastel_colors = ["#ffb3c1", "#c1e1ff", "#ffe4e1", "#d8bfd8", "#f9d5ec"]
sns.barplot(x=top_items.index, y=top_items.values, ax=ax1, palette=pastel_colors)
ax1.set_ylabel("Number of Orders")
ax1.set_title("Most Popular Items")
plt.xticks(rotation=30)
st.pyplot(fig1)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Monthly Sales
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("📈 Monthly Revenue Trends")
monthly = df.groupby(df['Date'].dt.to_period('M')).agg({
    "Total_Amount": "sum",
    "Order_ID": "count"
}).rename(columns={"Order_ID": "Total_Orders"})
fig2, ax2 = plt.subplots()
monthly["Total_Amount"].plot(kind="bar", ax=ax2, color="#fc9ab4")
ax2.set_title("Monthly Revenue")
ax2.set_ylabel("Revenue (₹)")
plt.xticks(rotation=45)
st.pyplot(fig2)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Peak Sale Days
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("🗓️ Peak Sale Days")
day_orders = df["Day"].value_counts()
fig3, ax3 = plt.subplots()
sns.barplot(x=day_orders.index, y=day_orders.values, ax=ax3, palette="coolwarm")
ax3.set_title("Orders by Day of the Week")
plt.xticks(rotation=45)
st.pyplot(fig3)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Payment Modes
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("💳 Payment Mode Distribution")
payment_modes = df["Payment_Mode"].value_counts()
fig4, ax4 = plt.subplots()
sns.barplot(x=payment_modes.index, y=payment_modes.values, ax=ax4, palette="pastel")
ax4.set_title("Payment Methods")
plt.xticks(rotation=0)
st.pyplot(fig4)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Customer Loyalty Pie
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("👥 Customer Loyalty")
customer_counts = df["Customer_Name"].value_counts()
repeat = customer_counts[customer_counts > 1].count()
new = customer_counts[customer_counts == 1].count()
fig5, ax5 = plt.subplots()
ax5.pie([repeat, new], labels=["Repeat", "New"], autopct="%1.1f%%", colors=["#66c2a5", "#fc8d62"])
ax5.set_title("Customer Type")
st.pyplot(fig5)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ ML: Customer Classification
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("🧠 Customer Purchase Classification")

df_class = df.groupby('Customer_Name').agg({
    'Total_Amount': 'sum',
    'Order_ID': 'count',
    'Item_Name': pd.Series.nunique
}).reset_index()
df_class.columns = ['Customer', 'Total_Spent', 'Order_Frequency', 'Unique_Items']
df_class['Label'] = df_class['Order_Frequency'].apply(lambda x: 1 if x > 1 else 0)

X = df_class[['Total_Spent', 'Order_Frequency', 'Unique_Items']]
y = df_class['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.markdown("🔍 **Classification Report**")
st.text(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["New", "Repeat"], yticklabels=["New", "Repeat"])
ax_cm.set_title("📊 Confusion Matrix: Repeat Buyer Prediction")
st.pyplot(fig_cm)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Smart Suggestions
st.markdown("""
<div class="section">
    <h2 style='color: #2b2b2b;'>💡 Smart Suggestions to Grow Your Bakery</h2>
    <ul>
        <li>🎯 <b>Boost Tuesday, Wednesday & Sunday Sales</b> – Run “Buy 2 Get 1 Free” or special launches.</li>
        <li>🍫 <b>Feature Chocolate Cake</b> – Spotlight it in offers and bundles.</li>
        <li>📦 <b>Stock Up by Monday</b> – Prepare for Tue-Wed-Sun demand.</li>
        <li>📱 <b>Instagram Strategy</b> – Post before 11 AM with engaging content.</li>
        <li>🔁 <b>Start Loyalty Program</b> – Collect emails or WhatsApp numbers.</li>
        <li>🎁 <b>Offer Repeat Discounts</b> – ₹300+ orders or birthdays = 20% off.</li>
        <li>📊 <b>Focus on June 23 & 25</b> – High demand forecast.</li>
        <li>⚠️ <b>Counter June 24 Drop</b> – “Today Only” or “Early Bird” deals.</li>
        <li>📋 <b>Collect Feedback</b> – Use QR codes on invoices or counters.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ✅ Footer
st.caption("🚀 Built by Mrudula • Grow With Data")

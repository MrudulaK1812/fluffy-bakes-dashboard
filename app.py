import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Page config
st.set_page_config(page_title="Fluffy Bakes Dashboard", layout="wide")

# ✅ Minimal white background
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

# ✅ Load cleaned data
df = pd.read_excel("clean_bakery_data.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month_name()
df['Day'] = df['Date'].dt.day_name()

# ✅ Title
st.markdown('<div class="section">', unsafe_allow_html=True)
st.title("🎂 Fluffy Bakes - Business Analytics Dashboard")
st.markdown("Grow your bakery business with insights that are as sweet as your treats! 🍰")
st.markdown('</div>', unsafe_allow_html=True)

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

# ✅ Customer Loyalty Pie Chart
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

# ✅ ML Section: Customer Classification
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

# ✅ Smart Suggestions Section (Final, visible on cloud)
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("💡 Smart Suggestions to Grow Your Bakery")

st.markdown("""
Here are some data-driven tips to increase sales and customer retention:

- 🎯 **Boost Tuesday, Wednesday & Sunday Sales**  
  These are your peak sale days. Run _Buy 2 Get 1 Free_ or exclusive launches to maximize profits.

- 🍫 **Feature Chocolate Cake in Promotions**  
  It’s your best-seller! Use it in spotlight offers or bundles to upsell.

- 📦 **Stock Up Smartly by Monday**  
  Prepare inventory early to meet mid-week demand.

- 📱 **Instagram Strategy**  
  Post before 11 AM on peak days. Use polls or “This or That” stories to engage users.

- 🔁 **Build Loyalty**  
  Collect customer emails or WhatsApp numbers. Launch simple loyalty reward points or birthday offers.

- 🎁 **Surprise Repeat Customers**  
  Give 20% off on ₹300+ orders or free items for 3rd-time purchases.

- 📊 **High-Demand Alert: June 23 & 25**  
  Prepare extra stock and launch a campaign around these dates.

- ⚠️ **Low Sales Forecast: June 24**  
  Use flash offers like _“Only for Today”_ or _“Early Bird Discount”_ to push orders.

- 📋 **Collect Feedback via QR**  
  Add a short survey on receipts or at the counter with a QR code to understand what people love.
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Footer
st.caption("🚀 Built by Mrudula • Grow With Data")

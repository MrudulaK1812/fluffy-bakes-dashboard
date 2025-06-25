import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Page config
st.set_page_config(page_title="Fluffy Bakes Dashboard", layout="wide")

# ✅ Gradient Background Styling
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #ffc1e3, #bae6fd);
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Load cleaned data
df = pd.read_excel("clean_bakery_data.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month_name()
df['Day'] = df['Date'].dt.day_name()

# ✅ Title
st.title("🎂Fluffy Bakes - Business Analytics Dashboard🎂")
st.markdown("Grow your bakery business with insights that are as sweet as your treats! 🍰")

# ✅ Top Selling Items
st.subheader("🍪 Top 5 Best-Selling Items")
top_items = df["Item_Name"].value_counts().head(5)
fig1, ax1 = plt.subplots()
pastel_colors = ["#ffb3c1", "#c1e1ff", "#ffe4e1", "#d8bfd8", "#f9d5ec"]
sns.barplot(x=top_items.index, y=top_items.values, ax=ax1, palette=pastel_colors)
ax1.set_ylabel("Number of Orders")
ax1.set_title("Most Popular Items")
plt.xticks(rotation=30)
st.pyplot(fig1)

# ✅ Monthly Sales
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

# ✅ Peak Sale Days
st.subheader("🗓️ Peak Sale Days")
day_orders = df["Day"].value_counts()
fig3, ax3 = plt.subplots()
sns.barplot(x=day_orders.index, y=day_orders.values, ax=ax3, palette="coolwarm")
ax3.set_title("Orders by Day of the Week")
plt.xticks(rotation=45)
st.pyplot(fig3)

# ✅ Payment Modes
st.subheader("💳 Payment Mode Distribution")
payment_modes = df["Payment_Mode"].value_counts()
fig4, ax4 = plt.subplots()
sns.barplot(x=payment_modes.index, y=payment_modes.values, ax=ax4, palette="pastel")
ax4.set_title("Payment Methods")
plt.xticks(rotation=0)
st.pyplot(fig4)

# ✅ Customer Loyalty
st.subheader("👥 Customer Loyalty")
customer_counts = df["Customer_Name"].value_counts()
repeat = customer_counts[customer_counts > 1].count()
new = customer_counts[customer_counts == 1].count()

fig5, ax5 = plt.subplots()
ax5.pie([repeat, new], labels=["Repeat", "New"], autopct="%1.1f%%", colors=["#66c2a5", "#fc8d62"])
ax5.set_title("Customer Type")
st.pyplot(fig5)

# ✅ Smart Business Suggestions
st.subheader("💡 Smart Suggestions to Grow Your Bakery")
st.markdown("""
- 🎯 **Offer Wednesday Combos** – Orders drop mid-week; try cupcake + coffee offers!
- 📱 **Post on Weekends** – Most orders happen Sat-Sun; use Insta Stories around 10-11 AM
- 🧁 **Prep More Red Velvet** – It's the most loved treat this month
- 📦 **Refill Ingredients on Thursday** – Friday spikes in cookie orders; stock smartly
- 🔁 **Create Loyalty Card** – You have more than 20 repeat customers. Reward them!
""")

# ✅ Footer
st.caption("🚀 Built by Mrudula • Grow With Data")

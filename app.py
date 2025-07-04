import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Page config
st.set_page_config(page_title="Fluffy Bakes Dashboard", layout="wide")

# ✅ Custom CSS
st.markdown("""
    <style>
        .stApp { background-color: #fff9f9; }
        .section {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        }
        h1, h2, h3 {
            color: #ff69b4;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Sidebar Branding

st.sidebar.title("🍰 Fluffy Bakes")
st.sidebar.markdown("Business Insights Dashboard\n\nMade with ❤️ by Mrudula")

# ✅ Load Data
df = pd.read_excel("clean_bakery_data.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month_name()
df['Day'] = df['Date'].dt.day_name()

# ✅ Header
st.markdown("""
<div class="section">
    <h1 style='text-align: center;'>🎂 Fluffy Bakes Dashboard</h1>
    <h4 style='text-align: center; color: gray;'>Sweet Insights for a Sweeter Business</h4>
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
plt.tight_layout()
st.pyplot(fig1)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Monthly Revenue
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
plt.tight_layout()
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
plt.tight_layout()
st.pyplot(fig3)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Payment Mode Distribution
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("💳 Payment Mode Distribution")
payment_modes = df["Payment_Mode"].value_counts()
fig4, ax4 = plt.subplots()
sns.barplot(x=payment_modes.index, y=payment_modes.values, ax=ax4, palette="pastel")
ax4.set_title("Payment Methods")
plt.xticks(rotation=0)
plt.tight_layout()
st.pyplot(fig4)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Customer Loyalty
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("👥 Customer Loyalty Breakdown")
customer_counts = df["Customer_Name"].value_counts()
repeat = customer_counts[customer_counts > 1].count()
new = customer_counts[customer_counts == 1].count()
fig5, ax5 = plt.subplots()
ax5.pie([repeat, new], labels=["Repeat", "New"], autopct="%1.1f%%", colors=["#66c2a5", "#fc8d62"])
ax5.set_title("Customer Type")
st.pyplot(fig5)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ ML Model: Customer Classification
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("🧠 Predicting Repeat Buyers")

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

st.markdown("🔍 **Classification Report:**")
st.text(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["New", "Repeat"], yticklabels=["New", "Repeat"])
ax_cm.set_title("📊 Confusion Matrix")
st.pyplot(fig_cm)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Business Suggestions
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("💡 Smart Suggestions to Grow Your Bakery")
st.markdown("""
- 🎯 **Boost Tuesday, Wednesday & Sunday Sales** – Run “Buy 2 Get 1 Free” or limited-time product launches on these peak days.
- 🍫 **Promote Chocolate Cake** – Your top-seller! Bundle it with slow movers or use it in Instagram Reels.
- 📦 **Stock Up Before Mondays** – Prep inventory for Tue–Wed–Sun demand surges.
- 📱 **Leverage Instagram** – Post before 11AM; use polls and reels to engage local buyers.
- 🔁 **Loyalty Focus** – Start collecting emails or numbers for deals and loyalty programs.
- 🎁 **Delight Repeat Customers** – Give ₹50 off on repeat ₹300+ orders or birthday surprises.
- 📊 **Prep for June 23 & 25** – Your best forecasted sales days. Plan deals or events around these.
- ⚠️ **Lift June 24 Sales** – Add urgency: “Today Only” or “Early Bird” offers.
- 📝 **Start Feedback Collection** – Use QR codes or digital forms to understand what customers love.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Footer
st.caption("🚀 Built with ❤️ by Mrudula • Grow with Data")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Page config
st.set_page_config(page_title="Fluffy Bakes Dashboard", layout="wide")

# ✅ CSS Styling (safe, no nested divs)
st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
        }
        .section {
            background-color: #f9f9f9;
            padding: 25px;
            border-radius: 15px;
            margin: 30px 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .section h2 {
            color: #d63384;
        }
        .section ul {
            padding-left: 1.2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Load Data
df = pd.read_excel("clean_bakery_data.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month_name()
df['Day'] = df['Date'].dt.day_name()

# ✅ Section 1: Title
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.title("🎂 Fluffy Bakes - Business Analytics Dashboard")
    st.write("Grow your bakery business with insights that are as sweet as your treats! 🍰")
    st.markdown('</div>', unsafe_allow_html=True)

# ✅ Section 2: Top Items
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🍪 Top 5 Best-Selling Items")
    top_items = df["Item_Name"].value_counts().head(5)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top_items.index, y=top_items.values, ax=ax1, palette="pastel")
    ax1.set_ylabel("Orders")
    ax1.set_title("Top Items")
    plt.xticks(rotation=30)
    st.pyplot(fig1)
    st.markdown('</div>', unsafe_allow_html=True)

# ✅ Section 3: Monthly Trends
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📈 Monthly Revenue Trends")
    monthly = df.groupby(df['Date'].dt.to_period('M')).agg({
        "Total_Amount": "sum",
        "Order_ID": "count"
    })
    fig2, ax2 = plt.subplots()
    monthly["Total_Amount"].plot(kind="bar", ax=ax2, color="#fc9ab4")
    ax2.set_title("Monthly Revenue")
    ax2.set_ylabel("₹")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

# ✅ Section 4: Peak Days
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🗓️ Peak Sale Days")
    day_orders = df["Day"].value_counts()
    fig3, ax3 = plt.subplots()
    sns.barplot(x=day_orders.index, y=day_orders.values, ax=ax3, palette="coolwarm")
    ax3.set_title("Orders by Day")
    plt.xticks(rotation=45)
    st.pyplot(fig3)
    st.markdown('</div>', unsafe_allow_html=True)

# ✅ Section 5: Payment Modes
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("💳 Payment Modes")
    payment_modes = df["Payment_Mode"].value_counts()
    fig4, ax4 = plt.subplots()
    sns.barplot(x=payment_modes.index, y=payment_modes.values, ax=ax4, palette="muted")
    ax4.set_title("Payment Method Usage")
    st.pyplot(fig4)
    st.markdown('</div>', unsafe_allow_html=True)

# ✅ Section 6: Customer Loyalty
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("👥 Customer Loyalty")
    customer_counts = df["Customer_Name"].value_counts()
    repeat = customer_counts[customer_counts > 1].count()
    new = customer_counts[customer_counts == 1].count()
    fig5, ax5 = plt.subplots()
    ax5.pie([repeat, new], labels=["Repeat", "New"], autopct="%1.1f%%", colors=["#66c2a5", "#fc8d62"])
    st.pyplot(fig5)
    st.markdown('</div>', unsafe_allow_html=True)

# ✅ Section 7: ML Model
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🧠 Customer Classification (ML)")
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
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.text("🔍 Classification Report")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["New", "Repeat"], yticklabels=["New", "Repeat"])
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)
    st.markdown('</div>', unsafe_allow_html=True)

# ✅ Section 8: Smart Suggestions
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("💡 Smart Suggestions to Grow Your Bakery")

    st.markdown("""
- 🎯 **Boost Tue, Wed & Sun** – Run offers or exclusive launches on peak days.  
- 🍫 **Promote Chocolate Cake** – Bundle it with slower items.  
- 📦 **Stock by Monday** – Ensure inventory is full before peak days.  
- 📱 **Instagram Timing** – Post before 11 AM with polls or giveaways.  
- 🔁 **Loyalty Drive** – Collect WhatsApp numbers or emails.  
- 🎁 **Repeat Reward** – Offer 20% off on ₹300+ repeat orders.  
- 📊 **Plan for June 23 & 25** – Forecast shows high demand.  
- ⚠️ **Counter June 24 Drop** – Use flash or “today-only” offers.  
- 📋 **Start Feedback** – Add QR-based feedback at the counter.
""")
    st.markdown('</div>', unsafe_allow_html=True)

# ✅ Footer
st.caption("🚀 Built by Mrudula • Powered by Streamlit & ML")

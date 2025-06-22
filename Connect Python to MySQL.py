import mysql.connector
import pandas as pd

# Connect to your MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Aditya@23",  # <-- replace with your actual password
    database="ecommerce"
)

# Query the purchases table
query = "SELECT * FROM purchases"
df = pd.read_sql(query, conn)

# Print the data
print(df.head())

from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Convert purchase_date to datetime
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# Define today's date for recency calculation
today = pd.to_datetime('2024-07-01')

# Create RFM table
rfm = df.groupby('customer_id').agg({
    'purchase_date': lambda x: (today - x.max()).days,  # Recency
    'customer_id': 'count',                             # Frequency
    'amount': 'sum'                                     # Monetary
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Scale data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Show the clustered RFM data
print(rfm)

# Plot
sns.pairplot(rfm.reset_index(), hue='Cluster', palette='tab10')
plt.suptitle('Customer Segmentation with KMeans', y=1.02)
plt.show()

rfm.to_csv("clustered_customers.csv")

conn.close()

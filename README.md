🎯 Customer Segmentation Dashboard
Streamlit App
Python 3.9+
License: MIT

AI-powered customer segmentation using K-Means clustering with RFM analysis and behavioral insights

Clusters
Silhouette Score
Customers Analyzed

🌟 Features
🎯 Smart Customer Segmentation - Automatically group customers into 5 distinct segments
📊 Interactive Dashboard - Explore customer behavior with beautiful visualizations
💰 RFM Analysis - Recency, Frequency, Monetary value insights
🎨 Modern Teal Theme - Professional, eye-catching design
📈 Actionable Insights - Get personalized recommendations for each segment
🔍 Deep Analytics - Category preferences, merchant analysis, spending patterns
🚀 Live Demo
👉 Try it now! (Deploy to get your link)

📊 Customer Segments Discovered
5 Distinct Customer Groups:
Segment	Size	Avg Spending	Key Characteristics
🌟 VIP Champions	15%	$2,450/month	High frequency, recent purchases, premium spenders
💎 Loyal Customers	25%	$1,200/month	Regular purchases, consistent engagement
🎯 Potential Loyalists	30%	$850/month	Growing engagement, moderate spending
⚠️ At Risk	20%	$450/month	Declining activity, need re-engagement
😴 Hibernating	10%	$180/month	Inactive, low engagement, churn risk
🛠️ Tech Stack
Machine Learning
K-Means Clustering - Customer segmentation algorithm
Scikit-learn - Model training and evaluation
RFM Analysis - Recency, Frequency, Monetary metrics
Pandas & NumPy - Data manipulation and feature engineering
Frontend
Streamlit - Interactive web application framework
Plotly - Dynamic, interactive visualizations
Custom CSS - Modern teal-themed UI design
Data Processing
Feature Engineering - Transaction aggregation, behavioral metrics
StandardScaler - Feature normalization
Silhouette Analysis - Optimal cluster validation
📦 Installation
Option 1: Run Locally
Copy
# Clone repository
git clone https://github.com/NourLouta/customer-segmentation.git
cd customer-segmentation

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run streamlit_app.py
The app will open at http://localhost:8501

Option 2: Docker (Optional)
Copy
# Build image
docker build -t customer-segmentation .

# Run container
docker run -p 8501:8501 customer-segmentation
📖 Usage
1. Overview Dashboard
Navigate to the 📊 Overview tab:

View total customers and segments
See cluster distribution pie chart
Analyze spending patterns across segments
Explore RFM metrics (Recency, Frequency, Monetary)
2. Segment Analysis
Navigate to the 🎯 Segment Analysis tab:

Select a customer segment from the dropdown
View detailed segment profile:
Average spending and transaction frequency
Recency of last purchase
Customer count and percentage
Get actionable recommendations for each segment
See top categories and merchants for that segment
3. Customer Insights
Navigate to the 👤 Customer Insights tab:

Search for specific customers by ID
View individual customer profile:
Total spending and transaction count
Assigned cluster and characteristics
Purchase history and patterns
Get personalized recommendations
4. Advanced Analytics
Navigate to the 📈 Analytics tab:

3D Cluster Visualization - Interactive scatter plot
Correlation Heatmap - Feature relationships
Spending Distribution - Box plots by segment
Time-based Analysis - Trends over time
📁 Project Structure
Copy
customer-segmentation/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore rules
├── data/
│   ├── raw/
│   │   └── transactions.csv         # Original transaction data
│   └── processed/
│       ├── customer_segments.csv    # Clustered customers
│       ├── customer_features.csv    # Engineered features
│       ├── cluster_profiles.csv     # Segment summaries
│       ├── transactions_eda.csv     # EDA results
│       ├── category_recommendations_by_cluster.csv
│       └── merchant_recommendations_by_cluster.csv
└── models/
    ├── kmeans_model.pkl             # Trained K-Means model
    ├── scaler.pkl                   # Feature scaler
    ├── cluster_visualization.png    # 3D cluster plot
    └── model_metrics.json           # Performance metrics
🎯 Key Features Analyzed
RFM Metrics:
📅 Recency - Days since last purchase
🔄 Frequency - Number of transactions
💰 Monetary - Total spending amount
Behavioral Features:
🛍️ Average Transaction Value - Spending per purchase
📊 Category Diversity - Number of different categories
🏪 Merchant Diversity - Number of different merchants
⏱️ Purchase Velocity - Transactions per month
🎯 Favorite Categories - Most purchased categories
🏆 Preferred Merchants - Most visited merchants
📈 Segmentation Methodology
1. Data Preprocessing
✅ Aggregated 10,000+ transactions into customer profiles
✅ Calculated RFM metrics for each customer
✅ Engineered behavioral features
✅ Handled missing values and outliers
✅ Standardized features for clustering
2. Optimal Cluster Selection
✅ Elbow method analysis
✅ Silhouette score optimization
✅ Davies-Bouldin index evaluation
✅ Selected K=5 as optimal number of clusters
3. Model Training
✅ K-Means clustering with K=5
✅ Feature scaling with StandardScaler
✅ Cluster validation and profiling
✅ Segment naming based on characteristics
4. Actionable Insights
✅ Personalized recommendations per segment
✅ Category and merchant preferences
✅ Marketing strategy suggestions
✅ Churn risk identification
💡 Business Use Cases
Marketing Teams
🎯 Targeted Campaigns - Personalized messaging per segment
📧 Email Marketing - Segment-specific promotions
💰 Budget Allocation - Focus on high-value segments
Sales Teams
🏆 VIP Programs - Reward top customers
🔄 Retention Strategies - Re-engage at-risk customers
📈 Upselling - Identify growth opportunities
Product Teams
🛍️ Product Recommendations - Category preferences
🏪 Merchant Partnerships - Popular merchant analysis
📊 Inventory Planning - Demand forecasting by segment
Executive Teams
📈 Customer Lifetime Value - Revenue potential by segment
🎯 Strategic Planning - Resource allocation decisions
📊 Performance Tracking - Segment growth monitoring
🚀 Deployment
Deploy to Streamlit Cloud (Free)
Push code to GitHub
Go to share.streamlit.io
Connect your GitHub repository
Deploy with one click!
Deploy to Heroku
Copy
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=$PORT" > Procfile

# Deploy
heroku create customer-segmentation-app
git push heroku main
🎨 Dashboard Screenshots
Overview Dashboard
📊 Cluster distribution pie chart
💰 Spending analysis by segment
📈 RFM metrics visualization
Segment Analysis
🎯 Detailed segment profiles
📋 Actionable recommendations
🏆 Top categories and merchants
Customer Insights
👤 Individual customer profiles
🔍 Purchase history analysis
💡 Personalized suggestions
🤝 Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit changes (git commit -m 'Add AmazingFeature')
Push to branch (git push origin feature/AmazingFeature)
Open a Pull Request
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

👤 Author
Nour Louta

GitHub: @NourLouta
LinkedIn: Nour Louta
Email: nour.louta@gmail.com
🙏 Acknowledgments
Dataset source: E-commerce transaction data
Built with Streamlit
ML powered by Scikit-learn
Visualizations by Plotly
RFM methodology inspired by marketing analytics best practices
📞 Support
If you found this project helpful, please ⭐ star the repository!

For issues or questions, please open an issue.

Made with ❤️ by Nour Louta | © 2026

🔮 Future Enhancements
 Add predictive churn modeling
 Implement real-time segmentation updates
 Add A/B testing framework for campaigns
 Include customer lifetime value prediction
 Add automated email campaign generator
 Implement dynamic segment optimization
 Add multi-language support
 Include export functionality (PDF reports)
📚 Learn More
K-Means Clustering Guide
RFM Analysis Methodology
Streamlit Documentation
Customer Segmentation Best Practices
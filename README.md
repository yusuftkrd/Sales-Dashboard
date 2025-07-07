# Superstore Sales Analytics Dashboard
A comprehensive, interactive sales performance monitoring dashboard built with Streamlit. Transform your sales data into actionable insights with advanced analytics, forecasting, and customer segmentation.

 ![image](https://github.com/user-attachments/assets/ff8ec085-d735-482a-b954-93e423201cf9)

# Key Features
## Analytics & KPIs
•	Real-time Metrics: Sales, profit, margins, and order tracking
•	Period Comparisons: Automatic period-over-period analysis
•	Interactive Gauges: Visual KPI monitoring with thresholds

## Advanced Filtering
•	Multi-dimensional Filters: Date range, category, region, segment
•	Dynamic Updates: Real-time dashboard updates based on selections
•	Smart Defaults: Intelligent default selections for optimal views

## Visualizations
•	Multiple Chart Types: Bar, line, area charts with customisation
•	Time Series Analysis: Monthly, quarterly, weekly, yearly views
•	Geographic Mapping: State-wise performance visualisation
•	Scatter Analysis: Sales vs profit relationship insights

 <img width="950" alt="image" src="https://github.com/user-attachments/assets/09f12847-3baa-436b-b9e8-e6e9ebd50174" />

## Customer Intelligence
•	RFM Segmentation: Recency, Frequency, Monetary analysis
•	Customer Segments: Champions, Loyal, At-Risk identification
•	Behavioural Analytics: Purchase patterns and trends 

![image](https://github.com/user-attachments/assets/1960042d-e12a-4aad-8b2e-65ac72aced2e)
![image](https://github.com/user-attachments/assets/095063aa-e99e-466f-ba40-6eae1d7291f6)

 
## Product Analytics
•	ABC Analysis: Product portfolio classification
•	Performance Ranking: Top and bottom performers
•	Profit Optimisation: Margin analysis and recommendations
 
 ![image](https://github.com/user-attachments/assets/a3b70d43-acf6-4f22-a294-e999830572f0)
![image](https://github.com/user-attachments/assets/e4c47f7e-e943-46b7-96ff-7a133f678bed)
![image](https://github.com/user-attachments/assets/ab859d92-669b-4980-a53a-76ca4775ebf4)
 
## Predictive Analytics
•	Sales Forecasting: Moving average with seasonal adjustments
•	Confidence Intervals: Statistical prediction ranges
•	Trend Analysis: Growth pattern identification

 ![image](https://github.com/user-attachments/assets/f82e2e8a-6d76-46ed-b3da-15c02448d861)
![image](https://github.com/user-attachments/assets/155d4646-bb1a-4605-823e-83b2ebde01ab)
![image](https://github.com/user-attachments/assets/c8bec0ca-0fae-457f-8f4f-9da1bcd12aef)
 
## AI-Driven Insights
•	Automated Recommendations: Business intelligence suggestions
•	Risk Identification: Concentration and performance alerts
•	Growth Opportunities: Data-driven business insights

## Quick Start
Installation
### Prerequisites - Python 3.8 or higher - pip package manager
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
Data Format
Your CSV file should include these columns:
order date, sales, profit, category, region, segment, 
customer id, product name, state, quantity, order id

### Dashboard Sections
1. Key Performance Metrics
	
- Total Sales & Profit with trend indicators
- Profit Margin gauge with performance zones
- Order volume tracking and growth rates
2. Sales & Profit Trends
	
- Time series visualisation with multiple granularities
- Target lines and performance benchmarks
- Annotations for peak performance periods
3. Category Performance
  
- Category-wise sales and profit analysis
- Small multiples for detailed category insights
- Profit margin comparison across categories
4. Sales vs Profit Analysis
	
- Scatter plot with quadrant analysis
- Profit margin colour coding
- Product-level performance insights
5. Geographic Performance
	
- US state choropleth mapping
- Regional performance comparison
- Location-based trend analysis
6. Customer Segmentation
	
- RFM analysis with customer scoring
- Segment distribution and characteristics
- Customer lifecycle insights
7. Product Portfolio
	
- ABC analysis for product prioritisation
- Top performers' identification
- Product search and filtering
8. Forecasting
	
- Sales prediction with confidence intervals
- Seasonal pattern recognition
- Growth trajectory analysis
9. Business Insights
	
- Automated insight generation
- Risk assessment and alerts
- Opportunity identification

### Technical Details
Built With
- Streamlit: Interactive web application framework
- Pandas: Data manipulation and analysis
- Plotly: Interactive visualisations
- NumPy: Numerical computing

### Key Algorithms
- RFM Segmentation: Customer value scoring
- ABC Analysis: Product portfolio classification
- Moving Average Forecasting: Time series prediction
- Seasonal Decomposition: Pattern analysis

### Performance Features
- Efficient Data Processing: Optimised pandas operations
- Smart Caching: Streamlit caching for improved performance
- Responsive Design: Works on desktop and mobile

### Sample Data
The dashboard is compatible with any sales data that contains the required columns. A sample dataset structure:

<img width="441" alt="image" src="https://github.com/user-attachments/assets/86ac403a-30ca-49df-a3fc-6a032c46d14d" />

### Customization
### Styling
- Colour palette customisation in the COLOR_PALETTE dictionary
- CSS styling through st.markdown() with unsafe_allow_html=True
- Responsive layout with Streamlit columns
### Metrics
-	Add new KPIs in the metrics calculation section
-	Customize gauge parameters in create_gauge() function
-	Modify insight generation logic for specific business rules
### Visualizations
-	Chart types configurable through sidebar radio buttons
-	Colour schemes and themes are easily adjustable
-	Plotly chart parameters are customizable

### Business Value

### For Sales Teams
-	Performance Tracking: Monitor individual and team performance
-	Target Management: Track progress against sales goals
-	Opportunity Identification: Find high-value prospects

### For Management
- Strategic Insights: Data-driven decision making
- Risk Assessment: Identify concentration risks
- Growth Planning: Forecast-based planning

### For Marketing
- Customer Segmentation: Targeted campaign planning
- Product Insights: Portfolio optimisation
- Geographic Analysis: Regional strategy development

### Configuration
### Environment Variables
# Optional: Set custom configurations
export DASHBOARD_TITLE=" Your Company Sales Dashboard"
export DEFAULT_DATE_RANGE="365"

### Deployment Options
- Streamlit Cloud: Easy deployment with GitHub integration
- Docker: Containerised deployment
- Heroku: Cloud platform deployment
- Local: Development and testing environment

### Deployment

Streamlit Cloud (Recommended)
1.	Push your code to GitHub
2.	Connect to share.streamlit.io
3.	Deploy directly from your repository
Docker Deployment
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt.
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
### Contributing
1.	Fork the repository
2.	Create your feature branch (git checkout -b feature/AmazingFeature)
3.	Commit your changes (git commit -m 'Add some AmazingFeature')
4.	Push to the branch (git push origin feature/AmazingFeature)
5.	Open a Pull Request
### License
This project is licensed under the MIT License - see the LICENSE file for details.
 


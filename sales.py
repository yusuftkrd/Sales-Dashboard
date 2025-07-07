import streamlit as st
import pandas as pd 
import numpy as np
import os
import altair as alt
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import colorsys
from scipy import stats
import logging

icon_path = os.path.abspath(r"c:\Users\yathi\Downloads\sales.png")

st.set_page_config(
    page_title="🛒 Superstore Sales Dashboard",
     page_icon=icon_path if os.path.exists(icon_path) else ":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.theme.enable("dark")

# Define a consistent color palette
COLOR_PALETTE = {
    'sales': '#3366CC',      # Deeper blue for sales
    'profit': '#33AA55',     # Rich green for profit
    'loss': '#CC3366',       # Ruby red for negative values
    'neutral': '#5D6D7E',    # Slate gray for neutral elements
    'highlight': '#FFD700',  # Gold for highlights/important metrics
    'background': 'white'  # white for backgrounds
}

# Create a custom color sequence for consistent visuals
def get_color_palette(n_colors, start_hue=0.6):
    colors = []
    for i in range(n_colors):
        hue = (start_hue + i/n_colors) % 1.0
        rgb = colorsys.hls_to_rgb(hue, 0.65, 0.75)
        colors.append(f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})')
    return colors

# Custom theme settings for Plotly
PLOTLY_THEME = {
    'layout': {
        'paper_bgcolor': COLOR_PALETTE['background'], #background color of the entire plot area
        'plot_bgcolor': COLOR_PALETTE['background'], #background color of the plotting area 
        'font': {'color': COLOR_PALETTE['neutral']}, #font color
        'title': {'font': {'size': 20, 'color': COLOR_PALETTE['neutral']}}, #title font size and font color
        'xaxis': {
            'gridcolor': '#DDDDDD', #color of grid lines on x-axis
            'zerolinecolor': '#DDDDDD', #color of zero line on x-axis
            'title': {'font': {'size': 14}} #X-axis title font size
        },
        'yaxis': {
            'gridcolor': '#DDDDDD', #color of grid lines on y-axis 
            'zerolinecolor': '#DDDDDD', #color of zero on y-axis
            'title': {'font': {'size': 14}} #Y-axis title font size 
        },
        'legend': {'font': {'size': 12}}, #Legend font size
        'margin': {'t': 50, 'b': 50, 'l': 50, 'r': 50} #plot margins, top margin, bottom margin, left margin, right margin
    }
}

# Create gauge for KPI visualization
def create_gauge(value, title, min_val=0, max_val=100, threshold=50):
    color = COLOR_PALETTE['profit'] if value >= threshold else COLOR_PALETTE['loss'] #determine color based on value and threshold
    #create gauage figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': COLOR_PALETTE['neutral']}},
        number={'suffix': '%', 'font': {'color': color}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': COLOR_PALETTE['neutral']},
            'bar': {'color': color},
            'bgcolor': 'black',
            'borderwidth': 2,
            'bordercolor': COLOR_PALETTE['neutral'],
            'threshold': {
                'line': {'color': COLOR_PALETTE['neutral'], 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(  #update layout for consistent styling
        height=150,
        autosize = True, 
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=COLOR_PALETTE['background'],
        font={'color': COLOR_PALETTE['neutral']}
    )
    return fig

# Function to color profit margin cells
def color_profit_margin(val):
    if val < 10:
        return f'color: {COLOR_PALETTE["loss"]}'
    elif val > 25:
        return f'color: {COLOR_PALETTE["profit"]}'
    else:
        return f'color: {COLOR_PALETTE["neutral"]}'

# READING THE DATASET
@st.cache_data
def load_data():
    return pd.read_csv('CA2 dataset.csv')

#Custom styling 
st.markdown(f"""
    <style>
        /* Root and Body Styling */
        .stApp {{
            background-color: {COLOR_PALETTE['background']} !important;
            color: blue !important;
        }}

        /* Metric Card Styling */
        .metric-card {{
            background-color: white !important;
            border-radius: 10px !important;
            padding: 15px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            transition: all 0.3s ease !important;
            border-left: 4px solid {COLOR_PALETTE['sales']} !important;
        }}

        .metric-card:hover {{
            transform: translateY(-5px) !important;
            box-shadow: 0 6px 8px rgba(0,0,0,0.15) !important;
        }}

        /* Metric Value Styling */
        .metric-card .stMetric-value {{
            color: black !important;
            font-weight: bold !important;
        }}

        .metric-card .stMetric-label {{
            color: {COLOR_PALETTE['neutral']} !important;
            font-weight: normal !important;
        }}

        .metric-card .stMetric-delta {{
            font-weight: normal !important;
        }}

        /* Header Styles */
        .main-header {{
            font-size: 36px !important;
            font-weight: 700 !important;
            color: {COLOR_PALETTE['sales']} !important;
            margin-bottom: 10px !important;
        }}

        .sub-header {{
            font-size: 24px !important;
            font-weight: 500 !important;
            color: {COLOR_PALETTE['neutral']} !important;
            margin-top: 0px !important;
            margin-bottom: 20px !important;
        }}

        /* Progress Bar */
        .stProgress > div > div > div > div {{
            background-color: {COLOR_PALETTE['profit']} !important;
        }}

        /* Dataframe Styling */
        .dataframe {{
            font-size: 12px !important;
            width: 100% !important;
            border-collapse: collapse !important;
        }}

        .dataframe th {{
            background-color: {COLOR_PALETTE['sales']} !important;
            color: white !important;
            padding: 10px !important;
        }}

        .dataframe td {{
            padding: 8px !important;
            border: 1px solid #ddd !important;
        }}

        /* Highlight Metric Styling */
        .highlight-metric {{
            color: {COLOR_PALETTE['highlight']} !important;
            font-weight: bold !important;
        }}

        /* Negative Value Styling */
        .negative-value {{
            color: {COLOR_PALETTE['loss']} !important;
        }}

        /* Sidebar Styling */
        .css-1aumxhk {{
            background-color: white !important;
            border-right: 1px solid #e0e0e0 !important;
        }}

        /* Button Styling */
        .stButton > button {{
            background-color: {COLOR_PALETTE['sales']} !important;
            color: white !important;
            border-radius: 5px !important;
            transition: all 0.3s ease !important;
        }}

        .stButton > button:hover {{
            background-color: {COLOR_PALETTE['profit']} !important;
            transform: scale(1.05) !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# Loading the data
try:
    df = load_data()
    
    # Convert all column names to lowercase early
    df.columns = df.columns.str.strip().str.lower()
    
    # Check and handle sales column
    if 'sales' not in df.columns:
        if 'Sales' in df.columns:
            df['sales'] = df['Sales']
        else:
            st.error("No sales column found in dataset")
            df['sales'] = 0.0
    
    # Ensure sales is numeric
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
    
    # DATA CLEANING AND PREPROCESSING
    # Convert date columns to datetime
    df['order date'] = pd.to_datetime(df['order date'], errors='coerce')
    if 'ship date' in df.columns:
        df['ship date'] = pd.to_datetime(df['ship date'], errors='coerce')
    
    # Create month-year and year columns for time analysis
    df['month-year'] = df['order date'].dt.strftime('%Y-%m')
    df['year'] = df['order date'].dt.year
    df['month'] = df['order date'].dt.month
    df['month_name'] = df['order date'].dt.month_name()
    df['quarter'] = df['order date'].dt.quarter
    df['quarter-year'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
    df['day'] = df['order date'].dt.day
    df['day_of_week'] = df['order date'].dt.day_name()
    df['week'] = df['order date'].dt.isocalendar().week
    df['week-year'] = df['year'].astype(str) + '-W' + df['week'].astype(str).str.zfill(2)
    
    # Handle missing Postal Code values
    if 'postal code' in df.columns and df['postal code'].isnull().sum() > 0:
        df['postal code'] = df['postal code'].fillna(df['postal code'].median())

    # Ensure sub-category is available
    if 'sub-category' not in df.columns and 'subcategory' in df.columns:
        df['sub-category'] = df['subcategory']

    # QUANTITY CALCULATION
    def calculate_quantity(row):
    # Category-based average unit prices
        category_avg_prices = {
        'furniture': 100,
        'office supplies': 20,
        'technology': 200
        }
    
    # Get average price for the category, with a default
        avg_price = category_avg_prices.get(
        str(row['category']).lower(), 50  # default to 50 if category not found
        )
    
    # Calculate quantity, rounding to nearest whole number
    # Ensure at least 1 quantity per sale
        return max(1, round(row['sales'] / avg_price))

    # Add quantity column
    df['quantity'] = df.apply(calculate_quantity, axis=1)
        
    # CALCULATION OF PROFIT
    if 'profit' not in df.columns:
        # Define profit margins by category
        category_profit_margin = {
            'furniture': 0.15,
            'office supplies': 0.25,
            'technology': 0.20
        }
        default_margin = 0.18
        
        # Create profit column
        if 'category' in df.columns:
            # Use vectorized operations where possible
            df['profit'] = df.apply(
                lambda row: float(row['sales']) * category_profit_margin.get(
                    str(row['category']).strip().lower(), default_margin
                ) if pd.notnull(row['sales']) else 0.0,
                axis=1
            )
        else:
            df['profit'] = df['sales'] * default_margin
    
    # Calculate profit margin
    df['profit margin'] = df.apply(
        lambda row: round((row['profit'] / row['sales'] * 100), 2) 
        if pd.notnull(row['sales']) and row['sales'] > 0 else 0.0,
        axis=1
    )
    
except Exception as e:
    st.error(f"Error loading or processing data: {e}")
    st.stop()

# Dashboard header
st.markdown('<p class="main-header">Superstore Sales Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive Sales Performance Monitor</p>', unsafe_allow_html=True)

# Sidebar filters
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shop.png", width=50)
    st.title("Dashboard Controls")
    
    # Date range filter
    min_date = df['order date'].min().date() if not df['order date'].empty else datetime.now().date()
    max_date = df['order date'].max().date() if not df['order date'].empty else datetime.now().date()
    default_start_date = max_date - timedelta(days=365)
    
    date_range = st.date_input(
        "📅 Select Date Range",
        value=[default_start_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Handling single date selection
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range[0]
        end_date = date_range[0]
    
    st.divider()
    
    # Category filter
    if 'category' in df.columns:
        categories = ['All'] + sorted(df['category'].unique().tolist())
        selected_category = st.selectbox("🏷️ Select Category", categories)
    else:
        selected_category = 'All'
        st.warning("Category column not found in data")
    
    # Region filter
    if 'region' in df.columns:
        regions = ['All'] + sorted(df['region'].unique().tolist())
        selected_region = st.selectbox("🌎 Select Region", regions)
    else:
        selected_region = 'All'
        st.warning("Region column not found in data")
    
    # Segment filter
    if 'segment' in df.columns:
        segments = ['All'] + sorted(df['segment'].unique().tolist())
        selected_segment = st.selectbox("👥 Select Segment", segments)
    else:
        selected_segment = 'All'
        st.warning("Segment column not found in data")
    
    st.divider()
    
    # Advanced options
    st.subheader("📊 Visualization Options")
    show_targets = st.checkbox("Show Target Lines", value=True)
    chart_type = st.radio("Chart Type", ["Bar", "Line", "Area"], horizontal=True)
    show_annotations = st.checkbox("Show Annotations", value=True)
    
    # Detail level selector
    detail_level = st.radio(
        "Detail Level",
        ["Monthly", "Quarterly", "Yearly", "Weekly"],
        horizontal=True
    )
    
    # Data summary
    st.divider()
    st.caption(f"Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption(f"Total records: {len(df):,}")

# Apply filters
filtered_df = df.copy()

# Date filter
filtered_df = filtered_df[(filtered_df['order date'].dt.date >= start_date) & 
                          (filtered_df['order date'].dt.date <= end_date)]

# Category filter (case-insensitive)
if selected_category != 'All' and 'category' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['category'].str.lower() == selected_category.lower()]

# Region filter (case-insensitive)
if selected_region != 'All' and 'region' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['region'].str.lower() == selected_region.lower()]

# Segment filter (case-insensitive)
if selected_segment != 'All' and 'segment' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['segment'].str.lower() == selected_segment.lower()]

# Display filter status
st.markdown(f"""
<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
    <b>📌 Active filters:</b> {start_date} to {end_date} | 
    Category: {selected_category} | 
    Region: {selected_region} | 
    Segment: {selected_segment}
</div>
""", unsafe_allow_html=True)

# Key Metrics Section
st.markdown("## 📈 Key Performance Metrics")

# Calculate metrics safely
total_sales = filtered_df['sales'].sum() if not filtered_df.empty else 0
total_profit = filtered_df['profit'].sum() if not filtered_df.empty else 0
avg_profit_margin = filtered_df['profit margin'].mean() if not filtered_df.empty else 0
total_orders = filtered_df['order id'].nunique() if 'order id' in filtered_df.columns and not filtered_df.empty else filtered_df.shape[0]

# Calculate period-over-period change
sales_change = 0
profit_change = 0
margin_change = 0
orders_change = 0

prev_sales = 0
prev_profit = 0
prev_margin = 0
prev_orders = 0

if len(date_range) == 2:
    days_in_period = (end_date - start_date).days
    previous_start = start_date - timedelta(days=days_in_period)
    previous_end = start_date - timedelta(days=1)

    previous_df = df[(df['order date'].dt.date >= previous_start) & 
                     (df['order date'].dt.date <= previous_end)]
    
    prev_sales = previous_df['sales'].sum() if not previous_df.empty else 0
    prev_profit = previous_df['profit'].sum() if not previous_df.empty else 0
    prev_margin = previous_df['profit margin'].mean() if not previous_df.empty else 0
    prev_orders = previous_df['order id'].nunique() if 'order id' in previous_df.columns and not previous_df.empty else previous_df.shape[0]
    
# Calculate changes
if prev_sales > 0:
    sales_change = ((total_sales - prev_sales) / prev_sales * 100)
if prev_profit > 0:
    profit_change = ((total_profit - prev_profit) / prev_profit * 100)
if prev_margin > 0:
    margin_change = avg_profit_margin - prev_margin
if prev_orders > 0:
    orders_change = ((total_orders - prev_orders) / prev_orders * 100)

# Creating metrics cards with styling and KPI gauges
col = st.columns((1.5, 1.5, 2,1.5), gap='medium')

with col[0]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="Total Sales",
        value=f"€{total_sales:,.2f}",
        delta=f"{sales_change:.1f}%" if sales_change != 0 else None,
        delta_color="normal" if sales_change >= 0 else "inverse"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col[1]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="Total Profit",
        value=f"€{total_profit:,.2f}",
        delta=f"{profit_change:.1f}%" if profit_change != 0 else None,
        delta_color="normal" if profit_change >= 0 else "inverse"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col[2]:
    # Profit margin gauge
    profit_margin_gauge = create_gauge(
        avg_profit_margin, 
        "Profit Margin", 
        min_val=0, 
        max_val=30, 
        threshold=15
    )
    st.plotly_chart(profit_margin_gauge, use_container_width=True)

with col[3]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="Total Orders",
        value=f"{total_orders:,}",
        delta=f"{orders_change:.1f}%" if orders_change != 0 else None,
        delta_color="normal" if orders_change >= 0 else "inverse"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Time Series Analysis with Enhanced Visualization
st.markdown("## 📊 Sales and Profit Trends")

if not filtered_df.empty:
    # Group by selected time granularity (using the detail_level from sidebar)
    if detail_level == "Monthly":
        time_data = filtered_df.groupby('month-year').agg({
            'sales': 'sum',
            'profit': 'sum',
            'profit margin': 'mean',
            'order id': 'nunique' if 'order id' in filtered_df.columns else 'count'
        }).reset_index()
        time_data['date'] = pd.to_datetime(time_data['month-year'], format='%Y-%m')
        x_column = 'month-year'
        x_title = 'Month'
    elif detail_level == "Quarterly":
        time_data = filtered_df.groupby('quarter-year').agg({
            'sales': 'sum',
            'profit': 'sum',
            'profit margin': 'mean',
            'order id': 'nunique' if 'order id' in filtered_df.columns else 'count'
        }).reset_index()
        time_data['date'] = pd.to_datetime(filtered_df.groupby('quarter-year')['order date'].min().reset_index()['order date'])
        x_column = 'quarter-year'
        x_title = 'Quarter'
    elif detail_level == "Weekly":
        time_data = filtered_df.groupby('week-year').agg({
            'sales': 'sum',
            'profit': 'sum',
            'profit margin': 'mean',
            'order id': 'nunique' if 'order id' in filtered_df.columns else 'count'
        }).reset_index()
        time_data['date'] = pd.to_datetime(filtered_df.groupby('week-year')['order date'].min().reset_index()['order date'])
        x_column = 'week-year'
        x_title = 'Week'
    else:  # Yearly
        time_data = filtered_df.groupby('year').agg({
            'sales': 'sum',
            'profit': 'sum',
            'profit margin': 'mean',
            'order id': 'nunique' if 'order id' in filtered_df.columns else 'count'
        }).reset_index()
        time_data['date'] = pd.to_datetime(time_data['year'], format='%Y')
        x_column = 'year'
        x_title = 'Year'
    
   # Sort by date
    time_data = time_data.sort_values('date')
    
    # Create time series visualization based on selected chart type
    if chart_type == "Bar":
        fig_time = px.bar(
            time_data,
            x=x_column,
            y=['sales', 'profit'],
            barmode='group',
            labels={
                'value': 'Amount (€)',
                'variable': 'Metric',
                x_column: x_title
            },
            color_discrete_map={
                'sales': COLOR_PALETTE['sales'],
                'profit': COLOR_PALETTE['profit']
            },
            title=f'Sales and Profit Trends by {x_title}'
        )
    elif chart_type == "Line":
        fig_time = px.line(
            time_data,
            x=x_column,
            y=['sales', 'profit'],
            labels={
                'value': 'Amount (€)',
                'variable': 'Metric',
                x_column: x_title
            },
            color_discrete_map={
                'sales': COLOR_PALETTE['sales'],
                'profit': COLOR_PALETTE['profit']
            },
            title=f'Sales and Profit Trends by {x_title}',
            markers=True,
            line_shape='spline'
        )
    else:  # Area
        fig_time = px.area(
            time_data,
            x=x_column,
            y=['sales', 'profit'],
            labels={
                'value': 'Amount (€)',
                'variable': 'Metric',
                x_column: x_title
            },
            color_discrete_map={
                'sales': COLOR_PALETTE['sales'],
                'profit': COLOR_PALETTE['profit']
            },
            title=f'Sales and Profit Trends by {x_title}'
        )
    
    # Add range slider to time series
    fig_time.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="category"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add annotations for key points if requested
    if show_annotations:
        # Add annotation for max sales
        max_sales_idx = time_data['sales'].idxmax()
        fig_time.add_annotation(
            x=time_data.iloc[max_sales_idx][x_column],
            y=time_data.iloc[max_sales_idx]['sales'],
            text=f"Peak: €{time_data.iloc[max_sales_idx]['sales']:,.0f}",
            showarrow=True,
            arrowhead=1,
            arrowcolor=COLOR_PALETTE['sales'],
            arrowsize=1,
            arrowwidth=2,
            ax=-40,
            ay=-40
        )
        
        # Add annotation for max profit
        max_profit_idx = time_data['profit'].idxmax()
        fig_time.add_annotation(
            x=time_data.iloc[max_profit_idx][x_column],
            y=time_data.iloc[max_profit_idx]['profit'],
            text=f"Peak: €{time_data.iloc[max_profit_idx]['profit']:,.0f}",
            showarrow=True,
            arrowhead=1,
            arrowcolor=COLOR_PALETTE['profit'],
            arrowsize=1,
            arrowwidth=2,
            ax=40,
            ay=-40
        )
    
    # Add target lines if requested
    if show_targets:
        # Calculate targets as 10% above average
        avg_sales = time_data['sales'].mean()
        avg_profit = time_data['profit'].mean()
        
        # Sales target line
        fig_time.add_shape(
            type="line",
            x0=0,
            y0=avg_sales * 1.1,
            x1=len(time_data) - 1,
            y1=avg_sales * 1.1,
            line=dict(
                color=COLOR_PALETTE['sales'],
                width=2,
                dash="dash",
            ),
            name="Sales Target"
        )
        
        # Profit target line
        fig_time.add_shape(
            type="line",
            x0=0,
            y0=avg_profit * 1.1,
            x1=len(time_data) - 1,
            y1=avg_profit * 1.1,
            line=dict(
                color=COLOR_PALETTE['profit'],
                width=2,
                dash="dash",
            ),
            name="Profit Target"
        )
        
        # Add annotations for targets
        fig_time.add_annotation(
            x=time_data.iloc[-1][x_column],
            y=avg_sales * 1.1,
            text="Sales Target",
            showarrow=False,
            xshift=50,
            font=dict(color=COLOR_PALETTE['sales'])
        )
        
        fig_time.add_annotation(
            x=time_data.iloc[-1][x_column],
            y=avg_profit * 1.1,
            text="Profit Target",
            showarrow=False,
            xshift=50,
            font=dict(color=COLOR_PALETTE['profit'])
        )
    
    st.plotly_chart(fig_time, use_container_width=True)

# Category Analysis Section with Small Multiples
st.markdown("## 📁 Category Performance Analysis")

if 'category' in filtered_df.columns and not filtered_df.empty:
    # Get categories data
    category_data = filtered_df.groupby('category').agg({
        'sales': 'sum',
        'profit': 'sum',
        'profit margin': 'mean',
        'order id': 'nunique' if 'order id' in filtered_df.columns else 'count'
    }).reset_index()
    
    # Sort by sales descending
    category_data = category_data.sort_values('sales', ascending=False)
    
    # Create the category bar chart
    fig_category = px.bar(
        category_data,
        x='category',
        y=['sales', 'profit'],
        barmode='group',
        title='Sales and Profit by Category',
        labels={'value': 'Amount (€)', 'variable': 'Metric', 'category': 'Category'},
        color_discrete_map={
            'sales': COLOR_PALETTE['sales'],
            'profit': COLOR_PALETTE['profit']
        }
    )
    
    # Enhanced tooltips
    fig_category.update_traces(
        hovertemplate='<b>%{x}</b><br>%{y:€,.2f}<extra></extra>'
    )
    
    # Add average profit margin as a line on secondary y-axis
    fig_category.add_trace(
        go.Scatter(
            x=category_data['category'],
            y=category_data['profit margin'],
            mode='lines+markers',
            name='Profit Margin',
            yaxis='y2',
            line=dict(color=COLOR_PALETTE['highlight'], width=3),
            marker=dict(size=8)
        )
    )
    
    # Update layout for dual y-axis
    fig_category.update_layout(
        yaxis2=dict(
            title='Profit Margin (%)',
            overlaying='y',
            side='right',
            ticksuffix='%'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display the chart
    st.plotly_chart(fig_category, use_container_width=True)
    
    # Small multiples for categories
    st.subheader("Category Details")
    
    # Get unique categories
    categories = filtered_df['category'].unique()
    
    # Create columns for small multiples
    cols = st.columns(min(3, len(categories)))
    
    # Loop through categories
    for i, category in enumerate(categories):
        with cols[i % 3]:
            cat_data = filtered_df[filtered_df['category'] == category]
            
            # Display category metrics
            st.markdown(f"**{category}**")
            st.metric(
                "Sales",
                f"€{cat_data['sales'].sum():,.0f}",
                f"Margin: {cat_data['profit'].sum() / cat_data['sales'].sum():.1%}" if cat_data['sales'].sum() > 0 else "N/A"
            )
            
            # Mini time series chart by month
            cat_time = cat_data.groupby('month-year').agg({'sales': 'sum'}).reset_index()
            cat_time['date'] = pd.to_datetime(cat_time['month-year'], format='%Y-%m')
            cat_time = cat_time.sort_values('date')
            
            if not cat_time.empty and len(cat_time) > 1:
                fig = px.line(
                    cat_time, 
                    x='month-year', 
                    y='sales',
                    height=150,
                    labels={'sales': 'Sales (€)', 'month-year': 'Month'}
                )
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10), 
                    showlegend=False,
                    xaxis_tickangle=-45,
                    xaxis_tickmode='auto',
                    xaxis_nticks=5
                )
                fig.update_traces(line_color=COLOR_PALETTE['sales'])
                st.plotly_chart(fig, use_container_width=True)

# Sales vs Profit Scatter Analysis
st.markdown("## 🔍 Sales vs Profit Analysis")

if not filtered_df.empty:
    # Create a scatter plot for deeper analysis
    hover_name = filtered_df.columns[0]  # Default to first column
    
    # Use 'product name' or similar if available
    for col in ['product name', 'product', 'name', 'item']:
        if col in filtered_df.columns:
            hover_name = col
            break
    
    # Create scatter plot with enhanced tooltips
    fig_scatter = px.scatter(
        filtered_df,
        x='sales',
        y='profit',
        color='profit margin',
        hover_name=hover_name,
        hover_data={
            'sales': ':€,.2f',
            'profit': ':€,.2f',
            'profit margin': ':.2f%',
            'order date': '|%B %d, %Y',
            'category': True
        },
        opacity=0.7,
        color_continuous_scale='RdYlGn')

if not filtered_df.empty:
    # Create a scatter plot for deeper analysis
    hover_name = filtered_df.columns[0]  # Default to first column
    
    # Use 'product name' or similar if available
    for col in ['product name', 'product', 'name', 'item']:
        if col in filtered_df.columns:
            hover_name = col
            break
    
    # Create scatter plot with enhanced tooltips
    fig_scatter = px.scatter(
        filtered_df,
        x='sales',
        y='profit',
        color='profit margin',
        hover_name=hover_name,
        hover_data={
            'sales': ':€,.2f',
            'profit': ':€,.2f',
            'profit margin': ':.2f%',
            'order date': '|%B %d, %Y',
            'category': True
        },
        opacity=0.7,
        color_continuous_scale='RdYlGn',  # Red for low margins, Green for high margins
        title='Sales vs Profit Relationship',
        labels={
            'sales': 'Sales (€)',
            'profit': 'Profit (€)',
            'profit margin': 'Profit Margin'
        },
        size_max=15
    )
    
    # Add quadrant lines
    fig_scatter.add_shape(
        type="line",
        x0=filtered_df['sales'].min(),
        y0=0,
        x1=filtered_df['sales'].max(),
        y1=0,
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        )
    )
    
    fig_scatter.add_shape(
        type="line",
        x0=filtered_df['sales'].mean(),
        y0=filtered_df['profit'].min(),
        x1=filtered_df['sales'].mean(),
        y1=filtered_df['profit'].max(),
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        )
    )
    
    # Add quadrant labels
    fig_scatter.add_annotation(
        x=filtered_df['sales'].mean() * 1.5,
        y=filtered_df['profit'].max() * 0.8,
        text="High Sales, High Profit",
        showarrow=False,
        font=dict(color="green"),
        bgcolor="white",
        opacity=0.8
    )
    
    fig_scatter.add_annotation(
        x=filtered_df['sales'].mean() * 0.5,
        y=filtered_df['profit'].max() * 0.8,
        text="Low Sales, High Profit",
        showarrow=False,
        font=dict(color="blue"),
        bgcolor="white",
        opacity=0.8
    )
    
    fig_scatter.add_annotation(
        x=filtered_df['sales'].mean() * 1.5,
        y=filtered_df['profit'].min() * 0.8,
        text="High Sales, Low Profit",
        showarrow=False,
        font=dict(color="orange"),
        bgcolor="white",
        opacity=0.8
    )
    
    fig_scatter.add_annotation(
        x=filtered_df['sales'].mean() * 0.5,
        y=filtered_df['profit'].min() * 0.8,
        text="Low Sales, Low Profit",
        showarrow=False,
        font=dict(color="red"),
        bgcolor="white",
        opacity=0.8
    )
    
    # Add reference for trend line
    fig_scatter.add_trace(
        go.Scatter(
            x=filtered_df['sales'],
            y=filtered_df['sales'] * filtered_df['profit'].mean() / filtered_df['sales'].mean(),
            mode='lines',
            line=dict(color='black', dash='dot'),
            name='Average Margin'
        )
    )
    
    # Update layout for better visualization
    fig_scatter.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        coloraxis_colorbar=dict(
            title="Profit Margin",
            ticksuffix="%"
        )
    )
    
    # Display the chart
    st.plotly_chart(fig_scatter, use_container_width=True)

# Geographic Analysis
st.markdown("## 🌎 Geographic Performance")

if 'state' in filtered_df.columns or 'country' in filtered_df.columns or 'region' in filtered_df.columns:
    geo_col = None
    for col in ['state', 'country', 'region']:
        if col in filtered_df.columns:
            geo_col = col
            break
    
    if geo_col:
        # Aggregate by geographic column
        geo_data = filtered_df.groupby(geo_col).agg({
            'sales': 'sum',
            'profit': 'sum',
            'profit margin': 'mean',
            'order id': 'nunique' if 'order id' in filtered_df.columns else 'count'
        }).reset_index()
        
        # Sort by sales
        geo_data = geo_data.sort_values('sales', ascending=False)
        state_to_code = {
            'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
            'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
            'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
            'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
            'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
            'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
            'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
            'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
            'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
            'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
            'District of Columbia': 'DC'
        }
        # Create choropleth for US states
        if geo_col == 'state':
            # Create a copy of the state column with codes
            geo_data['state_code'] = geo_data['state'].map(state_to_code)
            fig_geo = px.choropleth(
                geo_data,
                locations='state_code',  # Use state_code instead of state
                color='sales',
                hover_name='state',
                locationmode='USA-states',
                scope="usa",
                color_continuous_scale="Blues",
                labels={'sales': 'Sales (€)'},
                title='Sales by State'
            )
            
            # Add tooltips
            fig_geo.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>Sales: €%{z:,.2f}<br>Profit: €%{customdata[0]:,.2f}<br>Margin: %{customdata[1]:.1f}%<extra></extra>',
                customdata=geo_data[['profit', 'profit margin']].values
            )
            
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            # Create bar chart for other geographic levels
            fig_geo = px.bar(
                geo_data.head(15),
                x=geo_col,
                y='sales',
                color='profit margin',
                color_continuous_scale='RdYlGn',
                labels={
                    'sales': 'Sales (€)',
                    geo_col: geo_col.capitalize(),
                    'profit margin': 'Profit Margin'
                },
                title=f'Top 15 {geo_col.capitalize()}s by Sales'
            )
            
            # Add profit line
            fig_geo.add_trace(
                go.Scatter(
                    x=geo_data.head(15)[geo_col],
                    y=geo_data.head(15)['profit'],
                    mode='markers+lines',
                    name='Profit',
                    line=dict(color=COLOR_PALETTE['profit'], width=3),
                    marker=dict(size=8)
                )
            )
            
            st.plotly_chart(fig_geo, use_container_width=True)

# Customer Segmentation
if 'customer id' in filtered_df.columns or 'customer name' in filtered_df.columns:
    st.markdown("## 👥 Customer Analysis")
    
    customer_id_col = 'customer id' if 'customer id' in filtered_df.columns else 'customer name'
    
    # Customer metrics
    total_customers = filtered_df[customer_id_col].nunique()
    avg_order_value = filtered_df.groupby('order id')['sales'].sum().mean() if 'order id' in filtered_df.columns else filtered_df['sales'].mean()
    repeat_customers = filtered_df.groupby(customer_id_col).size()
    repeat_rate = (repeat_customers[repeat_customers > 1].count() / total_customers) * 100
    
    col = st.columns((2, 2, 2,), gap='medium')
    with col[0]:
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col[1]:
        st.metric("Average Order Value", f"€{avg_order_value:.2f}")
    
    with col[2]:
        st.metric("Repeat Purchase Rate", f"{repeat_rate:.1f}%") 

    # RFM Analysis if we have appropriate data
    if 'order date' in filtered_df.columns and 'order id' in filtered_df.columns:
        st.subheader("Customer Segmentation (RFM Analysis)")
        
        # Calculate RFM metrics
        # Latest date in dataset
        max_date = filtered_df['order date'].max()
        
        # RFM metrics
        rfm = filtered_df.groupby(customer_id_col).agg({
            'order date': lambda x: (max_date - x.max()).days,  # Recency
            'order id': 'nunique',  # Frequency
            'sales': 'sum'  # Monetary
        }).reset_index()
        
        # Rename columns
        rfm.columns = [customer_id_col, 'recency', 'frequency', 'monetary']
        
        # Calculate quintiles
        rfm['recency_rank'] = rfm['recency'].rank(method='first')
        rfm['R'] = pd.qcut(rfm['recency_rank'], 5, labels=[5, 4, 3, 2, 1])
        rfm['F'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['M'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        # Calculate RFM score
        rfm['RFM_score'] = rfm['R'].astype(int) + rfm['F'].astype(int) + rfm['M'].astype(int)
        
        # Segment customers
        def segment_customer(score):
            if score >= 13:
                return "Champions"
            elif score >= 10:
                return "Loyal Customers"
            elif score >= 7:
                return "Potential Loyalists"
            elif score >= 5:
                return "At Risk"
            else:
                return "Need Attention"
        
        rfm['segment'] = rfm['RFM_score'].apply(segment_customer)
        
        # Create segment visualization
        segment_counts = rfm['segment'].value_counts().reset_index()
        segment_counts.columns = ['segment', 'count']
        
        # Sort segments in a meaningful order
        segment_order = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Need Attention"]
        segment_counts['segment'] = pd.Categorical(segment_counts['segment'], categories=segment_order, ordered=True)
        segment_counts = segment_counts.sort_values('segment')
        
        # Create segment chart
        fig_segment = px.bar(
            segment_counts,
            x='segment',
            y='count', 
            color='segment',
            text='count',
            title='Customer Segments',
            labels={'count': 'Number of Customers', 'segment': 'Segment'},
            height=400
        )
        
        fig_segment.update_traces(texttemplate='%{text}', textposition='outside')
        
        # Display the segment chart
        st.plotly_chart(fig_segment, use_container_width=True)
        
        # Show segment details
        with st.expander("Segment Details"):
            segment_details = rfm.groupby('segment').agg({
                customer_id_col: 'count',
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': 'mean'
            }).reset_index()
            
            segment_details.columns = ['Segment', 'Count', 'Avg. Days Since Last Order', 'Avg. Orders', 'Avg. Spend']
            segment_details['Avg. Spend'] = segment_details['Avg. Spend'].map('€{:,.2f}'.format)
            segment_details['Avg. Days Since Last Order'] = segment_details['Avg. Days Since Last Order'].round(1)
            segment_details['Avg. Orders'] = segment_details['Avg. Orders'].round(1)
            
            st.dataframe(segment_details)

# Product Analysis 
if 'product name' in filtered_df.columns or 'product' in filtered_df.columns:
    st.markdown("## 📦 Product Analysis")
    
    product_col = 'product name' if 'product name' in filtered_df.columns else 'product'
    
    # Product metrics
    product_data = filtered_df.groupby(product_col).agg({
        'sales': 'sum',
        'profit': 'sum',
        'quantity': 'sum' if 'quantity' in filtered_df.columns else 'count',
        'order id': 'nunique' if 'order id' in filtered_df.columns else 'count'
    }).reset_index()
    
    # Calculate profit margin and rank
    product_data['profit margin'] = product_data['profit'] / product_data['sales'] * 100
    product_data['rank'] = product_data['sales'].rank(ascending=False)
    
    # Sort by sales
    product_data = product_data.sort_values('sales', ascending=False)
    
    # Display top products
    st.subheader("Top 10 Products by Sales")
    
    top_products = product_data.head(10)
    
    fig_products = px.bar(
        top_products,
        x=product_col,
        y='sales',
        color='profit margin',
        color_continuous_scale='RdYlGn',
        labels={
            'sales': 'Sales (€)',
            product_col: 'Product',
            'profit margin': 'Profit Margin (%)'
        },
        text='sales'
    )
    
    fig_products.update_traces(
        texttemplate='€%{text:.0f}',
        textposition='outside'
    )
    
    fig_products.update_layout(
        xaxis={'categoryorder': 'total descending'},
        coloraxis_colorbar=dict(
            title="Profit Margin",
            ticksuffix="%"
        ),
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_products, use_container_width=True)
    
    # ABC Analysis
    st.subheader("Product Portfolio Analysis (ABC)")
    
    # Calculate cumulative sales
    product_data = product_data.sort_values('sales', ascending=False)
    product_data['cumulative_sales'] = product_data['sales'].cumsum()
    product_data['percent_of_total'] = product_data['cumulative_sales'] / product_data['sales'].sum() * 100
    
    # Assign ABC categories
    def abc_category(percent):
        if percent <= 70:
            return 'A - Top Performers (70% of Sales)'
        elif percent <= 90:
            return 'B - Mid Performers (20% of Sales)'
        else:
            return 'C - Low Performers (10% of Sales)'
    
    product_data['abc_category'] = product_data['percent_of_total'].apply(abc_category)
    # Create ABC analysis visualization
    abc_counts = product_data['abc_category'].value_counts().reset_index()
    abc_counts.columns = ['category', 'count']
    
    # Calculate sales by category
    abc_sales = product_data.groupby('abc_category')['sales'].sum().reset_index()
    abc_sales.columns = ['category', 'sales']
    
    # Merge for complete data
    abc_data = abc_counts.merge(abc_sales, on='category')
    
    # Create ABC visualization
    fig_abc = px.bar(
        abc_data,
        x='category',
        y='sales',
        color='category',
        text='count',
        labels={
            'sales': 'Sales (€)', 
            'category': 'Category', 
            'count': 'Number of Products'
        },
        title='ABC Analysis: Product Portfolio Distribution'
    )
    
    fig_abc.update_traces(
        texttemplate='%{text} products',
        textposition='outside'
    )
    
    # Add sales percentage annotation
    for i, row in abc_data.iterrows():
        fig_abc.add_annotation(
            x=row['category'],
            y=row['sales'] / 2,
            text=f"{row['sales'] / abc_data['sales'].sum() * 100:.1f}% of sales",
            showarrow=False,
            font=dict(color="white", size=12)
        )
    
    st.plotly_chart(fig_abc, use_container_width=True)
    
    # Display detailed product table with search
    st.subheader("Product Details")
    
    search_term = st.text_input("Search Products", "")
    
    if search_term:
        filtered_products = product_data[product_data[product_col].str.contains(search_term, case=False)]
    else:
        filtered_products = product_data
    
    # Show table with sorting
    st.dataframe(
        filtered_products[[product_col, 'sales', 'profit', 'profit margin', 'quantity', 'rank']]
        .rename(columns={
            product_col: 'Product',
            'sales': 'Sales (€)',
            'profit': 'Profit (€)',
            'profit margin': 'Profit Margin (%)',
            'quantity': 'Quantity',
            'rank': 'Rank'
        })
        .style.format({
            'Sales (€)': '€{:,.2f}',
            'Profit (€)': '€{:,.2f}',
            'Profit Margin (%)': '{:.1f}%'
        }),
        height=400,
        use_container_width=True
    )

# Forecasting Section 
if 'order date' in filtered_df.columns and len(filtered_df) > 30:
    st.markdown("## 📈 Sales Forecast")
    
    # Check if we have enough time periods for forecasting
    if detail_level == "Monthly":
        time_col = 'month-year'
    elif detail_level == "Quarterly":
        time_col = 'quarter-year'
    elif detail_level == "Weekly":
        time_col = 'week-year'
    else:
        time_col = 'year'
    
    time_periods = filtered_df[time_col].nunique()
    
    if time_periods >= 6:  # Minimum periods for reasonable forecasting
        # Group by time period
        forecast_data = filtered_df.groupby(time_col).agg({
            'sales': 'sum'
        }).reset_index()
        
        # Define helper function for date parsing
        def parse_date_column(df, col_name, detail_level):
            """Parse date column with custom handling for different formats"""
            import re
            # Create a copy to avoid modifying the original
            date_series = df[col_name].copy()
            # Function to convert weekly format strings to datetime
            def convert_weekly_format(date_str):
                try:
                    if isinstance(date_str, str) and 'W' in date_str:
                        # Handle weekly format: "2017.0-W50" or "2017-W50"
                        match = re.search(r'(\d+)(?:\.0)?-W(\d+)', date_str)
                        if match:
                            year, week = match.groups()
                            # Convert to a proper ISO week format
                            return f"{year}-W{week}"
                        return date_str
                except:
                        return date_str
            # Clean and standardize date formats
            cleaned_dates = date_series.apply(convert_weekly_format)   
            # Apply appropriate parsing based on detail level
            if detail_level == "Monthly":
                return pd.to_datetime(cleaned_dates, format='%Y-%m')
            elif detail_level == "Weekly":
            # Use ISO week date format
                return pd.to_datetime(cleaned_dates, format='%Y-W%W')
            else:
                # Try different formats in sequence
                try:
                    return pd.to_datetime(cleaned_dates)
                except:
                # Fall back to coercion
                    return pd.to_datetime(cleaned_dates, errors='coerce')
        # Convert to datetime using the custom parser and sort
        forecast_data['date'] = parse_date_column(forecast_data, time_col, detail_level)
        forecast_data = forecast_data.sort_values('date')
        
        # Add index for forecasting
        forecast_data['time_idx'] = range(len(forecast_data))
        
        # Create forecast parameters
        num_periods = st.slider("Forecast Periods", 1, 12, 3)
        
        # Calculate moving average forecast
        sales_values = forecast_data['sales'].values
        
        # Choose window size based on data available
        window_size = min(3, len(sales_values) // 2)
        
        # Calculate moving average
        ma_values = []
        for i in range(len(sales_values)):
            if i < window_size:
                ma_values.append(np.mean(sales_values[:i+1]))
            else:
                ma_values.append(np.mean(sales_values[i-window_size+1:i+1]))
        
        forecast_data['ma_forecast'] = ma_values
        
        # Generate forecasted periods
        if forecast_data['date'].notna().any():
            last_date = forecast_data.loc[forecast_data['date'].notna(), 'date'].max()
            forecast_dates = []
    
            if detail_level == "Monthly":
                for i in range(1, num_periods + 1):
                    next_date = last_date + pd.DateOffset(months=i)
                    forecast_dates.append(next_date.strftime('%Y-%m'))
            elif detail_level == "Quarterly":
                for i in range(1, num_periods + 1):
                    next_date = last_date + pd.DateOffset(months=3*i)
                    forecast_dates.append(f"Q{(next_date.month-1)//3+1}-{next_date.year}")
            elif detail_level == "Weekly":
                for i in range(1, num_periods + 1):
                    next_date = last_date + pd.DateOffset(weeks=i)
                    forecast_dates.append(f"W{next_date.isocalendar()[1]}-{next_date.year}")
            else:  # Yearly
                for i in range(1, num_periods + 1):
                    next_date = last_date + pd.DateOffset(years=i)
                    forecast_dates.append(str(next_date.year))
        else:
            logging.error("No valid dates found in the dataset. Skipping forecast generation.")
            print("Warning: No valid dates found in the data. Cannot generate forecasts.")
            forecast_dates = []
            # Use current date as fallback to avoid errors later
            last_date = pd.Timestamp.now()  # last date to avoid Name Error last date = forecast data['date'].max()
            forecast_dates = []
        
        if detail_level == "Monthly":
            for i in range(1, num_periods + 1):
                next_date = last_date + pd.DateOffset(months=i)
                forecast_dates.append(next_date.strftime('%Y-%m'))
        elif detail_level == "Quarterly":
            for i in range(1, num_periods + 1):
                next_date = last_date + pd.DateOffset(months=3*i)
                forecast_dates.append(f"Q{(next_date.month-1)//3+1}-{next_date.year}")
        elif detail_level == "Weekly":
            for i in range(1, num_periods + 1):
                next_date = last_date + pd.DateOffset(weeks=i)
                forecast_dates.append(f"W{next_date.isocalendar()[1]}-{next_date.year}")
        else:  # Yearly
            for i in range(1, num_periods + 1):
                next_date = last_date + pd.DateOffset(years=i)
                forecast_dates.append(str(next_date.year))
        
        # Calculate seasonal factor 
        has_seasonality = False
        
        if len(forecast_data) >= 12 and detail_level == "Monthly":
            # Extract month for seasonality
            forecast_data['month'] = forecast_data['date'].dt.month
            
            # Calculate seasonal index
            seasonal_index = forecast_data.groupby('month')['sales'].mean() / forecast_data['sales'].mean()
            has_seasonality = True
        
        # Generate forecast values
        last_values = sales_values[-window_size:]
        forecast_values = []
        
        for i in range(num_periods):
            if has_seasonality and detail_level == "Monthly":
                # Predict next month with seasonality
                next_month = ((last_date.month - 1 + i + 1) % 12) + 1
                try:
                    seasonal_factor = seasonal_index[next_month]
                except KeyError:
                    print(f"Warning: Month {next_month} not found in seasonal index. Using default seasonal factor.")
                    seasonal_factor = 1.0  # Default to no seasonal effect
                next_value = np.mean(last_values) * seasonal_factor
            else:
                # Simple moving average
                next_value = np.mean(last_values)
            
            forecast_values.append(next_value)
            last_values = np.append(last_values[1:], next_value)
        
        # Create forecast dataframe
        forecast_future = pd.DataFrame({
            time_col: forecast_dates,
            'sales': forecast_values,
            'type': 'Forecast'
        })
        
        forecast_data['type'] = 'Actual'
        
        # Combine actual and forecast
        combined_data = pd.concat([
            forecast_data[[time_col, 'sales', 'type']],
            forecast_future
        ])
        
        # Create forecast visualization
        fig_forecast = px.line(
            combined_data,
            x=time_col,
            y='sales',
            color='type',
            title='Sales Forecast',
            labels={
                'sales': 'Sales (€)',
                time_col: 'Time Period',
                'type': 'Data Type'
            },
            markers=True,
            color_discrete_map={
                'Actual': COLOR_PALETTE['sales'],
                'Forecast': COLOR_PALETTE['highlight']
            }
        )
        
        # Add confidence interval 
        std_dev = forecast_data['sales'].std()
        upper_bound = forecast_future['sales'] + 1.96 * std_dev
        lower_bound = forecast_future['sales'] - 1.96 * std_dev
        
        # Ensure lower bound isn't negative
        lower_bound = lower_bound.clip(0)
        
        # Add confidence interval
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_future[time_col].tolist() + forecast_future[time_col].tolist()[::-1],
                y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,176,246,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            )
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        with st.expander("Forecast Details"):
            st.write("""
            **Forecast Methodology:**
            - It use's Historical data to predict the future data based on the trend using moving average method.
            - Sales can be seen based on diffrent time frames: Monthly, Quarterly, Weekly and Yearly.
            - For monthly data, it checks if your sales have seasonal patterns.
            """)
            
            st.dataframe(
                forecast_future.rename(columns={
                    time_col: 'Time Period',
                    'sales': 'Forecasted Sales (€)',
                    'type': 'Type'
                }).style.format({
                    'Forecasted Sales (€)': '€{:,.2f}'
                }),
                use_container_width=True
            )  

# Generate insights
st.markdown("## 💡 Key Insights")

if not filtered_df.empty:
    # Calculate insights
    insights = []
    
    # Profit margin insight
    if 'profit margin' in filtered_df.columns:
        avg_margin = filtered_df['profit margin'].mean()
        if avg_margin > 20:
            insights.append(f"📈 Strong overall profit margin of {avg_margin:.1f}%, indicating healthy pricing strategies.")
        elif avg_margin < 10:
            insights.append(f"⚠️ Low overall profit margin of {avg_margin:.1f}%. Consider reviewing pricing or cost structures.")
    
    # Top product concentration
    if 'product name' in filtered_df.columns or 'product' in filtered_df.columns:
        product_col = 'product name' if 'product name' in filtered_df.columns else 'product'
        top_product_sales = filtered_df.groupby(product_col)['sales'].sum().nlargest(1).values[0]
        top_product_percentage = (top_product_sales / filtered_df['sales'].sum()) * 100
        
        if top_product_percentage > 30:
            insights.append(f"⚠️ Product concentration risk: Top product accounts for {top_product_percentage:.1f}% of sales.")
        elif top_product_percentage < 10:
            insights.append(f"📊 Well-distributed product sales with top product accounting for only {top_product_percentage:.1f}% of sales.")
    
    # Growth trends
    if 'order date' in filtered_df.columns:
        # Calculate growth by time period
        if detail_level == "Monthly":
            time_col = 'month-year'
        elif detail_level == "Quarterly":
            time_col = 'quarter-year'
        elif detail_level == "Weekly":
            time_col = 'week-year'
        else:
            time_col = 'year'
        
        growth_data = filtered_df.groupby(time_col).agg({
            'sales': 'sum'
        }).reset_index()
        
        if len(growth_data) >= 3:
            # Calculate growth rates
            growth_data['sales_prev'] = growth_data['sales'].shift(1)
            growth_data['growth'] = (growth_data['sales'] - growth_data['sales_prev']) / growth_data['sales_prev'] * 100
            
            # Calculate recent growth (last 3 periods)
            recent_growth = growth_data.dropna().tail(3)['growth'].mean()
            
            if recent_growth > 10:
                insights.append(f"📈 Strong recent growth of {recent_growth:.1f}% over the past 3 periods.")
            elif recent_growth < 0:
                insights.append(f"📉 Sales decline of {abs(recent_growth):.1f}% over the past 3 periods.")
    
    # Category insights
    if 'category' in filtered_df.columns:
        category_data = filtered_df.groupby('category').agg({
            'sales': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        category_data['profit_margin'] = category_data['profit'] / category_data['sales'] * 100
        
        # Find best and worst categories
        best_category = category_data.loc[category_data['profit_margin'].idxmax()]
        worst_category = category_data.loc[category_data['profit_margin'].idxmin()]
        
        margin_diff = best_category['profit_margin'] - worst_category['profit_margin']
        
        if margin_diff > 20:
            insights.append(f"📊 Large profit margin variance between categories: {best_category['category']} ({best_category['profit_margin']:.1f}%) vs {worst_category['category']} ({worst_category['profit_margin']:.1f}%).")
    
    # Display insights
    if insights:
        for i, insight in enumerate(insights):
            st.markdown(f"{insight}")
    else:
        st.markdown("No significant insights detected with the current data selection.")

# Export options
st.markdown("## 📤 Export Dashboard")

export_col1, export_col2 = st.columns(2)

with export_col1:
    if st.button("Export to CSV"):
        # Generate CSV export options for the filtered data
        st.markdown("""
        💾 To export the data:
        1. Click on the three dots in the top-right corner of any data table
        2. Select "Download as CSV"
        """)

with export_col2:
    if st.button("Generate Report"):
        st.markdown("###  Executive Summary")
        
        # Generate summary text
        summary_text = f"""
        Sales Dashboard Executive Summary
        
        Period: {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}
        
        Key Metrics:
        - Total Sales: €{total_sales:,.2f}
        - Total Profit: €{total_profit:,.2f}
        - Average Profit Margin: {avg_profit_margin:.1f}%
        - Total Orders: {total_orders:,}
        
        Top Insights:
        """
        
        for insight in insights[:3]:  # Show top 3 insights
            summary_text += f"- {insight}\n"
        
        st.markdown(summary_text)

# Footer
st.markdown("---")
st.markdown("**Interactive Sales Dashboard** | Last updated: " + datetime.now().strftime('%Y-%m-%d'))            
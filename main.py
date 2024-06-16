import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as AC

# Set Streamlit page configuration
st.set_page_config(page_title='Chicago Crime Dashboard', layout='wide')

# Function to scale and plot heatmap
def scale_and_plot(df, ix=None, xlabel='Hour', ylabel='Type'):
    df_copy = df.copy()
    df_marginal_scaled = scale_df(df_copy.T).T
    if ix is None:
        ix = AC(4).fit(df_marginal_scaled).labels_.argsort()
    cap = np.min([np.max(df_marginal_scaled.to_numpy()), np.abs(np.min(df_marginal_scaled.to_numpy()))])
    df_marginal_scaled = np.clip(df_marginal_scaled, -1 * cap, cap)
    plot_hmap(df_marginal_scaled, ix=ix, xlabel=xlabel, ylabel=ylabel)

# Function to scale dataframe
def scale_df(df, axis=0):
    df_copy = df.copy()
    return (df_copy - df_copy.mean(axis=axis)) / df_copy.std(axis=axis)

# Function to plot heatmap
def plot_hmap(df, ix=None, cmap='bwr', xlabel='Hour', ylabel='Type'):
    if ix is None:
        ix = np.arange(df.shape[0])
    plt.figure(figsize=(12, 8))
    plt.imshow(df.iloc[ix, :], cmap=cmap)
    plt.colorbar(fraction=0.03)
    plt.yticks(np.arange(df.shape[0]), df.index[ix])
    plt.xticks(np.arange(df.shape[1]), df.columns, rotation=45, ha='right')  # Rotate x-axis labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    st.pyplot(plt)

# Load and preprocess data
@st.cache_data
def load_data():
    # Replace this with your actual data loading step
    df = pd.read_csv("crime_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month 
    df['Weekday'] = df['Date'].dt.day_name()
    df.index = pd.DatetimeIndex(df.Date)
    return df

df_cleaned = load_data()
crime_types = df_cleaned['Primary Type'].unique()

# Sidebar for user input
st.sidebar.title("Crime Data Filters")
selected_years = st.sidebar.multiselect('Select Year(s)', options=df_cleaned['Year'].unique(), default=None)
selected_crime_types = st.sidebar.multiselect('Select Crime Type(s)', options=crime_types, default=None)
selected_weekdays = st.sidebar.multiselect('Select Weekday(s)', options=df_cleaned['Weekday'].unique(), default=df_cleaned['Weekday'].unique())

# Filter data based on user input
filtered_df = df_cleaned[(df_cleaned['Year'].isin(selected_years)) &
                        (df_cleaned['Primary Type'].isin(selected_crime_types)) &
                        (df_cleaned['Weekday'].isin(selected_weekdays))]

# Convert weekdays to ordered categorical for sorting
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
filtered_df['Weekday'] = pd.Categorical(filtered_df['Weekday'], categories=weekday_order, ordered=True)

# Pivot table by Day of week and Primary Type
dayofweek_by_type = pd.pivot_table(filtered_df, values='ID', index='Weekday', columns='Primary Type', aggfunc='count').fillna(0)

# Tabs for different visualizations
tabs = st.sidebar.radio("Select Visualization", ["Temporal and Spatial Patterns", "Heatmaps, Line Charts, Bar Charts"])

# Temporal and Spatial Patterns tab
if tabs == "Temporal and Spatial Patterns":
    st.header("Temporal and Spatial Patterns")
    
    # Filter out NaN values in Latitude and Longitude
    filtered_df = filtered_df.dropna(subset=['Latitude', 'Longitude'])

    # Scatter plot with regplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='Longitude', y='Latitude',
                data=filtered_df,
                fit_reg=False, marker='o',
                scatter_kws={'alpha': 0.1, 'color': 'grey'},
                ax=ax)
    
    # Set labels for scatter plot
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 2D Histogram plot
    hist = ax.hist2d(x= filtered_df['Longitude'], y= filtered_df['Latitude'], bins=50, cmap='jet', alpha=0.65)
    
    # Add color bar
    cb = plt.colorbar(hist[3], ax=ax)
    cb.set_label('Counts')

    # Set the title for the plot
    ax.set_title("2D Histogram of Crime Locations")
    ax.set_xlim(-87.9, -87.5)
    ax.set_ylim(41.60, 42.05)
    ax.set_axis_off()
    st.pyplot(fig)

# Heatmaps, Line Charts, Bar Charts tab
elif tabs == "Heatmaps, Line Charts, Bar Charts":
    st.header("Heatmaps, Line Charts, Bar Charts")
    
    # Heatmap
    if st.checkbox("Show Heatmap"):
        st.subheader("Heatmap")
        # Create a mapping of numerical days to day names
        day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

        # Pivot tables
        hour_by_location = filtered_df.pivot_table(values='ID', index='Location Description', columns=filtered_df.index.hour, aggfunc=np.size).fillna(0)
        hour_by_type = filtered_df.pivot_table(values='ID', index='Primary Type', columns=filtered_df.index.hour, aggfunc=np.size).fillna(0)
        hour_by_week = filtered_df.pivot_table(values='ID', index=filtered_df.index.hour, columns=filtered_df.index.dayofweek, aggfunc=np.size).fillna(0)
        dayofweek_by_location = filtered_df.pivot_table(values='ID', index='Location Description', columns=filtered_df.index.dayofweek, aggfunc=np.size).fillna(0)
        dayofweek_by_type = filtered_df.pivot_table(values='ID', index='Primary Type', columns=filtered_df.index.dayofweek, aggfunc=np.size).fillna(0)
        location_by_type = filtered_df.pivot_table(values='ID', index='Location Description', columns='Primary Type', aggfunc=np.size).fillna(0)


        hour_by_week.columns = hour_by_week.columns.map(day_mapping)

        # Reorder columns according to the order of days
        hour_by_week = hour_by_week[selected_weekdays].T

        # Rename columns to day names for these pivot tables too
        dayofweek_by_location.columns = dayofweek_by_location.columns.map(day_mapping)
        dayofweek_by_type.columns = dayofweek_by_type.columns.map(day_mapping)

        scale_and_plot(location_by_type, xlabel='Location', ylabel='Crime Type')
        scale_and_plot(dayofweek_by_type, xlabel='Days of week', ylabel='Crime Type')
    
    # Line chart
    if st.checkbox("Show Rolling Sums Line Chart"):
        st.subheader("Rolling Sums Line Chart (Not related to days of the week)")
        
        # Pivot table by Date and Primary Type
        df_count_date = df_cleaned.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index=df_cleaned['Date'].dt.date, fill_value=0)
        df_count_date.index = pd.DatetimeIndex(df_count_date.index)
        
        # Compute rolling sums over 365 days
        df_rolling = df_count_date.rolling(365).sum()
        
        # Plotting
        fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 30), sharex=False, sharey=False)
        for i, (crime_type, ax) in enumerate(zip(df_rolling.columns, axes.flatten())):
            df_rolling[crime_type].plot(ax=ax)
            ax.set_title(crime_type)
            ax.set_xlabel('Date')
            ax.set_ylabel('Rolling Sum')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Bar chart
    if st.checkbox("Show Bar Chart"):
        st.subheader("Bar Chart")
        
        crime_counts = filtered_df['Primary Type'].value_counts()
        crime_counts = filtered_df['Primary Type'].value_counts()
        st.bar_chart(crime_counts)

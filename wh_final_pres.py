#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install plotly')


# In[5]:


import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import shap

# Function to convert regime code to descriptive names
def convert_political_regime(value):
    mapping = {
        0: 'Closed autocracy',
        1: 'Electoral autocracy',
        2: 'Electoral democracy',
        3: 'Liberal democracy'
    }
    return mapping.get(value, "Unknown")

# Load data
@st.cache_data
def load_data():
    df_final = pd.read_csv('world_happiness_latest_panel_imputed.csv')
    iq_data = pd.read_csv('avgIQpercountry.csv')
    regime_data = pd.read_csv('political-regime.csv')
    return df_final, iq_data, regime_data

df_final, iq_data, regime_data = load_data()

# Feature scaling
features = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
            'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']

scaler = MinMaxScaler(feature_range=(1, 5))
df_final_scaled = df_final.copy()
df_final_scaled[features] = scaler.fit_transform(df_final[features])

# Convert political regime codes to descriptive names
regime_data['Political regime'] = regime_data['Political regime'].apply(convert_political_regime)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Part 1: Data Exploration", "Part 2: Data Enrichment", "Part 3: Modeling & Machine Learning", "Part 4: Find Your Perfect Country"])

if page == "Home":
    st.title("World Happiness Report Analysis")
    st.text("\n")
    st.image("https://simulation.datascientest.com/assets/pictures/ds-logo.png", use_column_width=True)
    st.text("\n")
    st.image("https://regenerativesocietyfoundation.com/wp-content/uploads/2023/03/WHR-Reg_Graphic-768x384.png", use_column_width=True)

elif page == "Part 1: Data Exploration":
    st.title("Part 1: Data Exploration")

    # Count der Länder pro Jahr
    st.subheader("Number of Countries per Year")
    country_count = df_final.groupby('year')['Country name'].nunique().reset_index()
    fig = px.bar(country_count, x='year', y='Country name', labels={'Country name': 'Number of Countries'},
                 title="Number of Countries per Year")
    st.plotly_chart(fig)
    st.write("""
    The bar chart shows the number of countries included in the World Happiness Report for each year from 2005 to 2023.
    We can see an increasing trend in the number of countries reported until around 2010, after which the number stabilizes at around 140 countries per year.
    There is a noticeable dip in the number of countries reported around 2020, likely due to disruptions caused by the global pandemic.
    """)

     # Average Life Ladder over Years
    st.subheader("Average Life Ladder over Years")
    avg_lifeladder = df_final.groupby('year')['Life Ladder'].mean().reset_index()
    fig = px.line(avg_lifeladder, x='year', y='Life Ladder', title='Average Life Ladder Over Years')
    st.plotly_chart(fig)
    st.write("""
    This line chart shows the average Life Ladder score over the years from 2005 to 2023.
    After a sharp decline between 2005 and 2006, the Life Ladder score remained stable until 2016. From 2017 onward,
    there is a notable increase in happiness, peaking in 2019, followed by a decline in 2020 likely due to the pandemic,
    with a slight recovery afterward.
    """)
    
    
    # Life Ladder Distribution
    st.subheader("Life Ladder Distribution")
    plt.figure(figsize=(10, 6))
    plt.hist(df_final['Life Ladder'], bins=20, edgecolor='black')
    plt.title('Distribution of Life Ladder')
    plt.xlabel('Life Ladder')
    plt.ylabel('Number of Countries')
    st.pyplot(plt)

    st.write("""
    This histogram illustrates the distribution of Life Ladder scores, which measure the happiness level across different countries.
    The distribution follows a bell-shaped curve, with most countries having a Life Ladder score between 4 and 6.
    A smaller number of countries report either very low (around 1-3) or very high (above 7) happiness scores.
    """)


    # Correlation Matrix
    numeric_df = df_final.select_dtypes(include=[np.number])

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    plt.figure(figsize=(12, 8))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of World Happiness Factors')
    st.pyplot(plt)
    
    st.write("""
    This heatmap shows the correlation matrix of various World Happiness factors.
    The Life Ladder has a strong positive correlation with factors like GDP per capita (0.78), 
    social support (0.72), and healthy life expectancy (0.61). On the other hand, it shows a negative correlation
    with perceptions of corruption (-0.43) and negative affect (-0.35), indicating that higher happiness is associated with
    lower levels of corruption and negative emotions.
    """)

    # Average Life Ladder Map
    st.subheader("Average Life Ladder Map")
    average_life_ladder = df_final.groupby('Country name')['Life Ladder'].mean().reset_index()
    fig = px.choropleth(
        average_life_ladder,
        locations="Country name",
        locationmode='country names',
        color="Life Ladder",
        hover_name="Country name",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        title="Average Life Ladder over the years"
    )
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        )
    )
    st.plotly_chart(fig)

    st.write("""
    This map visualizes the average Life Ladder scores across different countries over the years.  
    Countries in Northern Europe and North America tend to have the highest Life Ladder scores, indicating higher levels of happiness, while many countries in Africa and the Middle East show lower scores.  
    The color gradient from green to red illustrates the range of happiness, with green representing higher scores and red indicating lower levels of happiness.
    """)
    
    # Life Ladder Development Over Years (Animated Map)
    st.subheader("Life Ladder Development Over Years")
    df_final['year'] = df_final['year'].astype(str)
    df_final = df_final.sort_values(by='year')

    fig = px.choropleth(
        df_final,
        locations="Country name",
        locationmode='country names',
        color="Life Ladder",
        hover_name="Country name",
        animation_frame="year",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        title="Life Ladder over Years")

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        )
    )
    st.plotly_chart(fig)

elif page == "Part 2: Data Enrichment":
    st.title("Part 2: Data Enrichment")
    st.write("""
    We combined two datasets that appeared interesting to us in order to further examine which factors might influence the Happiness Score. 
    Specifically, we focused on **political systems** and **education**, aiming to identify how these factors might impact global happiness.
    """)


    # Merge IQ and Political Regime
    merged_data = pd.merge(df_final, iq_data, left_on='Country name', right_on='Country', how='inner')
    merged_data_with_regime = pd.merge(merged_data, regime_data, left_on=['Country name', 'year'], right_on=['Entity', 'Year'], how='inner')
    ordered_regimes = merged_data_with_regime.groupby('Political regime')['Life Ladder'].median().sort_values().index
    
    # World Happiness and Regime Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=merged_data_with_regime['Political regime'], 
            y=merged_data_with_regime['Life Ladder'], 
            order=ordered_regimes, # Sortierte Reihenfolge der Regime
            palette="Set2")

    # Füge Titel und Achsenbeschriftungen hinzu
    plt.title('World Happiness Score by Political Regime')
    plt.xlabel('Political Regime')
    plt.ylabel('Happiness Score')
    plt.xticks(rotation=45)

# Zeige das Diagramm in Streamlit an
    st.pyplot(plt)

    st.write("""
    This boxplot shows the distribution of happiness scores across different political regimes.
    Liberal democracies tend to have the highest happiness scores, with a median close to 7,
    while closed autocracies and electoral autocracies generally report lower scores.  
    The variation within each regime is also notable, particularly in electoral democracies and autocracies,
    which have a wider range of happiness scores.
    """)

    # Scatterplot World Happiness and IQ
    st.subheader("World Happiness Score vs. Average IQ per Political Regime")
    fig = px.scatter(merged_data_with_regime, x='Average IQ', y='Life Ladder', color='Political regime',
                 title="World Happiness Score vs. Average IQ",
                 labels={"Life Ladder": "Happiness Score", "Average IQ": "Average IQ"})
    st.plotly_chart(fig)
    st.write("""
    This scatterplot illustrates the relationship between World Happiness Score and Average IQ across different political regimes.
    While there is no clear linear trend, countries with higher IQs tend to cluster around higher happiness scores, particularly in liberal democracies.
    Interestingly, there is a wide range of happiness scores at lower IQ levels, especially in electoral and closed autocracies.
    """)


elif page == "Part 3: Modeling & Machine Learning":
    st.title("Part 3: Modeling & Machine Learning")

    st.subheader("Missing Values")
    
    st.write("""  
    Log GDP per capita: 28  
    Social support: 13  
    Healthy life expectancy at birth: 63  
    Freedom to make life choices: 36  
    Generosity: 81  
    Perceptions of corruption: 125  
    Positive affect: 24  
    Negative affect: 16  
      
    The dataset had some missing values. To address this, we used the **KNN Imputer** method,
    which allowed us to fill in all the missing values in the dataset.
    """)
    
    # Feature selection and data preparation
    features = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
    X = df_final[features].dropna()
    y = df_final['Life Ladder'].loc[X.index]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optimized Gradient Boosting model
    optimized_gradient_boosting_model = GradientBoostingRegressor(
        learning_rate=0.1, max_depth=7, n_estimators=150, random_state=42
    )

    # Model initialization
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Optimized Gradient Boosting': optimized_gradient_boosting_model
    }

    # Training and evaluation
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train model
        y_pred = model.predict(X_test)  # Predict on test set
        mae = mean_absolute_error(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -np.mean(cv_scores)
        results[name] = {'MAE': mae, 'Cross-Validation MAE': cv_mae}
  
    # Display MAE scores
    st.subheader("Model Evaluation Scores (MAE)")
    results_mae_df = pd.DataFrame(results).T
    st.dataframe(results_mae_df)
    st.write("""
    Although the Random Forest model showed a slightly better Mean Absolute Error (MAE) compared to the Gradient Boosting model, we decided to use Gradient Boosting.
    This decision was made because Gradient Boosting tends to perform better on more complex, non-linear data by sequentially improving its predictions through learning from previous mistakes.
    Additionally, Gradient Boosting offers more flexibility in tuning hyperparameters, which allows for further optimization and improvements in future iterations.
    It also provides better insights into feature importance, enabling us to analyze which factors have the most influence on the Happiness Score.
    """)

    st.subheader("Model Decision")
    st.write("""
    We tested various parameters for the Gradient Boosting model to find the optimal configuration.
    The table below shows the different values we experimented with for the number of estimators, learning rate, and maximum depth.
    Based on the results, we identified the best combination of parameters, which is displayed below along with the best cross-validation score.
    """)

    
    # Testing how to improve Gradient Boosting as chosen model
    # Erstelle ein DataFrame für die Parameter
    param_data = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
    }
    param_df = pd.DataFrame(param_data)

    # Zeige den Titel und die Tabelle
    st.subheader("Testing Parameters")
    st.dataframe(param_df)

    # Best parameters und Best score (als Beispiel)
    best_params = {
        "learning_rate": 0.1,
        "max_depth": 7,
        "n_estimators": 200
    }
    best_score = 0.358

    # Ausgabe der besten Parameter und des besten Scores
    st.write("**Best parameters:**")
    st.json(best_params)  # Zeigt die Parameter im JSON-Format

    st.write(f"**Best cross-validation score:** {best_score:.3f}")


    # Feature importance for Optimized Gradient Boosting
    st.subheader("Feature Importance for Optimized Gradient Boosting Model")
    optimized_gradient_boosting_model.fit(X_train, y_train)
    feature_importances = optimized_gradient_boosting_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance for Optimized Gradient Boosting Model')
    plt.gca().invert_yaxis()
    st.pyplot(plt)

    st.write("""
    This bar chart shows the feature importance for the Gradient Boosting model used to predict happiness scores.
    We can see that **Log GDP per capita** is by far the most significant factor, followed by **Social support** and **Healthy
    life expectancy at birth**. Interestingly, factors like **Perceptions of corruption** and **Generosity** have much less influence on
    the overall happiness score compared to economic and social factors.
    """)

    #SHAP Analysis for Optimized Gradient Boosting
    st.subheader("SHAP Analysis for Optimized Gradient Boosting Model")
    explainer = shap.TreeExplainer(optimized_gradient_boosting_model)
    shap_values = explainer.shap_values(X_test)

    # SHAP Summary plot
    shap.summary_plot(shap_values, X_test, plot_type="dot")
    st.pyplot(plt.gcf())

    st.write("""
    This SHAP summary plot visualizes the impact of each feature on the predictions made by the Gradient Boosting model.
    Features like **Log GDP per capita** and **Social support** have the largest influence, with higher values typically leading to higher happiness scores.
    The color gradient indicates the feature values, where pink represents higher values and blue represents lower values, showing how different feature ranges affect the model's output.
    """)


    # Scatterplot: Actual vs Predicted for Optimized Gradient Boosting
    st.subheader("Actual vs Predicted Happiness Scores")
    y_pred_gb = optimized_gradient_boosting_model.predict(X_test)
    fig = px.scatter(x=y_test, y=y_pred_gb, labels={'x': 'Actual', 'y': 'Predicted'},
                     title="Actual vs Predicted Happiness Scores")
    st.plotly_chart(fig)

    st.write("""
    This scatter plot compares the actual happiness scores with the predicted values from the Gradient Boosting model.
    The points are closely aligned along the diagonal, indicating that the model's predictions are quite accurate for most of the data points.
    Slight deviations from the diagonal suggest areas where the model may over- or under-predict, but overall the results show a strong predictive performance.
    """)


    # Predictions for interesting countries
    st.subheader("Predictions for Interesting Countries in 2023")

    # Ensure the 'year' column is correctly formatted as integer
    df_final['year'] = df_final['year'].astype(int)

    # Generate predictions for all data points
    df_final['Predicted_Life_Ladder'] = optimized_gradient_boosting_model.predict(df_final[features])

    # Define the list of interesting countries
    interesting_countries = ['Germany', 'Finland','Afghanistan','Nigeria']

    # Filter the DataFrame for interesting countries and year 2023
    interesting_data = df_final[(df_final['Country name'].isin(interesting_countries)) & (df_final['year'] == 2023)]

    # Generate a summary table with descriptive statistics for predicted values
    summary_table_predicted = interesting_data.groupby('Country name').agg(
        Mean_Predicted_Life_Ladder=('Predicted_Life_Ladder', 'mean'),
        Median_Predicted_Life_Ladder=('Predicted_Life_Ladder', 'median'),
        Mean_Log_GDP=('Log GDP per capita', 'mean'),
        Mean_Social_Support=('Social support', 'mean'),
        Mean_Healthy_Life_Expectancy=('Healthy life expectancy at birth', 'mean'),
        Mean_Freedom=('Freedom to make life choices', 'mean'),
        Mean_Generosity=('Generosity', 'mean'),
        Mean_Corruption_Perception=('Perceptions of corruption', 'mean')
    ).reset_index()

    # Display the summary table
    #st.dataframe(summary_table_predicted)

    # Create a bar plot for predicted happiness scores
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.barplot(x='Country name', y='Mean_Predicted_Life_Ladder', data=summary_table_predicted)
    plt.title('Predicted Happiness Scores for Several Countries (2023)')
    plt.xlabel('Country')
    plt.ylabel('Predicted Happiness Score')
    plt.xticks(rotation=45)  # Rotate country names for better readability
    st.pyplot(plt)

    st.write("""
    Afghanistan: As a country enduring ongoing conflict and humanitarian crises, Afghanistan’s low predicted happiness score reflects the profound challenges its population faces, making it a crucial outlier in the analysis.

    Finland: Regularly ranked as the happiest country in the world, Finland serves as a standard for evaluating the effectiveness of the prediction model. Its score can be used to compare how social security, equality, and overall well-being impact happiness.

    Germany: A strong economy with a high standard of living, Germany is interesting due to its complex socio-political landscape, including issues such as immigration and political polarization, which may influence the happiness score.

    Nigeria: As a rapidly developing country with strong contrasts in wealth and poverty, Nigeria presents a unique case for predicting happiness. Economic growth and social challenges coexist, making its score an important reflection of these dynamics.
    """)

elif page == "Part 4: Find Your Perfect Country":
    st.title("Part 4: Find Your Perfect Country")

    # Interactive sliders for feature selection
    st.subheader("Adjust the sliders based on your preferences:")

    gdp = st.slider('How important is the economy (GDP per capita)?', 1, 5, 3)
    social_support = st.slider('How important is social support?', 1, 5, 3)
    life_expectancy = st.slider('How important is healthy life expectancy?', 1, 5, 3)
    freedom = st.slider('How important is freedom to make life choices?', 1, 5, 3)
    generosity = st.slider('How important is generosity?', 1, 5, 3)
    corruption = st.slider('How important is the perception of low corruption?', 1, 5, 3)

    # Button to calculate the perfect country
    if st.button('Calculate'):
        # Create a DataFrame with the user's input
        user_input = pd.DataFrame({
            'Log GDP per capita': [gdp],
            'Social support': [social_support],
            'Healthy life expectancy at birth': [life_expectancy],
            'Freedom to make life choices': [freedom],
            'Generosity': [generosity],
            'Perceptions of corruption': [corruption]
        })

        # Rescale user input back to original scale
        user_input_scaled = scaler.inverse_transform(user_input)
        user_input_df = pd.DataFrame(user_input_scaled, columns=features)

        # Predict happiness score with the trained model
        gradient_boosting_model = GradientBoostingRegressor(random_state=42)
        gradient_boosting_model.fit(df_final[features], df_final['Life Ladder'])
        predicted_happiness = gradient_boosting_model.predict(user_input_df)[0]

        # Find the country with the closest happiness score
        df_final['Score Difference'] = abs(df_final['Life Ladder'] - predicted_happiness)
        closest_country = df_final.loc[df_final['Score Difference'].idxmin(), 'Country name']
        closest_score = df_final.loc[df_final['Score Difference'].idxmin(), 'Life Ladder']

        st.success(f'The perfect country for your preferences is {closest_country} with a Happiness Score of {closest_score:.2f}.')

# Add signature at the bottom of the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Presented by Britta & Robert")


# In[ ]:





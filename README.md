# Analysis on House/Property rates in United States

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://housingprices.streamlit.app/)

## Introduction
Lately, there is a lot of information on prices of residential properties at different locations. Although it seems easier to find out all these data, considering how fast and reliable ChatGPT and other AI tools have emerged, but the real problem is how to analyze once you have all the data. The app I designed is a small replica of analyzation of residential and house properties, and helps users to analyze in an organized manner. 

This dashboard enables users to view property/housing prices at diifferent states and cities of United States and can research about their favorite locations to settle in the long run.

The app has three pages:

- **Home**: Introduction
- **Search**: It allows user to search different locations through zip codes and cities and can compare them with different metrics or explore   them.
- **Analyze**: Getting insights of house prices through boxplot measures.

## Data Sources and Preparation 
The data used in this app is found in different locations. 
1. Demographic, housing at zip level (https://data.census.gov/)
2. Environmental metrics (https://data.cdc.gov/)
3. Mapping zip to USPS city (https://data.opendatasoft.com/pages/home/)
4. Appreciation Rates Q2 2023 (https://www.statista.com/statistics/1240802/annual-home-price-appreciation-by-state-usa/)

I have mapped zip codes to cities with other metrics and demographics in one csv file. Appreciation rates are extracted into another csv file.  

## Use Case
It is important to note why, where, and how the property rates are showcasing the growth in different parts of United States. This can enable users like buyers, dealerships, and people companies target specific cities/counties depending on prices and location.

## App Features
- Analyze which cities and states have better housing rates depending on various metrics.
- View and compare the cities to analyze the similarity scores and choose the desired living location.
- Analyze the annual appreciation of residential properties to determine the housing rates of 50 states from United States. View data in different quartiles for better insights.

## Future Work
This app can be extended to include additional countries and add more living location metrics to analyze the best city/state for buying the properties.

## Streamlit App
https://housingprices.streamlit.app/

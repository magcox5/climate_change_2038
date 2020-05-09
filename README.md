# Final_project

## By John Byun, Kat Anggasastra, Melissa Wright and Molly Cox
### [With additional technical support from tutors Mark Steadman and Alexis McKenzie]
## The Task
1. Find a problem worth solving, analyzing, or visualizing
2. Use ML in the context of technologies learned.
3. Use: Scikit-Learn and/or another machine learning library. We use: SkLearn AND Prophet  https://facebook.github.io/prophet/
4. Use at least two of: We Use:  Python Pandas , Python Matplotlib , Plotly, Tableau
5. 15-minute data deep-dive or infrastructure walkthrough that shows machine learning in the context of what we’ve already learned.
6. Examples: Create an analysis of existing data to make a prediction, classification, or regression. We use: Prediction and Regression

## Story/Presentation

 ## Climate Analysis and Forecasting with Machine Learning (1993-2015)

#### Motivation:  
* Inspired by Earth Day, and our previous research on contributing factors to CO2 emissions from our first bootcamp project (Commute Chaos), we decided to research historical datasets to make climate predictions with machine learning. We also wondered what impacts the global shelter-in place might have on the climate. We looked at population, emissions and what relationship they have on temperature, the melting of glaciers and rise in sea level. Climate Scientists tell us that the Earth is warming, which will have catastrophic global effects, so our team used machine learning to study just how soon we will reach the dangerous temperatures climate scientists may be referring to. 

####  Background: 
* According to the Paris Climate Agreement of 2015,  the global goal is to limit the overall increase in global temperature by 1.5℃ or less from the average prior to pre-industrial times. Determining how to calculate the starting temperature is controversial, but for the sake of argument, we took the average temperature in our dataset for 1850, which was around 14.87℃ or 58.76℉. An increase of 1.5℃ would be 16.37℃, or 61.47℉.  

#### Summary:
* We compared data of temperature, sea level, CO2 emissions and population from 1993-2015. We levered our machine learning forecast to  answer at When will the temperature rise to a dangerous level of 16.37 𝇈C /61.47𝇈F? Our research and analysis showed that there is a positive trend in the rise of population, CO2 emissions, temperature and global mean sea level. We leveraged machine learning to create a forecast for the next 10 years. Our model showed as a good model from the testing and training data. The forecast showed a positive trend that we saw that we will reach the concerning temperature between 2028-2038 . We used sklearn and prophet along with visualizations in Tableau to examine our datasets. Our sources of data are all listed below at the bottom of this paper. 

### Our question : "When will we reach the concerning temperature?" 

* CO2 emissions from burning fossil fuels. CO2 emissions are also produced by natural processes, but up to 1880, the earth could absorb those and keep temperatures stable. Man-made emissions are pushing the Earth out of balance,  causing more CO2 to stay in the atmosphere and create a “greenhouse gas” effect to warm the planet. 

* Population -  increased population results in increased energy consumption and fossil fuel burning, which would lead to higher CO2 levels.

Effects of Climate Change:
* Temperatures - increasing - leads to more extreme weather events and melting of sea ice, which leads to:
* Sea Level - Sea Level is constantly ranging as it is affected by tidal changes. Satellite sensors are used to measure the global mean sea level. Overall sea level rising causes coastal flooding resulting in islands to be submerged, consequently causing displacements of population therefore it is an important dataset to monitor. Data used from 1993-2015 was the global mean sea level and that number averaged by year shows a downward trend.  Data from 1993-2020 was also looked at further and that showed an upward trend.  


#### Analysis:

* Temperature rise by Year:
We took the temperature dataset from Berkeley Earth, which had temperature data going back to 1850.  We used the average combined land and sea temperature for that year.  Our regression produced the following plot:
![sklearn regression year to temp](images/Year_to_Temp_correlation.png)


* The chart shows a slow increase in temperature from 1850 to roughly 1980, and then a sharper rise noted by the regression line.  Plugging in the temperature 16.37 into the linear equation, the Year this temperature is reached would be 2038.

* Temperature rise based on population:
Using the same temperature dataset for the years 1993 to 2015, we looked at the relationship between population data and temperature. Our regression produced this plot:

![sklearn regression population to temp](images/pop_to_temp.png)

This chart shows that temperature rises with population.  According to this regression line, a temperature of 16.37 would be reached when the population reaches approximately 9.29 Billion, which is in the ballpark for projected population size for that year.

* CO2 Emissions as related to Population growth
According to this regression line, the population has a direct correlation to CO2 Emissions. 

![CO2 emissions with population](images/population_to_co2.png)

* CO2 Emissions over time. 
![CO2 emissions with population](images/CO2_emission_overtime.png)
 linear regression model based on year and total CO2 emission tons per capita. 
~ The peak you see prior to 2006 is the rapid growth of carbon emissions with no sign of slowing down! 
~ However, it quickly declining by 14% from the peak in 2005 and end in 2016; 
~ The drop you see results from our new alternative technologies. Main 3 technologies are:  
The power sector are switching gear from coal to gas 
3% use of Solar technology   
17% use of Wind technology   

* CO2 Emissions per capita by country. 
![CO2 emissions tons per capita](images/CO2_emission_tons_per_capita.png)
According to this Tableau visualization, The map shows the top 10 culprits with the highest CO2 emission per capita
* But hey look! US is number one on the list since 1945! 
* However, the one unexpected one is Trinidad and Tobago - a tiny little country off the coast northeastern Venezuela is #8 on the list! 

* The map shows the total sum of co2 tons per capita breakdown by each country from 1949 - 2017. Top 4 CO2 emission countries are: United States(2,136), England(1,927), Australia(1,187), and Canada(1,519)   According to our other data on co2 based on yearly average ppm(parts per millions); the report mentions that the ppm had rapidly grown since 1910 where the ppm is around 300 and now in 2018, are at 408.52. In the past 108 years, our co2 ppm had grown 108.52 around 1% per each year. 
China’s ppm is rather low because in they only ramping up since 1947.

##### Update 30 April 2020: The International Energy Agency has forecast the CO2 impact of the crisis, suggesting emissions could fall by 8% this year, some 2,600MtCO2. It draws on more recent data and covers 100% of global emissions, whereas Carbon Brief’s estimate was based on information covering only around 75% of the total. Global emissions would need to fall by some 7.6% every year this decade  – in order to limit warming to less than 1.5C above pre-industrial temperatures.




#### General Visualization

=======
![Tableau all ](images/Tableau_all_fig1.png)

##### Machine Learning Trend

![ML 1](images/ML_trend_fig2.png)

#####  Machine Learning Prediction

![ML 2](images/ML_forecast_daily_fig7.png)

##### Machine Learning Testing/Training

![ML 3](images/ML_train_model_fig9.png)


## Datasets & Data Scrubbing Process

#### Temperature Data
Data Source:  https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data#GlobalTemperatures.csv
Original Source is Berkeley Earth Surface Temperature Study, a group affiliated with Lawrence Berkeley National Laboratory
We are interested in Average Yearly Land and Ocean Data from 1990 on.
* Source File:        GlobalTemperatures.csv
* Jupyter Notebooks:   Temp_Scrub, Temp_Scrub-AllYears, Temp_Scrub-AllYears-Min-Max
* Scrubbed Files:      temp_data.csv, temp_data-AllYears.csv, temp_data-Min-Max.csv

#### Sea Level Data 
Data Source: https://sealevel.nasa.gov/understanding-sea-level/key-indicators/global-mean-sea-level/
This dataset contains the Global Mean Sea Level (GMSL) generated from the Integrated Multi-Mission Ocean Altimeter Data for Climate Research (GMSL dataset) available for download. 
* Source File: GMSL_merged_nasa_1993_2020
* Jupyter Notebook:   sealevel_scrub.ipynb
* Scrubbed File:      sealevel_data.csv


#### Population Data found :

* Source File:        WORLD_POP.csv 
* Jupyter Notebook:   population_scrub.ipynb
* Scrubbed File:      population_data.csv

#### Emissions Data found :

* Source File:        CO2_data
* Jupyter Notebook:   emissions_scrub.ipynb
* Scrubbed File:      co2_ppm.csv and co-emissions-per-capita



 ## Challenges



 ## If we had more time...

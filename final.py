"""
This script performs data wrangling and exploratory data analysis (EDA) on the Formula 1 (F1) Championship dataset.
It merges multiple data files related to F1 races, circuits, drivers, constructors, and more into a single dataframe.
The script then performs various visualizations to analyze circuit usage, driver nationalities, race victories, and more.
"""

# Rest of the code...
#%% [markdown]

# # F1 Championship Prediction


#%% [markdown]
# # What is Formula 1?
#
# Formula 1 (F1) is the pinnacle of international auto racing, featuring highly specialized, cutting-edge single-seater cars that compete in a series of races known as Grand Prix. 
# Governed by the Fédération Internationale de l'Automobile (FIA), F1 attracts the world's top racing teams and drivers. 
# Races take place on diverse circuits, including traditional tracks and street circuits, with the championship culminating in a points-based system where drivers and teams vie for titles. 
# The sport is characterized by intense speed, technological innovation, strategic pit stops, and aerodynamic excellence, making it a global spectacle that captivates millions of fans worldwide.
#
#  The Formula 1 point system is a crucial component in determining the overall standings for drivers and teams throughout the season. 
# Points are awarded based on finishing positions in each Grand Prix, with the winner receiving 25 points. 
# The subsequent positions receive decreasing points, with the top ten finishers earning points on a scale that extends to 10th place. 
# Additionally, bonus points are awarded for achievements such as the fastest lap in the race, provided the driver finishes in the top ten. 
# The point system serves to highlight consistency and performance across the season, ultimately crowning the World Drivers' and Constructors' Champions based on their accumulated points.

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import plotly.express as px

# %%[markdown]
# # Dataset Overview
#
# The dataset for this project was sourced from Kaggle. It contains data for F1 races from 1950 to 2023.
# We have 14 files. Each file gives information on the following:
# 1. Circuits 
# 2. Constructor Results
# 3. Constructor Standings
# 4. Constructors
# 5. Driver Standings
# 6. Drivers
# 7. Lap Times
# 8. Pit Stops
# 9. Qualifying
# 10. Results
# 11. Seasons
# 12. Sprint Results
# 13. Status
# 14. Races
#
# Each file has a primary key which can be used to join the files together. 


#%% [markdown]
# # Smart Questions
#
# 1. How have lap times changed over the years?
# 2. How do lap times change over the course of a season?
# 3. Can we do lap time analysis using EDA?
# 4. Utilizing supervised learning techniques, can we predict the final points earned by a driver?
# 5. Can we predict the final points earned by a driver?
# 
#%% [markdown]
# # Loading the Data


# %%
# Load data
circuits = pd.read_csv('./data/circuits.csv', na_values='\\N')
constructor_results = pd.read_csv('./data/constructor_results.csv', na_values='\\N')
constructor_standings = pd.read_csv('./data/constructor_standings.csv', na_values='\\N')
constructors = pd.read_csv('./data/constructors.csv', na_values='\\N')
driver_standings = pd.read_csv('./data/driver_standings.csv', na_values='\\N')
drivers = pd.read_csv('./data/drivers.csv', na_values='\\N')
lap_times = pd.read_csv('./data/lap_times.csv', na_values='\\N')
pit_stops = pd.read_csv('./data/pit_stops.csv', na_values='\\N')
qualifying = pd.read_csv('./data/qualifying.csv', na_values='\\N')
races = pd.read_csv('./data/races.csv', na_values='\\N')
results = pd.read_csv('./data/results.csv', na_values='\\N')
seasons = pd.read_csv('./data/seasons.csv', na_values='\\N')
sprint_results = pd.read_csv('./data/sprint_results.csv', na_values='\\N')
status = pd.read_csv('./data/status.csv', na_values='\\N')

print('Data loaded successfully!')
#%% [markdown]
# # Merging the Data
#
# We will merge the data into a single dataframe. This will make it easier to work with the data.
#k=races['raceId', 'year', 'name', 'round', 'date','circuitId']
columns = ["raceId","points"]

# Create a new list with column names appended with "name"
new_columns = [col for col in columns]

# Print the new list of columns
new_columns=pd.DataFrame(new_columns, columns=['driver_standings_column_names'])
new_columns
#%%
dfmerged = pd.merge(results, races[['raceId', 'year', 'name', 'round', 'date','circuitId']], on='raceId', how='left')
dfmerged = pd.merge(dfmerged, circuits[['circuitId','circuitRef', 'location', 'country']],on='circuitId', how='left')
dfmerged = pd.merge(dfmerged, drivers[['driverId', 'driverRef', 'forename', 'surname', 'nationality', 'dob']], on='driverId',
              how='left')
dfmerged = pd.merge(dfmerged, constructors[['constructorId', 'name', 'nationality']], on='constructorId', how='left')
dfmerged = pd.merge(dfmerged, status[['statusId', 'status']], on='statusId', how='left')
dfmerged=pd.merge(dfmerged,driver_standings[["raceId","points"]],on='raceId',how='left')
#dfmerged=pd.merge(dfmerged,constructor_standings[["raceId","points"]],on='raceId',how='left')
print('Data merged successfully!')

#%%
dfmerged.head()

#%% [markdown]
#  The merged data has 35 columns and 246,857 rows.
#  The columns give information on driver name, circuit name, constructor id, results, nationality, date of birth, etc.

#%% [markdown]
# # EDA

#%% [markdown]
# # Circuit Visualization
# We will keep only the required columns for circuit visualization.
      

# %%
#Merging dataframes
merge_circuits = circuits.merge(races, how="inner", on="circuitId")
merge_circuits.head()
df = merge_circuits[['year', 'date', 'name_y', 'name_x', 'location', 'country', 'lat', 'lng', 'time']]
df.head()

# %%
# Most frequent F1 circuits
circuits_frequence = df.groupby('name_x')['year'].count().sort_values(ascending = True).tail(10)
palette = sns.color_palette('Set1', len(df['name_x']))
circuits_frequence.plot.barh(color=palette)
plt.title("Most Frequently Used Circuits in F1")
plt.xlabel("Frequency")
plt.ylabel("Circuit")

print(" Ther most frequently used circuit is Monza, Italy.")

# %%
# Most frequent F1 countries
countries_frequence = df.groupby('country')['year'].count().sort_values(ascending = True).tail(10)
palette2 = sns.color_palette('Set2', len(df['country']))
countries_frequence.plot.barh(color=palette2)
plt.title("Most Frequent Countries in F1")
plt.xlabel("Frequency")
plt.ylabel("Country")

print(" Ther most frequently used country is Italy.")
#%%
# Circuits by Country
circuits_countries = df.groupby('country')['name_x'].nunique().sort_values(ascending = True).tail(10)
palette2 = sns.color_palette('Set2', len(df['country']))
circuits_countries.plot.barh(color=palette)
plt.title("Circuits by Country in F1")
plt.xlabel("Quantity")
plt.ylabel("Country")

print(" USA has the most circuits in F1.")


# %%
# Mapping circuits
import folium
coordinates=[]
for lat,lng in zip(df['lat'],df['lng']):
    coordinates.append([lat,lng])
maps = folium.Map(zoom_start=2,tiles='OpenStreetMap')
for i,j in zip(coordinates,df.name_x):
    marker = folium.Marker(
        location=i,
        icon=folium.Icon(icon="flag-checkered",color='gray',prefix='fa'),
        popup="<strong>{0}</strong>".format(j))
    marker.add_to(maps)
maps
#%%
print("The map shows the location of the circuits in the world.")
#%%
#Number of circuits by year
plt.plot(df.groupby('year')['name_x'].count(), linestyle="-", color='red')
plt.title("Circuits by Season in F1")
plt.xlabel("Year Edition")
plt.ylabel("Circuits")

print("The number of circuits has been increasing over the years.")
print(" We can see a slight dip in 2020. This is due to the COVID-19 pandemic.")

# %% [markdown]
# # Drivers  and Constructor Visualization

#%%
# Nationality of drivers
nationality_counts = drivers.groupby("nationality").size()
sorted_nationalities = nationality_counts.sort_values(ascending=False)
sorted_nationalities.head(10)


# %%
# Top 10 Nationalities of Drivers
plt.figure(figsize=(10, 6))
sorted_nationalities.plot(kind='bar', color='skyblue')
plt.title('Top 10 Nationalities of Drivers')
plt.xlabel('Nationality')
plt.ylabel('Number of Drivers')
plt.show()

print("The top 3 nationalities of drivers are British, American and Italian.")


# %%
# Max points by driver overall
df_driver_won = dfmerged.groupby('surname')['points_x'].sum().reset_index()
df_driver_won = df_driver_won.sort_values('points_x', ascending=False)
plt.title('Max points by Driver overall')
plt.xlabel('Constructor')
plt.ylabel('Points')
sns.barplot(x='surname', y='points_x', data=df_driver_won.head(20))
plt.xticks(rotation=90)

print("The driver with the most points is Lewis Hamilton.")

# %%
# Max points by constructor overall
df_constructor_won = dfmerged.groupby('name_y')['points_x'].sum().reset_index()
df_constructor_won = df_constructor_won.sort_values('points_x', ascending=False)
sns.barplot(x='name_y', y='points_x', data=df_constructor_won.head(20))
plt.title('Max points by constructor overall')
plt.xlabel('Constructor')
plt.ylabel('Points')
plt.xticks(rotation=90)

print("The constructor with the most points is Ferrari.")

#%%
# most_races = dfmerged.groupby('surname')[['raceId']].count().reset_index()
# most_races = most_races.sort_values('raceId', ascending= False).head(10)
# print(most_races)
# most_races = most_races.rename(columns ={'raceId': 'total_races'})
# plt.figure(figsize = (20,10))
# plt.title('Top 10 Drivers with most race entries in Formula 1')
# sns.barplot(x = 'total_races' , y = 'driver_name' , data = most_races )

# %% [markdowm]
# # Preprocessing the number of race wins for top 5 drivers
#
drivers = pd.read_csv('./data/drivers.csv')
results = pd.read_csv('./data/results.csv')

concat_driver_name = lambda x: f"{x.forename} {x.surname}" 

drivers['driver'] = drivers.apply(concat_driver_name, axis=1)
# Preparing F1 history victories dataset
results_copy = results.set_index('raceId').copy()
races_copy = races.set_index('raceId').copy()

results_copy = results_copy.query("position == '1'")
results_copy['position'] = 1 # casting position 1 to int 

results_cols = ['driverId', 'position']
races_cols = ['date']
drivers_cols = ['driver', 'driverId']

results_copy = results_copy[results_cols]
races_copy = races_copy[races_cols]
drivers_copy = drivers[drivers_cols]

f1_victories = results_copy.join(races_copy)
f1_victories = f1_victories.merge(drivers_copy, on='driverId', how='left')

# Victories cumulative sum
f1_victories = f1_victories.sort_values(by='date')

f1_victories['victories'] = f1_victories.groupby(['driverId']).cumsum()   

# Getting the top five f1 biggest winners drivers id
f1_biggest_winners = f1_victories.groupby('driverId').victories.nlargest(1).sort_values(ascending=False).head(5)
f1_biggest_winners_ids = [driver for driver, race in f1_biggest_winners.index]

# Dataset ready
f1_victories_biggest_winners = f1_victories.query(f"driverId == {f1_biggest_winners_ids}")

# Prepare dataset to plot

cols = ['date', 'driver', 'victories']
winner_drivers = f1_victories_biggest_winners.driver.unique()

colors = {
    'Alain Prost': '#d80005', 
    'Max Verstappen': '#ffffff', 
    'Michael Schumacher': '#f71120',
    'Sebastian Vettel': '#10428e',
    'Lewis Hamilton': '#e6e6e6'
}

winners_history = pd.DataFrame()

# Including other drivers races date (like a cross join matrix, 
# but cosidering column "victories" in a shift operation) 
for driver in winner_drivers:
    # Current driver victories
    driver_history = f1_victories_biggest_winners.query(f"driver == '{driver}'")[cols]
    
    # Other drivers list
    other_drivers = winner_drivers[winner_drivers != driver]
    other_drivers = list(other_drivers)
    
    # Other drivers victories
    other_driver_history = f1_victories_biggest_winners.query(f"driver == {other_drivers}")[cols]
    
    # Renaming other drivers victories to current driver
    other_driver_history['driver'] = driver
    
    # This isn't current driver victory, so receive zero to "shift" operation
    other_driver_history['victories'] = 0    
    
    driver_history = pd.concat([driver_history, other_driver_history])

    driver_history['color'] = colors[driver]
    
    # Sorting by date to correct "shift" operation
    driver_history.sort_values(by='date', inplace=True)
    
    # Reset index to get the last row (index-1) when necessary
    driver_history.reset_index(inplace=True)
    
    # Iterating each row for remain current driver victory when 
    # race date isn't the current driver victory
    for index, row in driver_history.iterrows():
        if not row['victories'] and index-1 > 0:
            driver_history.loc[index, 'victories'] = driver_history.loc[index-1, 'victories']
        
    # Plot dataset ready
    winners_history = pd.concat([winners_history, driver_history])   

#%% [markdown]
# Plotting the top 5 drivers 
import plotly.graph_objects as go
fig = go.Figure()

fig = px.bar(
    winners_history, 
    x='victories', 
    y='driver',
    color='driver',
    color_discrete_sequence=winners_history.color.unique(),
    orientation='h',
    animation_frame="date",
    animation_group="driver",
)

# Bar border line color
fig.update_traces(dict(marker_line_width=1, marker_line_color="black"))

# X axis range
fig.update_layout(xaxis=dict(range=[0, 100]))

# Setting title
fig.update_layout(title_text="Race wins in F1 history between the top 5 winners drivers")

# Animation: Buttons labels and animation duration speed
fig.update_layout(
    updatemenus = [
        {
            "buttons": [
                # Play
                {
                    "args": [
                        None, 
                        {
                            "frame": {
                                "duration": 100, 
                                 "redraw": False
                            }, 
                            "fromcurrent": True,
                            "transition": {
                                "duration": 100, 
                                "easing": "linear"
                            }
                        }
                    ],
                    "label": "Play",
                    "method": "animate"
                },
                # Pause
                {
                    "args": [
                        [None], 
                        {
                            "frame": {
                                "duration": 0, 
                                "redraw": False
                            },
                            "mode": "immediate",
                            "transition": {
                                "duration": 0
                            }
                        }
                    ],
                    "label": "Pause",
                    "method": "animate"
                }
            ]
        }
    ]
)

fig.show()

# %%[markdown]
# # Lap Time Changes over the Years

#%%
merged_data = pd.merge(races, lap_times, on='raceId')
merged_data = pd.merge(merged_data, drivers, on='driverId')
average_lap_time_by_race = merged_data.groupby(['year','name'])['milliseconds'].mean().reset_index()

target_race_name = 'Italian Grand Prix'
start_year = 1950
end_year = 2023

# Filter data for the specified race and time range
filtered_data = merged_data[(merged_data['name'] == target_race_name) & (merged_data['year'].between(start_year, end_year))]

# Calculate average lap time for the specified race
average_lap_time_silverstone = filtered_data.groupby(['year', 'name'])['milliseconds'].mean().reset_index()

# Plotting
plt.figure(figsize=(16, 6))
sns.lineplot(x='year', y='milliseconds', data=average_lap_time_silverstone)
plt.xlabel('Year')
plt.ylabel('Average Lap Time (milliseconds)')
plt.title(f'Trend of Average Lap Times in {target_race_name} from {start_year} to {end_year}')
plt.show()
# plt.figure(figsize=(16, 6))
# sns.lineplot(x='year', y='milliseconds', data=average_lap_time_by_race)
# plt.xlabel('Year')
# plt.ylabel('Average Lap Time (milliseconds)')
# plt.title('Trend of Average Lap Times in Races Over the Years')
# plt.show()

print("The average lap time has been decreasing over the years.")
print("This is due to the technological advances in the cars and the tracks.")

# %%
merged_data = pd.merge(races, lap_times, on='raceId')
merged_data['average_speed'] = merged_data['milliseconds'] / merged_data['milliseconds'].max() 
average_speed_by_track = merged_data.groupby('circuitId')['average_speed'].mean().reset_index()
average_speed_by_track = pd.merge(average_speed_by_track, circuits, left_on='circuitId', right_on='circuitId')
worst_tracks = average_speed_by_track.sort_values(by='average_speed')

plt.figure(figsize=(16, 6))
plt.bar(worst_tracks['name'], worst_tracks['average_speed'], color='purple')
plt.xlabel('Circuit')
plt.ylabel('Average Race Speed')
plt.title('Worst Tracks Based on Average Race Speed')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()

print("The worst track is Monaco.")
print("This is due to the narrow roads and sharp turns.")

# %%[markdown]

# # Lap Time Analysis

# Objective

# Questions to answer:
# * How have lap times changed over the years?
# * How do lap times change over the course of a season?
# * How do lap times change through an entire race?
# * When can we expect to see the fastest lap in a race?

# %%
# Read data

circuits = pd.read_csv('./data/circuits.csv', index_col=0, na_values=r'\N')
constructorResults = pd.read_csv('./data/constructor_results.csv', index_col=0, na_values=r'\N')
constructors = pd.read_csv('./data/constructors.csv', index_col=0, na_values=r'\N')
constructorStandings = pd.read_csv('./data/constructor_standings.csv', index_col=0, na_values=r'\N')
drivers = pd.read_csv('./data/drivers.csv', index_col=0, na_values=r'\N')
driverStandings = pd.read_csv('./data/driver_standings.csv', index_col=0, na_values=r'\N')
lapTimes = pd.read_csv('./data/lap_times.csv')
pitStops = pd.read_csv('./data/pit_stops.csv')
qualifying = pd.read_csv('./data/qualifying.csv', index_col=0, na_values=r'\N')
races = pd.read_csv('./data/races.csv', na_values=r'\N')
results = pd.read_csv('./data/results.csv', index_col=0, na_values=r'\N')
seasons = pd.read_csv('./data/seasons.csv', index_col=0, na_values=r'\N')
status = pd.read_csv('./data/status.csv', index_col=0, na_values=r'\N')

#%%
# Post-read in formatting
circuits = circuits.rename(columns={'name':'circuitName','location':'circuitLocation','country':'circuitCountry','url':'circuitUrl'})
drivers = drivers.rename(columns={'number':'driverNumber','nationality':'driverNationality','url':'driverUrl'})
drivers['driverName'] = drivers['forename']+' '+drivers['surname']
constructors = constructors.rename(columns={'name':'constructorName','nationality':'constructorNationality','url':'constructorUrl'})
#races['date'] = races['date'].apply(lambda x: dt.datetime.strptime(x,'%d/%m/%y'))
races = races.rename(columns={'year':'raceYear','name':'raceName','date':'raceDate','time':'raceTime','url':'raceUrl','round':'raceRound'})
lapTimes = lapTimes.rename(columns={'time':'lapTime','position':'lapPosition','milliseconds':'lapMilliseconds'})
lapTimes['lapSeconds'] = lapTimes['lapMilliseconds'].apply(lambda x: x/1000)
pitStops = pitStops.rename(columns={'time':'pitTime','milliseconds':'pitMilliseconds'})
pitStops['pitSeconds'] = pitStops['pitMilliseconds'].apply(lambda x: x/1000)
results = results.rename(columns={'position':'resultsPosition','time':'resultsTime','milliseconds':'resultsMilliseconds','number':'resultsNumber'})
results['resultsSeconds'] = results['resultsMilliseconds'].apply(lambda x: x/1000)


# %%
# Constructor color mapping
constructor_color_map = {
    'Toro Rosso':'#0000FF',
    'Mercedes':'#6CD3BF',
    'Red Bull':'#1E5BC6',
    'Ferrari':'#ED1C24',
    'Williams':'#37BEDD',
    'Force India':'#FF80C7',
    'Virgin':'#c82e37',
    'Renault':'#FFD800',
    'McLaren':'#F58020',
    'Sauber':'#006EFF',
    'Lotus':'#FFB800',
    'HRT':'#b2945e',
    'Caterham':'#0b361f',
    'Lotus F1':'#FFB800',
    'Marussia':'#6E0000',
    'Manor Marussia':'#6E0000',
    'Haas F1 Team':'#B6BABD',
    'Racing Point':'#F596C8',
    'Aston Martin':'#2D826D',
    'Alfa Romeo':'#B12039',
    'AlphaTauri':'#4E7C9B',
    'Alpine F1 Team':'#2293D1'
}

# %%
resultsAnalysis = pd.merge(results,races,left_on='raceId',right_on='raceId',how='left')
resultsAnalysis = pd.merge(resultsAnalysis,circuits,left_on='circuitId',right_index=True,how='left')
resultsAnalysis = pd.merge(resultsAnalysis,constructors,left_on='constructorId',right_index=True,how='left')
resultsAnalysis = pd.merge(resultsAnalysis,drivers,left_on='driverId',right_index=True,how='left')
resultsAnalysis

# %%
lapTimesAnalysis = pd.merge(lapTimes,races,left_on='raceId',right_on='raceId',how='left')
lapTimesAnalysis = pd.merge(lapTimesAnalysis,resultsAnalysis,left_on=['raceId','driverId','raceYear','raceRound','circuitId','raceName','raceUrl'],right_on=['raceId','driverId','raceYear','raceRound','circuitId','raceName','raceUrl'],how='left')
lapTimesAnalysis

# %%[markdown]
# Lap Times Exploratory Analysis
## Lap Times over the Years

# %%
# Convert 'lapSeconds' to total seconds
circuitName = lapTimesAnalysis['circuitName'].unique()[10]
#%%
df = lapTimesAnalysis[(lapTimesAnalysis['circuitName']==circuitName)].groupby(by=['raceYear','constructorName']).mean().reset_index()
#%%
# create figure
fig = px.line(
    df,
    x='raceYear',
    y='lapSeconds',
    color='constructorName',
    color_discrete_map=constructor_color_map,
)

fig.update_layout(
    title_text=f'Lap Time Trend by Constructor - {circuitName}',
)

fig.update_traces(opacity=0.65)
fig.show()

# %%[markdown]
## Lap Times Over the Course of a Season (by Circuit)

# %%
year = 2021
print(f'Year: {year}')
circuitName = lapTimesAnalysis['circuitName'].unique()[5]
driverList = lapTimesAnalysis[(lapTimesAnalysis['raceYear']==year)]['driverName'].unique()

#%%
df = lapTimesAnalysis[(lapTimesAnalysis['raceYear']==year)&(lapTimesAnalysis['driverName'].isin(driverList))].groupby(by=['circuitName','raceYear','driverName']).mean().reset_index().sort_values(by='raceId')[['raceYear','lap','lapSeconds','driverName','circuitName']]

fig = px.line(
    data_frame = df,
    x='circuitName',
    y='lapSeconds',
    color='driverName',
)

fig.update_layout(
    title_text=f'Average Lap Times by Circuit - {year}',
)
fig.show()

# %%[markdown]
## Lap Times over the course of a Race

cname='Yas Marina Circuit'
df = lapTimesAnalysis[(lapTimesAnalysis['raceYear']==year)&(lapTimesAnalysis['circuitName']==cname)&(lapTimesAnalysis['driverName'].isin(driverList))][['raceYear','lap','lapSeconds','driverName','circuitName','constructorName']]

fig = px.line(
    data_frame = df,
    x='lap',
    y='lapSeconds',
    color='driverName',
)

fig.update_layout(
    title_text=f'Lap Time by Lap - {cname} {year}',
)
fig.show()

# %%[markdown]
## Lap time distribution over the course of a race

# %%
df = lapTimesAnalysis[(lapTimesAnalysis['raceYear']==year)&(lapTimesAnalysis['circuitName']==cname)]

# create figure
fig = px.histogram(df,
                   x='lapSeconds',
                  color='constructorName',
                  color_discrete_map=constructor_color_map,
                  )

fig.update_layout(
    title_text=f'Lap Time Distribution by Constructor - {year} {cname}',
    barmode='overlay',
)

fig.update_traces(opacity=0.65)
fig.show()

# %%[markdown]
## Average Historical Lap Times by Circuit

# %%
fig = px.line(lapTimesAnalysis.groupby(by=['circuitName','raceYear',]).mean().reset_index(),
                 x='raceYear',
                 y='lapSeconds',
                 color='circuitName',
                )
fig.update_layout(
    title_text='Average Lap Times Over Time by Circuit',
)
fig.show()


# %%[markdown]
## Where in the race can we expect to see the fastest lap in a race?

# %%
df_temp = pd.merge(lapTimesAnalysis.groupby(by=['circuitName']).mean().sort_values(by='fastestLap',ascending=True)['fastestLap'],lapTimesAnalysis.groupby(by=['circuitName']).max().sort_values(by='fastestLap',ascending=True)['laps'],left_index=True,right_index=True, how='inner')
df_temp['fastestLapRacePct'] = df_temp['fastestLap']/df_temp['laps']*100
print(df_temp['fastestLapRacePct'].mean())
df_temp

# %%[markdown]
# create figure
fig = px.histogram(df_temp,
                   x='fastestLapRacePct',
                  )

fig.update_layout(
    title_text=f'Fastest Lap Time Lap Distribution',
    barmode='overlay',
)

fig.update_traces(opacity=0.65)
fig.show()

# %%[markdown]
# Conclusions

# We set out initially to explore the lap time data of the F1 races available in the dataset. Going temporally through the year does not glean much information since lap time data is highly dependent on the circuit being raced on. Fixing the track, one can see slightly more meaningful result when comparing them year to year. 

# ## How have lap times changed over the years?
# Generally speaking the lap times decrease over time, but have behaviours where lap times jump up. Looking at the behaviour, it leads me to believe they are step changes related to rule changes in F1. In particular, you can see a meaningful increase in the lap times especially going from 2021 to 2022 in large part to the major design changes implemented.

# ## How do lap times change over the course of a season?
# Yes. The changes however are largely due the track and not much can really be said about this particular question.

# ## How do lap times change through an entire race?
# As expected, lap times decrease as we progress through the race as fuel weight goes down.

# ## When can we expect to see the fastest lap in a race?
# The fastest lap typically occurs about 74% into a race. For good reason, the cars are running on lower fuel and depending on the current standings teams opt different strategies to gain the extra point from having the fastest lap in the race.

#%%[markdown]
# # Data Wrangling

# %%[markdown]
# Let's take a look at the data we have available to us.
# %%
circuits = pd.read_csv('./data/circuits.csv', na_values='\\N')
constructor_results = pd.read_csv('./data/constructor_results.csv', na_values='\\N')
constructor_standings = pd.read_csv('./data/constructor_standings.csv', na_values='\\N')
constructors = pd.read_csv('./data/constructors.csv', na_values='\\N')
driver_standings = pd.read_csv('./data/driver_standings.csv', na_values='\\N')
drivers = pd.read_csv('./data/drivers.csv', na_values='\\N')
lap_times = pd.read_csv('./data/lap_times.csv', na_values='\\N')
pit_stops = pd.read_csv('./data/pit_stops.csv', na_values='\\N')
qualifying = pd.read_csv('./data/qualifying.csv', na_values='\\N')
races = pd.read_csv('./data/races.csv', na_values='\\N')
results = pd.read_csv('./data/results.csv', na_values='\\N')
seasons = pd.read_csv('./data/seasons.csv', na_values='\\N')
sprint_results = pd.read_csv('./data/sprint_results.csv', na_values='\\N')
status = pd.read_csv('./data/status.csv', na_values='\\N')

# %%[markdown]
# ## Check for na values
# %%
# Now Lets look at the data
def check_for_data(data):
    """
    Prints the number of missing values in each column of the given DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame to check for missing values.

    Returns:
    None
    """
    print(data.isna().sum())
check_for_data(races)

# %%[markdown]
# ## Fetaure Engineering

# %%
data = pd.merge(results, races, on='raceId')
data = pd.merge(data, drivers, on='driverId')
data

#%%
# Drop posterior data column (time, milliseconds, fastestLap, fastestLapTime, fastestLapSpeed, statusId)
posterior_data = ['laps', 'milliseconds', 'fastestLap', 'fastestLapTime', 'fastestLapSpeed', 'statusId', 'time_x',
                  'time_y', 'positionOrder']

data = data.drop(posterior_data, axis=1)
data

#%%
#Drop the columns which are not required
data = data.drop(columns=['position', 'positionText', 'number_x', 'sprint_date', 'sprint_time', 'driverRef', 'number_y',
                          'nationality', 'url_x', 'url_y', 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'quali_date',
                          'quali_time', 'fp3_date', 'fp3_time', 'name'])
data

#%%
#Lets check for null values again just to be sure
data.isna().sum()
#TODO: Fix the null values

#%%[markdown]
# For Our model we have too many features, lets try to reduce the number of features. 
# We will do this by using our domain knowledge and intuition to select the features which we think are important.
# We will crete some new features as well based on our domain knowledge which will make more sense to predict the Wins for a driver.

#%%[markdown]
# We will calculate the age of driver using date of race and date of birth.
#%%
#Change data type from string to datetime
data['dob'] = pd.to_datetime(data['dob'])
data['date'] = pd.to_datetime(data['date'])
#Add age column to dataframe
dates = data['date'] - data['dob']
age = dates.dt.days / 365
data['age'] = round(age)
data

#%%[markdown]
# 
# winRate - Likelihood of winning a race 
# This feature 

#%%
# INITIALISATION
df_driver = drivers.copy()
df_driver['totalWins'] = 0

race_dates = races[['raceId', 'date']]
# CLEANING AND ADDING

# adding dates to each race
driver_standings = driver_standings.merge(race_dates[['raceId', 'date']], how='left', on='raceId')

# Convert the "date" column to a datetime object
driver_standings['date'] = pd.to_datetime(driver_standings['date'])

# Create a new column 'year' to extract the year from the 'date' column
driver_standings['year'] = driver_standings['date'].dt.year
# driver_standings_csv.head()

# count the number of races each driver has driven in
num_races_per_driver = driver_standings.groupby('driverId')['raceId'].nunique()
num_races_per_driver_df = num_races_per_driver.reset_index()
num_races_per_driver_df = num_races_per_driver_df.rename(columns={'raceId': 'totalRaces'})
# num_races_per_driver_df.head()

# FINDING TOTAL WINS FOR EACH DRIVER
for index, row in df_driver.iterrows():
    driverId = row['driverId']

    # filtering out rows with ['driverId'] == driverId
    driver_standings_csv_driverId = driver_standings[driver_standings['driverId'] == driverId]

    # Group the dataframe by year and find the maximum date for each year
    latest_day_in_year = driver_standings_csv_driverId.groupby(driver_standings_csv_driverId['date'].dt.year)[
        'date'].max()

    # Use the latest day in each year to filter the original dataframe
    filtered_dataframe = driver_standings_csv_driverId.loc[
        driver_standings_csv_driverId['date'].isin(latest_day_in_year)]

    total_wins = filtered_dataframe['wins'].sum()

    index = df_driver.index[df_driver['driverId'] == driverId].tolist()[0]
    df_driver.at[index, 'totalWins'] = total_wins

# adding dates to each race
df_driver = df_driver.merge(num_races_per_driver_df, how='left', on='driverId')

# calculate win rate and drop totalWins columns
df_driver['winRate'] = df_driver['totalWins'] / df_driver['totalRaces']
df_driver = df_driver.drop(['totalWins'], axis=1)
df_driver['dob'] = pd.to_datetime(df_driver['dob'])
# df_driver['age'] = 2023 - df_driver['dob'].dt.year

# PRINT CURRENT DATASET    
df_driver.head()

#%%[markdown]
# fastestLapRate - Likelihood of winning fastest lap

#%%
# Group the dataframe by raceId and find the index of the row with the minimum milliseconds
idx = lap_times.groupby('raceId')['milliseconds'].idxmin()

# Use the index to select the rows with the minimum milliseconds for each raceId
df_min_milliseconds = lap_times.loc[idx]

# Sort the result by raceId
df_min_milliseconds.sort_values('raceId', inplace=True)

counts = pd.DataFrame(df_min_milliseconds['driverId'].value_counts())
counts.columns = ['totalFastestLaps']
counts['driverId'] = counts.index
counts.reset_index(drop=True, inplace=True)

# adding totalFastestLaps to df maindata_wnames
df_driver = df_driver.merge(counts, how='left', on='driverId')
df_driver = df_driver.fillna(0)

# calculate fastest lap rate and drop totalFastestLaps
df_driver['fastestLapRate'] = df_driver['totalFastestLaps'] / df_driver['totalRaces']
df_driver = df_driver.drop(['totalFastestLaps'], axis=1)

# PRINT CURRENT DATASET    
df_driver.head()

#%%[markdown]
# qualifyingWinRate - Likelihood of winning qualifying

#%%
# COUNTING THE NUMBER OF QUALIFYING WINS
# Group by driverId and position, then count the number of occurrences
position_1_counts = qualifying[qualifying['position'] == 1].groupby('driverId')['position'].count().reset_index()

# Rename the 'position' column to 'position_1_count'
position_1_counts = position_1_counts.rename(columns={'position': 'position_1_count'})

# # Print the resulting DataFrame
# position_1_counts.head()

# merge
df_driver = df_driver.merge(position_1_counts, how='left', on='driverId')
df_driver = df_driver.fillna(0)

# calculate fastest lap rate and drop totalFastestLaps
df_driver['qualifyingWinRate'] = df_driver['position_1_count'] / df_driver['totalRaces']
df_driver = df_driver.drop(['position_1_count'], axis=1)

# filling in NaN values for qualifyingWinRate
df_driver.fillna(0, inplace=True)

df_driver.head()

# #%%[markdown]
# # podiumRate - Likelihood of finishing in top 3

# #%%
# # COUNTING THE NUMBER OF PODIUMS
# # Group by driverId and position, then count the number of occurrences
# position_1_counts = results[results['position'] == 1].groupby('driverId')['position'].count().reset_index()

# # Rename the 'position' column to 'position_1_count'
# position_1_counts = position_1_counts.rename(columns={'position': 'position_1_count'})

# # # Print the resulting DataFrame
# position_1_counts.head()
#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor
features = ['driverId', 'winRate', 'fastestLapRate', 'qualifyingWinRate']
final = df_driver[features]
vif = variance_inflation_factor(final, 3)
vif

#%%[markdown]
# Correlation Matrix



# %%
features = ['driverId', 'winRate', 'fastestLapRate', 'qualifyingWinRate']
sns.heatmap(df_driver[features].corr(), annot=True , cmap='coolwarm' , linewidths=1, linecolor='black')
final = df_driver[features]


# %%
data_new = pd.merge(data, final, on='driverId')


# %%[markdown]
# Now lets clasify race as the first half and second half by a new variable first_half

# %%
driver_result_withdate_divided = data_new.copy()
driver_result_withdate_divided['firstHalf'] = (driver_result_withdate_divided['date'].dt.month <= 6).astype(int)
driver_result_withdate_divided

#%%
#%%
driver_result_withdate_groupby_year_divided = driver_result_withdate_divided.groupby([driver_result_withdate_divided['date'].dt.year, driver_result_withdate_divided['firstHalf'], driver_result_withdate_divided['driverId']])
driver_result_withdate_groupby_year_divided.head()

#%%
driver_result_withdate_groupby_year_divided = driver_result_withdate_divided.groupby([driver_result_withdate_divided['date'].dt.year, driver_result_withdate_divided['firstHalf'], driver_result_withdate_divided['driverId']])
point_year_divided = driver_result_withdate_groupby_year_divided["points"].sum(0).unstack()
point_year_divided
#%%
age_year = driver_result_withdate_groupby_year_divided["age"].mean().unstack()
age_year


#%%
#Loop through each team 
#Column name is ID of each team
df_final = []
id_driver =[]
point_first_half_all = []
whole_year_point_all = []
for column in point_year_divided:
    point_year_driver_divided = point_year_divided[column].unstack()
    point_first_half = []
    whole_year_point = []
    ages = []
    age_one_year = age_year[column].unstack()
    point_year_driver_divided = pd.merge(point_year_driver_divided, age_one_year, on='date')
    #Loop through each year
    for row in point_year_driver_divided.iterrows():
        if not np.isnan(row[1][0]) and not np.isnan(row[1][1]) and (not np.isnan(row[1][2]) or not np.isnan(row[1][3])):
            if not np.isnan(row[1][2]):
                age = row[1][2]
            else:
                age = row[1][3]
            ages.append(age)
            id_driver.append(column)
            point_first_half.append(row[1][1])
            point_first_half_all.append(row[1][1])
            whole_year_point.append(row[1][0]+row[1][1])
            whole_year_point_all.append(row[1][0])

    new_df = pd.DataFrame({'first_half_point':point_first_half, 'ages': ages, 'whole_year_point':whole_year_point, 'id_driver':column})
    df_final.append(new_df)
 
    sns.scatterplot(x='first_half_point', y='whole_year_point', data=new_df)
#%%
df_final = pd.concat(df_final, ignore_index=True)

#%%
df_join = df_final.merge(df_driver, left_on='id_driver', right_on='driverId')
df_join



#%%[markdown]
# # Modelling
#%%[markdown]
# # 1. Linear Regression

#%%
# Ensure the data is in the right format
features = ['first_half_point', 'winRate', 'fastestLapRate', 'qualifyingWinRate']
df_join[features] = df_join[features].apply(pd.to_numeric, errors='coerce')

# Extract the features and target variables
X = df_join[features]
y = df_join['whole_year_point']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Output model performance metrics
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)
print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE): %.2f' % np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score (R^2): %.2f' % r2_score(y_test, y_pred))
print('Mean Absolute Error (MAE): %.2f' % mean_absolute_error(y_test, y_pred))

rmse_lr =  np.sqrt(mean_squared_error(y_test, y_pred))
r2_lr = r2_score(y_test, y_pred)
# Calculate MAPE
mask = y_test != 0
mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
print('Mean Absolute Percentage Error (MAPE): %.2f' % mape)

#%%
# Scatter plot of actual vs predicted values for the test set
plt.scatter(X_test['first_half_point'], y_test, color='blue', label='Actual')
plt.scatter(X_test['first_half_point'], y_pred, color='red', label='Predicted')
plt.xlabel('first_half_point')
plt.ylabel('whole_year_point')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.show()


#%%[markdown]

# # 2. Decision Tree Regression
features = ['first_half_point', 'winRate', 'fastestLapRate', 'qualifyingWinRate']

# Ensure the data is in the right format
df_join[features] = df_join[features].apply(pd.to_numeric, errors='coerce')

# Extract the features and target variables
X = df_join[features]
y = df_join['whole_year_point']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Output model performance metrics
print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE): %.2f' % np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score (R^2): %.2f' % r2_score(y_test, y_pred))
print('Mean Absolute Error (MAE): %.2f' % mean_absolute_error(y_test, y_pred))

# Calculate MAPE
mask = y_test != 0
mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
print('Mean Absolute Percentage Error (MAPE): %.2f' % mape)

# Scatter plot
plt.scatter(X_test['first_half_point'], y_test, color = 'blue')
plt.scatter(X_test['first_half_point'], y_pred, color = 'red')
plt.xlabel('first_half_point')
plt.ylabel('Predicted whole_year_point')
plt.show()

rmse_dt =  np.sqrt(mean_squared_error(y_test, y_pred))
r2_dt = r2_score(y_test, y_pred)
#%%[markdown]
# # 3. MLP
driver_result_withdate_divided = driver_result_withdate_divided[driver_result_withdate_divided["points"]<=20]
driver_ids = driver_result_withdate_divided["driverId"].unique()
last_ten_race_results=[]
current_race_results=[]
for driver_id in driver_ids:
    temp_df = driver_result_withdate_divided[driver_result_withdate_divided["driverId"]==driver_id]
    temp_df = temp_df.sort_values('date')
    if(temp_df.shape[0]<=50):
        continue
    for i in range(51, temp_df.shape[0]):
        last_ten_race_results.append(temp_df.iloc[i-51:i-11]['points'].tolist()+temp_df.iloc[i-10:i]['age'].tolist()+temp_df.iloc[i-10:i]['round'].tolist()+temp_df.iloc[i-10:i]['grid'].tolist()+temp_df.iloc[i-10:i]['raceId'].tolist())
        current_race_results.append(temp_df.iloc[i-10:i]['points'].mean())

#Check x, y size
assert len(last_ten_race_results)==len(current_race_results)
#Split train and test set
train_x, test_x, train_y, test_y = train_test_split(last_ten_race_results, current_race_results, train_size=0.8)
#Train Model
mlp_regressor = MLPRegressor(hidden_layer_sizes=(30, 20, 30,10,20,40), max_iter=300)
mlp_regressor.fit(train_x, train_y)
train_y_hat = mlp_regressor.predict(train_x)
# Predict on test set
test_y_pred = mlp_regressor.predict(test_x)
# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset")
print("Accuracy score \t:", mlp_regressor.score(train_x, train_y))
print("Mean Squared Error (MSE):", mean_squared_error(train_y_hat, train_y))
print()

rmse_mlp =  np.sqrt(mean_squared_error(train_y_hat, train_y))
r2_mlp = r2_score(train_y_hat, train_y)
# Check the Goodness of Fit and Prediction Accuracy (on Test Data)
print("Goodness of Fit of Model and Prediction Accuracy \tTest Dataset")
print("Accuracy score\t:", mlp_regressor.score(test_x, test_y))

# Prediction on test data
test_y_hat = mlp_regressor.predict(test_x)
# Create a scatter plot
plt.figure(figsize=(10, 7))
plt.scatter(test_y, test_y_hat, alpha=0.5)
plt.title("MLP Regressor: predicted vs. actual")
plt.xlabel("Actual points")
plt.ylabel("Predicted points")
plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], color='red')  # Diagonal line
plt.show()

#%%[markdown]
# # 4. KNN Regressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


### Finding the best K value using Elbow method
# Assuming last_ten_race_results and current_race_results are already defined

# Initialize a list to store MSE values for different values of k
mse_values = []

# Try different values of k
for k in range(1, 21):  # You can adjust the range based on your requirements
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(train_x, train_y)
    y_pred = knn_regressor.predict(test_x)
    mse = mean_squared_error(test_y, y_pred)
    mse_values.append(mse)

# Plot the elbow curve
plt.plot(range(1, 21), mse_values, marker='o')
plt.title('Elbow Method for k-NN Regression')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error (MSE)')
plt.show()

features = ['first_half_point', 'winRate', 'fastestLapRate', 'qualifyingWinRate']

# Ensure the data is in the right format
df_join[features] = df_join[features].apply(pd.to_numeric, errors='coerce')

# Extract the features and target variables
X = df_join[features]
y = df_join['whole_year_point']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a k-NN Regressor with a specified number of neighbors (e.g., 5)
k_neighbors = 5
model = KNeighborsRegressor(n_neighbors=k_neighbors)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Output model performance metrics
print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE): %.2f' % np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score (R^2): %.2f' % r2_score(y_test, y_pred))
print('Mean Absolute Error (MAE): %.2f' % mean_absolute_error(y_test, y_pred))

rmse_knn =  np.sqrt(mean_squared_error(y_test, y_pred))
r2_knn = r2_score(y_test, y_pred)
# Calculate MAPE
mask = y_test != 0
mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
print('Mean Absolute Percentage Error (MAPE): %.2f' % mape)

# Scatter plot
plt.scatter(X_test['first_half_point'], y_test, color='blue', label='Actual')
plt.scatter(X_test['first_half_point'], y_pred, color='red', label='Predicted')
plt.xlabel('first_half_point')
plt.ylabel('Predicted whole_year_point')
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

def compare_models(met,label):
    """
    Plots a line plot comparing the MSE and R2 values of different models.

    Parameters:
    met (list): A list of RMSE and R2 values for different models.
    label (string): Name of the metric to be used as the y-axis label.

    Returns:
    None
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(range(len(met)), met, label=label)
    plt.plot(met, label=label)
    plt.xticks(range(len(met)), ['Linear Regression', 'Decision Tree', 'MLP','KNN'])
    plt.title(f'Comparing Models using {label} metric')
    plt.xlabel("Models")
    plt.ylabel("Values")
    plt.legend()
    plt.show()

# %%
rmse = [rmse_lr, rmse_dt, rmse_mlp,rmse_knn]
r2 = [r2_lr, r2_dt, r2_mlp,r2_knn]

compare_models(rmse, 'RMSE')
compare_models(r2, 'R2')

# %% [markdown]
# # Summary

# 1. Lap time decreased over time.
#
# 2. Fastest Lap occurs after 70% of the race is over.
#
# 3. Linear Model gave the best R2 score. 
#
# 4. MLP regressor gave the least RMSE score.
#
# 5. While modelling we observed that due to introduction of new features, we might need to further tune the models for better performance metrics.


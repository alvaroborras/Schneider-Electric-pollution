
In this competition, we are given meteorological data from several stations in Seoul, collecting hourly atmospherical data for different pollutants.

Each station is located in a different region in Seoul, and the initial and final times for the registered data slightly differ.

For each station, we are given its precise location, as well as the status of their measurement equipment.
We are given for each measurement, the status information about the sensor, 


# Approach

This competition consisted of 3 tasks.

- ### **Task 1** 
    **Basic EDA and data manipulation**
    - **Q1**: Average daily SO<sub>2</sub> concentration across all districts over the entire period. Give the station average. Provide the answer with 5 decimals.

    - **Q2**: Analyse how pollution levels vary by season. Return the average levels of CO per season at the station 209. (Take the whole month of December as part of winter, March as spring, and so on.)

    - **Q3**: Which hour presents the highest variability (Standard Deviation) for the pollutant O<sub>3</sub>? Treat all stations as equal.

    - **Q4**: Which is the station code with more measurements labeled as "Abnormal data"?

    - **Q5**: Which station code has more "not normal" measurements ($\neq 0$)?

    - **Q6**: Return the count of `Good`, `Normal`, `Bad` and `Very bad` records for all the station codes of PM<sub>2.5</sub> pollutant.

    All the questions are more or less straight forward and can be computed using only Pandas.

- ### **Task 2**
    **Predict hourly pollutant concentrations for the following stations and periods, assuming error-free measurements**

        Station code: 206 | pollutant: SO2   | Period: 2023-07-01 00:00:00 - 2023-07-31 23:00:00
        Station code: 211 | pollutant: NO2   | Period: 2023-08-01 00:00:00 - 2023-08-31 23:00:00
        Station code: 217 | pollutant: O3    | Period: 2023-09-01 00:00:00 - 2023-09-30 23:00:00
        Station code: 219 | pollutant: CO    | Period: 2023-10-01 00:00:00 - 2023-10-31 23:00:00
        Station code: 225 | pollutant: PM10  | Period: 2023-11-01 00:00:00 - 2023-11-30 23:00:00
        Station code: 228 | pollutant: PM2.5 | Period: 2023-12-01 00:00:00 - 2023-12-31 23:00:00

    
    In this task, we are asked to make several predictions. For each one, we should predict the concentration of a pollutant for a specific station, over the month after the last measurement.

    Initially, I thought this was a pure time series forecasting problem. I implemented a SARIMA with some feature engineering, but the score was not as good as I expected. Some of the pollutants exhibit large fluctuations due to their high volatility, so an accurate forecast would be hard.

    Since the data available for each station is given over different periods of time, we can use the measurements from neighboring stations to obtain an accurate estimate of the pollutant at the current station.
    For example, for station 206 we must report the values of SO<sub>2</sub> in July 2023. In this same month, there are other nearby stations still measuring SO<sub>2</sub>, so we can use those to predict the values that would be measured at station 206.

    My final solution for this task is simply a weighted average of the measurements of nearby stations.
    For each station and pollutant for which we need to make the predictions, I average the valid measurements in the nearest stations, which I find using their geographical coordinates.

    This task would have been significantly harder if the last measurement date was the same for all pollutants and stations. Then it would have become a pure time series forecasting problem.

- ### **Task 3**
    **Detect instrument anomalies for the following stations and periods**

        Station code: 205 | pollutant: SO2   | Period: 2023-11-01 00:00:00 - 2023-11-30 23:00:00
        Station code: 209 | pollutant: NO2   | Period: 2023-09-01 00:00:00 - 2023-09-30 23:00:00
        Station code: 223 | pollutant: O3    | Period: 2023-07-01 00:00:00 - 2023-07-31 23:00:00
        Station code: 224 | pollutant: CO    | Period: 2023-10-01 00:00:00 - 2023-10-31 23:00:00
        Station code: 226 | pollutant: PM10  | Period: 2023-08-01 00:00:00 - 2023-08-31 23:00:00
        Station code: 227 | pollutant: PM2.5 | Period: 2023-12-01 00:00:00 - 2023-12-31 23:00:00


    For each station, we are given the status of the instrument. Usually, the measurement is `Normal`, meaning that there are no issues with the data. However, it has other possible states:

        0: "Normal"
        1: "Need for calibration"
        2: "Abnormal"
        4: "Power cut off"
        8: "Under repair"
        9: "Abnormal data"

    We should predict for the requested stations and pollutants, the hourly state of the instrument during a specific month.

    This problem is considerably harder than the previous one. For the requested stations, we don't have measurement data during the prediction timeframe, so we can't detect anomalies from the data in the same station.

    Using neighboring data like in Task 2 is not straight forward, since the state of each station is more or less independent. I ended up training a set LightGBM Classifiers to predict the status for each instrument. I only considered the 3 nearest stations for feature generation.
    Finally, I did some postprocessing in the predictions in order to ensure that the patterns for the status were similar to the historical ones.

    I wanted to train a custom LSTM, which would take as input the lagged status of each pollutant in the current station and would forecast the multiclass probabilities for a specific hour, but I didn't have enough time to implement it.

## Final score

<div align="center">
  <img src="img/final_score.png" alt="Final Score" width="30%">
</div>


## Final leaderboard

<div align="center">
  <img src="img/final_leaderboard.png" alt="Final Score">
</div>

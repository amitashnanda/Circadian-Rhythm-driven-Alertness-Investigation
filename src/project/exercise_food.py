import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import row, column
from bokeh.io import output_notebook, show, save
from bokeh.models import CategoricalColorMapper
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource, Range1d, FactorRange
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6, Spectral10, Spectral4
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import cumsum
from math import pi
from bokeh.palettes import Category20c
import plotly.graph_objects as go
import plotly.express as px
import warnings
from bokeh.models.glyphs import Text
from bokeh.transform import dodge
import plotly.express as px
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objs as go
path = ('../data/data.csv')
df_s2 = pd.read_csv(path)
 

class exercise_food:
    def __init__(self):
        self
    def function1():
        path = ('../data/data.csv')
        df_s2 = pd.read_csv(path)
        df_exercise = df_s2 [['L5VALUE', 'M10VALUE_daybefore', 'Morning Alertness']]
        # count the number of occurrences of each value in the M10VALUE_daybefore column
        counts = df_exercise['M10VALUE_daybefore'].value_counts()

        # check the percentage of people above 100
        percent_above_100 = (counts[counts.index > 100].sum() / counts.sum()) * 100

        print(f"{percent_above_100:.2f}% of people have an M10 value above 100")
        
        df_exercise = df_exercise[df_exercise['M10VALUE_daybefore'] <= 100]
        # Only less than 0.01% of people did above this so we can exclude
        print(df_exercise['L5VALUE'].max())
        print(df_exercise['L5VALUE'].min())
        print(df_exercise['L5VALUE'].mean())
        
        counts = df_exercise['L5VALUE'].value_counts()

        # check the percentage of people above 30
        percent_above_30 = (counts[counts.index > 30].sum() / counts.sum()) * 100

        print(f"{percent_above_30:.2f}% of people have an L5 value above 30")
        
        # create the scatter plot
        fig = px.scatter(df_exercise, x='M10VALUE_daybefore', y='Morning Alertness')
        # show the plot
        fig.show()

        # Raw data, plotting all M10 values and alertness ratings
        # create the scatter plot
        fig = px.scatter(df_exercise, x='L5VALUE', y='Morning Alertness')
        # show the plot
        fig.show()

        # Raw data, plotting all L5 values and alertness ratings
        # round off M10 values to the nearest integers
        df_exercise['M10'] = df_exercise['M10VALUE_daybefore'].round()

        # group the dataframe by M10 and calculate the mean alertness for each M10 value
        df_grouped = df_exercise.groupby('M10', as_index=False).agg({'Morning Alertness': 'mean'})

        # create the line plot
        fig = px.line(df_grouped, x='M10', y='Morning Alertness')

        # fit a linear regression model
        model = sm.OLS(df_grouped['Morning Alertness'], sm.add_constant(df_grouped['M10'])).fit()

        # add the trendline to the plot
        fig.add_traces(
            go.Scatter(x=df_grouped['M10'], y=model.predict(), mode='lines', 
                    line=dict(color='red', width=3), name='Trendline')
        )

        # show the plot
        fig.show()


        # This trend shows higher the M10 value the day before the better alertness the person has
        # M10 is associated with more excersize
        
        # round off L5VALUE values to the nearest integers
        df_exercise['L5'] = df_exercise['L5VALUE'].round()

        # group the dataframe by L5 and calculate the mean alertness for each L5 value
        df_grouped = df_exercise.groupby('L5', as_index=False).agg({'Morning Alertness': 'mean'})

        # create the line plot
        fig = px.line(df_grouped, x='L5', y='Morning Alertness')

        # fit a linear regression model
        model = sm.OLS(df_grouped['Morning Alertness'], sm.add_constant(df_grouped['L5'])).fit()

        # add the trendline to the plot
        fig.add_traces(
            go.Scatter(x=df_grouped['L5'], y=model.predict(), mode='lines', 
                    line=dict(color='red', width=3), name='Trendline')
        )

        # show the plot
        fig.show()

        # This trend shows higher the L5 value the day before the lower the alertness the person has
        # L5 is associated with more movement at the 5 least active hours of the day which includes the sleeping time
        # round off M10 values to the nearest integers using numpy
        df_exercise['M10'] = np.round(df_exercise['M10VALUE_daybefore']).astype(int)

        # create the bins and assign them to the M10 values
        bins = pd.IntervalIndex.from_tuples([(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (60, 70), (80, 90), (90, 100)])
        df_exercise['M10_bin'] = pd.cut(df_exercise['M10'], bins=bins)

        # create a new column 'M10_bin_mid' with the midpoint of each bin interval
        df_exercise['M10_bin_mid'] = df_exercise['M10_bin'].apply(lambda x: x.mid)

        # group the dataframe by M10_bin and calculate the mean alertness for each bin
        df_grouped = df_exercise.groupby('M10_bin_mid', as_index=False).agg({'Morning Alertness': 'mean'})

        # create the line plot with best-fit line, thicker line, and filled area
        fig = px.line(df_grouped, x='M10_bin_mid', y='Morning Alertness', line_shape='spline')

        # add a trendline to the plot
        fig.add_trace(go.Scatter(x=df_grouped['M10_bin_mid'], y=np.poly1d(np.polyfit(df_grouped['M10_bin_mid'], df_grouped['Morning Alertness'], 1))(df_grouped['M10_bin_mid']), mode='lines', name='Trendline', line=dict(width=3, dash='dot', color='red')))
        # fill the area under the line
        fig.add_trace(go.Scatter(x=df_grouped['M10_bin_mid'], y=df_grouped['Morning Alertness'], mode='lines', name='Filled Area', line=dict(width=0.5, color='rgb(114, 175, 212)'), fill='tonexty'))

        # set the x-axis and y-axis labels
        fig.update_layout(xaxis_title='M10 Values', yaxis_title='Average Alertness', title='Average Alertness by M10 Values')

        # show the plot
        fig.show()



        # This trend shows higher the M10 value the day before the better alertness the person has
        # M10 is associated with more excersize
        
        # create the bins and assign them to the L5VALUE values
        bins = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30)])
        df_exercise['L5VALUE_bin'] = pd.cut(df_exercise['L5VALUE'], bins=bins)

        # create a new column 'L5VALUE_bin_mid' with the midpoint of each bin interval
        df_exercise['L5VALUE_bin_mid'] = df_exercise['L5VALUE_bin'].apply(lambda x: x.mid)

        # group the dataframe by L5VALUE_bin and calculate the mean alertness for each bin
        df_grouped = df_exercise.groupby('L5VALUE_bin_mid', as_index=False).agg({'Morning Alertness': 'mean'})

        # create the line plot with best-fit line, thicker line, and filled area
        fig = px.line(df_grouped, x='L5VALUE_bin_mid', y='Morning Alertness', line_shape='spline')

        # add a trendline to the plot
        fig.add_trace(go.Scatter(x=df_grouped['L5VALUE_bin_mid'], y=np.poly1d(np.polyfit(df_grouped['L5VALUE_bin_mid'], df_grouped['Morning Alertness'], 1))(df_grouped['L5VALUE_bin_mid']), mode='lines', name='Trendline', line=dict(width=3, dash='dot', color='red')))

        # fill the area under the line
        fig.add_trace(go.Scatter(x=df_grouped['L5VALUE_bin_mid'], y=df_grouped['Morning Alertness'], mode='lines', name='Filled Area', line=dict(width=0.5,dash='dot', color='rgb(114, 175, 212)'), fill='tonexty'))

        # set the x-axis and y-axis labels
        fig.update_layout(xaxis_title='L5VALUE', yaxis_title='Average Alertness', title='Average Alertness by L5VALUE')

        # show the plot
        fig.show()


        # This trend shows higher the L5 value the day before the lower the alertness the person has
        # L5 is associated with more movement at the 5 least active hours of the day which includes the sleeping time
        # round off M10 values to the nearest integers using numpy
        df_exercise['M10'] = np.round(df_exercise['M10VALUE_daybefore']).astype(int)

        # create the bins and assign them to the M10 values
        bins = pd.IntervalIndex.from_tuples([(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (60, 70), (80, 90), (90, 100)])
        df_exercise['M10_bin'] = pd.cut(df_exercise['M10'], bins=bins)

        # create a new column 'M10_bin_mid' with the midpoint of each bin interval
        df_exercise['M10_bin_mid'] = df_exercise['M10_bin'].apply(lambda x: x.mid)

        # group the dataframe by M10_bin and calculate the mean and standard deviation of alertness for each bin
        df_grouped = df_exercise.groupby('M10_bin_mid', as_index=False).agg({'Morning Alertness': ['mean', 'std']})

        # rename the columns to remove the multi-level index
        df_grouped.columns = ['M10_bin_mid', 'Morning Alertness', 'stdev_alertness']

        # create the line plot with error bars
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_grouped['M10_bin_mid'],
            y=df_grouped['Morning Alertness'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Average Alertness',
            error_y=dict(
                type='data',
                array=df_grouped['stdev_alertness'],
                visible=True
            )
        ))

        # set the x-axis and y-axis labels
        fig.update_layout(xaxis_title='M10 Values', yaxis_title='Average Alertness', title='Average Alertness by M10 Values')

        # show the plot
        fig.show()



        # This trend shows higher the M10 value the day before the better alertness the person has
        # M10 is associated with more excersize
        
        bins = pd.IntervalIndex.from_tuples([(0, 2.5), (2.5, 5), (5, 7.5), (7.5, 10), (10, 12.5), (12.5, 15), (15, 17.5), (17.5, 20), 
                                     (20, 22.5), (22.5, 25), (25, 27.5), (27.5, 30)])
        df_exercise['L5VALUE_bin'] = pd.cut(df_exercise['L5VALUE'], bins=bins)

        # create a new column 'L5VALUE_bin_mid' with the midpoint of each bin interval
        df_exercise['L5VALUE_bin_mid'] = df_exercise['L5VALUE_bin'].apply(lambda x: x.mid)

        # group the dataframe by L5VALUE_bin and calculate the mean and standard deviation of alertness for each bin
        df_grouped = df_exercise.groupby('L5VALUE_bin_mid', as_index=False).agg({'Morning Alertness': ['mean', 'std']})

        # rename the columns to remove the multi-level index
        df_grouped.columns = ['L5VALUE_bin_mid', 'Morning Alertness', 'stdev_alertness']

        # create the line plot with error bars
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_grouped['L5VALUE_bin_mid'],
            y=df_grouped['Morning Alertness'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Average Alertness',
            error_y=dict(
                type='data',
                array=df_grouped['stdev_alertness'],
                visible=True
            )
        ))

        # set the x-axis and y-axis labels
        fig.update_layout(xaxis_title='L5VALUE bin', yaxis_title='Average Alertness', 
                        title='Average Alertness by L5VALUE bin', 
                        xaxis=dict(tickvals=[1.25, 3.75, 6.25, 8.75, 11.25, 13.75, 16.25, 18.75, 21.25, 23.75, 26.25, 28.75], 
                                    ticktext=['0-2.5', '2.5-5', '5-7.5', '7.5-10', '10-12.5', '12.5-15', '15-17.5', '17.5-20', 
                                            '20-22.5', '22.5-25', '25-27.5', '27.5-30']))

        # show the plot
        fig.show()


        # This trend shows higher the L5 value the day before the lower the alertness the person has
        # L5 is associated with more movement at the 5 least active hours of the day which includes the sleeping time
        
    def function2():
        path = ('../data/data.csv')
        df_s2 = pd.read_csv(path)
        data = df_s2 
        # check if there is missing data
        data = data.drop(['family_id', 'username'], axis = 1)
        print('missing data?', np.any(data.isnull()))
        # 123
        print(data[data.isnull().any(axis=1)])
        df_food = data
        #box plot meal_type_breakfast/meal_log_iauc_breakfast
        var = 'meal_type_breakfast'
        data = pd.concat([df_food['meal_log_iauc_breakfast'], df_food[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="meal_log_iauc_breakfast", data=data)
        fig.axis(ymin=0, ymax=2.5)
        #box plot meal_type_breakfast/meal_offset_to_breakfast_hr
        var = 'meal_type_breakfast'
        data = pd.concat([df_food['meal_offset_to_breakfast_hr'], df_food[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="meal_offset_to_breakfast_hr", data=data)
        fig.axis(ymin=-5, ymax=12)
        #box plot meal_type_breakfast/Morning Alertness
        var = 'meal_type_breakfast'
        data = pd.concat([df_food['Morning Alertness'], df_food[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="Morning Alertness", data=data)
        fig.axis(ymin=0, ymax=125)
        #scatter plot meal_log_iauc_breakfast/Morning Alertness
        var = 'meal_log_iauc_breakfast'
        data = pd.concat([df_food['Morning Alertness'], df_food[var]], axis=1)
        data.plot.scatter(x=var, y='Morning Alertness', ylim=(0,150))
        #scatter plot meal_offset_to_breakfast_hr/Morning Alertness
        var = 'meal_offset_to_breakfast_hr'
        data = pd.concat([df_food['Morning Alertness'],  df_food[var]], axis=1)
        data.plot.scatter(x=var, y='Morning Alertness', ylim=(0,150))
        data_1009 = df_s2.loc[df_s2['family_id'] == 'predict1009']
        #box plot meal_type_breakfast/meal_log_iauc_breakfast
        var = 'meal_type_breakfast'
        data = pd.concat([data_1009['meal_log_iauc_breakfast'], df_food[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="meal_log_iauc_breakfast", data=data)
        fig.axis(ymin=0, ymax=2.5)
        #box plot meal_type_breakfast/meal_offset_to_breakfast_hr
        var = 'meal_type_breakfast'
        data = pd.concat([data_1009['meal_offset_to_breakfast_hr'], df_food[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="meal_offset_to_breakfast_hr", data=data)
        fig.axis(ymin=-5, ymax=12)
        
        #box plot meal_type_breakfast/Morning Alertness
        var = 'meal_type_breakfast'
        data = pd.concat([data_1009['Morning Alertness'], df_food[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="Morning Alertness", data=data)
        fig.axis(ymin=0, ymax=125)
        
        #scatter plot meal_log_iauc_breakfast/Morning Alertness
        var = 'meal_log_iauc_breakfast'
        data = pd.concat([data_1009['Morning Alertness'], df_food[var]], axis=1)
        data.plot.scatter(x=var, y='Morning Alertness', ylim=(0,150))
        
        #scatter plot meal_log_iauc_breakfast/Morning Alertness
        var = 'meal_offset_to_breakfast_hr'
        data = pd.concat([data_1009['Morning Alertness'], df_food[var]], axis=1)
        data.plot.scatter(x=var, y='Morning Alertness', ylim=(0,150))
        
        
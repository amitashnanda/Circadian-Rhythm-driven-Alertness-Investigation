import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

class heritability:
    def  __init__(self) -> None:
        pass
    def function1():
        path  = "../data/main.xlsx"
        table3_hr = pd.read_excel(path, sheet_name='Table 3')
        table3_hr =table3_hr[['spt', 'se', 'al_rating_morning','happiness','zygosity','sleepoffset_hr','sleeponset_hr']]
        table3_hr.isna().sum()
        table3_hr = table3_hr.dropna(subset=['al_rating_morning'])
        table3_hr = table3_hr.dropna(subset=['happiness'])
        table3_hr.isna().sum()
        df = table3_hr
        '''This plot shows the morning alertness as compared between different zygosities, Monozygotic (MZ) and 
        Dizygotic (DZ), We can see that the Alertness is quite close to each other, thus the type of zygosity is not an 
        essential parameter for Alertness characterisation'''


        fig = px.box(df, x='zygosity', y='al_rating_morning', color='zygosity', facet_col='zygosity',
                    title='Morning Alertness by Zygosity')

        fig.update_layout(xaxis_title='Zygosity', yaxis_title='Morning Alertness', coloraxis_colorbar_title='Zygosity')
        fig.update_xaxes(matches=None) # remove x-axis ticks and labels from facet columns

        fig.show()
        '''This plot shows the alertness as compared between sleep period time, happiness and sleep efficiency. 
        We can see staying in a state of happiness increases the alertness. We can see that the Alertness is 
        highest for happy people with a sleep period time of  7 to 9 hrs. '''

        fig = px.scatter(df, x="spt", y="al_rating_morning", size="se", color="happiness",
                        title="Relationship between Sleep Period Time, Alertness and Happiness")

        fig.update_layout(xaxis_title='Sleep Period Time', yaxis_title='Alertness Rating', coloraxis_colorbar_title='Happiness')
        fig.show()
        # create the 3D scatter plot
        fig = px.scatter_3d(df, x='se', y='al_rating_morning', z='happiness', color='al_rating_morning', opacity=0.7)

        # add axis labels and title
        fig.update_layout(scene=dict(xaxis_title='Sleep Efficiency',
                                    yaxis_title='Alertness Rating (Morning)',
                                    zaxis_title='Happiness'),
                        title='Relationship between Sleep Efficiency, Alertness Rating, and Happiness')

        # show the plot
        fig.show()
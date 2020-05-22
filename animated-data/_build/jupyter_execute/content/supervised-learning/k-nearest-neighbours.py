# k-Nearest Neighbors

## Introduction

k-nearest neighbors (*kNN*) is...

## Animation

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.offline as py
pio.renderers.default = "notebook"

np.random.seed(11)
df = pd.DataFrame({'X1': np.random.randint(1, 10, 9),
                   'X2': np.random.randint(1, 10, 9),
                    'Y': np.random.choice(['Class 2', 'Class 1'], size=9)})
df.loc[len(df)] = [6, 3, 'Unknown'] # query point
df['Distance'] = ((df[['X1', 'X2']] - df.iloc[-1, :2]) ** 2).sum(axis=1) # distances from query point
df = df.sort_values(by='Distance')
df['Predicted Class'] = ['Unknown', 'Class 1', 'Class 1', 'Class 1', 'Class 1', 'Class 1', 'Class 1', 'Class 2', 'Class 2', 'Class 2']
df

color_dict = {"Class 1": "#636EFA", "Class 2": "#EF553B", "Unknown": "#7F7F7F"}
fig = px.scatter(df, x="X1", y="X2", color='Y', color_discrete_map=color_dict,
                 range_x=[0, 10], range_y=[0, 10],
                 width=650, height=520)
fig.update_traces(marker=dict(size=20,
                              line=dict(width=1)))

# Add lines
shape_dict = {} # create a dictionary
for k in range(0, len(df)):
    shape_dict[k] = [dict(type="line", xref="x", yref="y",x0=x, y0=y, x1=6, y1=3, layer='below',
                          line=dict(color="Black", width=2)) for x, y in df.iloc[1:k+1, :2].to_numpy()]
    if k != 0:
        shape_dict[k].append(dict(type="circle", xref="x", yref="y",x0=5.75, y0=2.75, x1=6.25, y1=3.25,
                                  fillcolor=color_dict[df.iloc[k, 4]]))

# Add dropdown
fig.update_layout(
    updatemenus=[dict(buttons=[dict(args=[{"shapes": shape_dict[k]}],
                                    label=str(k),
                                    method="relayout") for k in range(0, len(df))],
                      direction="down", showactive=True,
                      x=0.115, xanchor="left", y=1.11, yanchor="top")])

# Add dropdown label
fig.update_layout(annotations=[dict(text="k = ",
                                    x=0, xref="paper", y=1.105, yref="paper",
                                    align="left", showarrow=False)],
                  font=dict(size=20))
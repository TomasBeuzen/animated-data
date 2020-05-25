# k-Nearest Neighbors

## Introduction

k-nearest neighbors (*kNN*) is an intuitively simple algorithm in which the label (in classification) or continuous value (in regression) of an unknown test data point is determined using the closest *k* points to it in a training dataset. The distance between data points is often calculated using [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance), although there are other possible choices.

```{note}
*kNN* is sometimes called a "lazy" model because it leaves all the computational work until testing time. Training simply involves storing the training dataset, whereas testing an unknown data point involves searching the training dataset for the *k* nearest neighbors.
```

## Animation

The animation below shows an example of a *kNN* classifier, where an unknown data point is classified using the most common class amongst the *k* nearest neighbors (meassured by Euclidean distance). In the example, there are two features (*X1* and *X2*) and the response (*Y*). Choose different values of *k* from the drop down menu and observe how the classification of the unknown data point changes.

```{note}
In the regression setting, it is common to assign the average value of the *k* nearest neighbors to the test data point.
```

# Imports
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.offline as py
pio.renderers.default = "notebook"

# Create the data
np.random.seed(11)
df = pd.DataFrame({'X1': np.random.randint(1, 10, 9),
                   'X2': np.random.randint(1, 10, 9),
                    'Y': np.random.choice(['Class 2', 'Class 1'], size=9)})
df.loc[len(df)] = [6, 3, 'Unknown'] # query point
df['Distance'] = ((df[['X1', 'X2']] - df.iloc[-1, :2]) ** 2).sum(axis=1) # distances from query point
df = df.sort_values(by='Distance')
df['Predicted Class'] = ['Unknown', 'Class 1', 'Class 1', 'Class 1', 'Class 1', 'Class 1', 'Class 1', 'Class 2', 'Class 2', 'Class 2']

# Plot with plotly
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
                      x=0.115, xanchor="left", y=1.14, yanchor="top")])

# Add dropdown label
fig.update_layout(annotations=[dict(text="k = ",
                                    x=0, xref="paper", y=1.13, yref="paper",
                                    align="left", showarrow=False)],
                  font=dict(size=20))
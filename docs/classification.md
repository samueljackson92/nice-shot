# Classification

The **Classification** tab trains a supervised model on the shot table. You
select the input columns and one target column. The app trains the model, gives
a class to each shot, and shows the result on the scatter plots.

The tab is in the lower left panel, next to **Clustering** and **Outlier
Detection**.

---

## Procedure

1. Select **Model**: Gradient Boosting, Random Forest or Gaussian Process.
2. Select **Target**: the column to learn. The list shows only the columns with
   two to twenty different values. A column with more values than this is a
   quantity, not a label.
3. Select **Features**: one or more numeric columns.
4. Set the hyperparameters. The tab shows only the ones that the selected model
   uses.
5. Click **Train model**. A spinner shows while the model trains.

The status line shows the number of classes, the number of shots, and the
accuracy on the training rows and on the held-out rows.

---

## Results

### The label column

Training makes a `label` column. This column holds the predicted class of each
shot. The Projection and the Pairwise Scatter change to **Label (model)** and
show this column as the point colour.

Shots with no target value are also given a label. The model does not train on
them, but it makes a prediction for them.

### The decision surface

The background of the Projection and the Pairwise Scatter shows the probability
of one class. Red is a high probability, blue is a low probability.

Use the **Surface class** list to select the class. For a target with two
classes, the default is the second class. Thus a 0/1 target shows P(1).

To remove the background, clear the **Decision surface** checkbox.

The surface obeys the filters. It covers only the shots that the Filters tab
keeps, and it is calculated again when you change a filter. Thus you can look
at the decision boundary of one part of the data set.

!!! note "The surface is a two-dimensional view"

    The model uses all the selected feature columns, but a plot has only two
    axes. The app calculates the probability of each shot with the model, then
    interpolates those probabilities across the plane of the plot. The surface
    is thus a view of the decision boundary, not the boundary itself.

    The app does not draw the surface where there are no shots. It also does
    not draw the surface when a filter keeps fewer than three shots, because
    too few points show no boundary.

The surface obeys a logarithmic axis. The app makes the grid in log space, so
the cells stay equally spaced on the screen. On such an axis, the app ignores
the shots with a value of zero or less, because the plot cannot show them
either.

### SHAP

Click a point to see a SHAP decision plot for that shot in the **SHAP** tab.
The plot shows how each feature moved the prediction away from the average.
For a target with more than two classes, the plot shows the class that the
**Surface class** list selects.

SHAP is an optional extra:

```sh
pip install "nice-shot[shap]"
```

### Export

**Download CSV** in the Data Table tab adds the `label` column and one
probability column for each class.

---

## Models

| Model | Hyperparameters | Use it when |
|---|---|---|
| Gradient Boosting | `n_estimators`, `max_depth`, `learning_rate`, `subsample` | The default. It is accurate on tabular data. |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, Balance classes | The classes have very different sizes, or you want a fast fit. |
| Gaussian Process | `kernel`, `length_scale`, `restarts`, `max_train_rows` | There are few shots and you want a smooth probability. |

A Gaussian Process becomes slow when there are many shots. The app takes a
sample of the training rows if there are more than `max_train_rows`. The status
line tells you when this occurs.

---

## Evaluation

**test_fraction** sets the part of the labelled shots that the app keeps back
from the fit. The default is 0.25. The app keeps the class proportions of the
full set in both parts.

The status line shows:

- **train acc** — the accuracy on the rows used for the fit.
- **test acc** — the accuracy on the held-out rows.
- **macro F1** — the mean F1 score of the classes on the held-out rows. Use
  this score when the classes have different sizes, because accuracy alone can
  be high when the model always predicts the largest class.

Set **test_fraction** to 0 to fit on all the labelled shots. The status line
then shows no test score.

**seed** makes the split and the fit repeatable. The same seed gives the same
result.

---

## Normalisation

The app normalises the features before the fit. It puts a missing value to the
median of its column, then scales each column by its median and its
interquartile range.

This is a robust normalisation. A small number of extreme shots — a disruption,
or a bad measurement — does not set the scale for all the other shots. A column
in megamps and a column in per-cent thus have the same importance to the model.

---

## When the results are cleared

A change to the projection settings clears the labels, the probabilities and
the model. The feature matrix came from the previous dataset, so the
predictions are no longer correct for the new one. Train the model again.

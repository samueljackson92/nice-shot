# Using the Dashboard

This page tells you how to use each tab of the dashboard.

---

## Layout

The dashboard has two panes. Each pane has groups of tabs.

| Pane | Tabs |
|------|------|
| Left pane, upper part | Time Traces, SHAP, Cluster Traces, Outlier Traces |
| Left pane, lower part | Shot Info, Clustering, Outlier Detection, Classification, Filters |
| Right pane | Projection, Pairwise Scatter, Data Table, Search, Correlation, Lineage, Configuration |

Click a point on a scatter plot, or a row in the Data Table, to select a shot. Many tabs use the selected shot.

---

## Projection

The **Projection** tab shows a UMAP or PCA scatter plot of all the shots. Set the method and its options with [`projection_method`](configuration.md#projection_method) and [`projection_options`](configuration.md#projection_options), or in the [Configuration tab](configuration.md#configuration-tab).

Use **Color by** to select the column that sets the colour of the points. After you run clustering, outlier detection or classification, the plot changes to **Cluster**, **Outliers** or **Label (model)**.

The app keeps the projection in a cache. The next start with the same data and settings does not calculate it again.

---

## Pairwise Scatter

The **Pairwise Scatter** tab plots two numeric columns against each other.

- Select the **X axis** and the **Y axis** columns.
- Use **Lin** or **Log** to set the scale of each axis.
- Use **Color by (optional)** to set the colour of the points.

---

## Data Table

The **Data Table** tab shows the shot records. You can sort the table and search it by shot ID. Click a row to select that shot on the scatter plots.

**Download CSV** exports the table. The file also has these columns:

| Column | When |
|--------|------|
| `cluster_id` | After clustering. `-1` is DBSCAN noise. |
| `cluster_name` | After clustering. The name you gave in the Clustering tab, or `Cluster N`. |
| `label` | After classification. The predicted class. |
| One probability column for each class | After classification. |

---

## Filters

The **Filters** tab keeps only the shots that agree with a set of conditions. All the plots and tables obey the filters, except the [Lineage tab](#lineage).

You can set a maximum of six filters. Each filter has a column, an operator and a value. The operators are `>=`, `<=`, `>`, `<`, `==`, `!=` and `contains`. The `contains` operator compares text and ignores case.

Use **Combine with** to select **AND** (a shot must agree with all the filters) or **OR** (a shot must agree with one filter or more). Click **x** to remove one filter, or **Clear all** to remove all of them. The tab shows the number of shots that the filters keep.

---

## Shot Info

The **Shot Info** tab shows all the values for the selected shot.

---

## Time Traces

The **Time Traces** tab shows the [signals](configuration.md#signals) of the selected shot. The app reads them with the [trace backend](configuration.md#backend).

The app hides the time-trace panel when it cannot find trace data, for example when `--data-dir` does not exist.

---

## Clustering

The **Clustering** tab puts the shots into groups from their statistics.

1. Select the algorithm: K-Means, DBSCAN or Agglomerative.
2. Select the feature columns. The default is the projection features.
3. Set the parameters:
    - `n_clusters` — the number of clusters (K-Means and Agglomerative).
    - `eps` and `min_samples` — the neighbourhood radius and the density limit (DBSCAN).
4. Click **Run clustering**. The scatter plots change to the cluster colours.
5. Type a name for each cluster. The legend and the cluster traces change immediately.

### Cluster Traces

After clustering, the **Cluster Traces** tab shows the mean time trace of each cluster, with one line for each cluster. The app calculates the traces after each run. Click **Compute centroid traces** to calculate them again.

This tab needs a [trace backend](configuration.md#backend). If there is no trace data, the tab shows a message.

---

## Outlier Detection

The **Outlier Detection** tab finds unusual shots.

1. Select the algorithm: Isolation Forest or Local Outlier Factor.
2. Select the feature columns. The default is the projection features.
3. Set the parameters:
    - `contamination` — the expected fraction of outliers, from 0.01 to 0.5. The default is 0.1.
    - `n_neighbors` — the neighbourhood size, for Local Outlier Factor only. The default is 20.
4. Click **Run outlier detection**. The scatter plots show outliers in red and inliers in blue.

### Outlier Traces

After each run, the **Outlier Traces** tab shows the time traces of a maximum of five outlier shots. Each shot has its own line.

This tab needs a [trace backend](configuration.md#backend).

---

## Classification

The **Classification** tab trains a supervised model on the shot table. Refer to [Classification](classification.md).

---

## SHAP

The **SHAP** tab shows a SHAP decision plot for the selected shot. The plot shows how each feature moved the prediction away from the mean prediction.

The tab gets SHAP values from one of these sources:

- A file that you give with `--shap-data`. Refer to [SHAP values file](data-formats.md#shap-values-file-shap-data).
- A model that you train in the [Classification](classification.md#shap) tab.

```sh
nice-shot outputs/shot_stats.parquet --shap-data outputs/shap_values.nc
```

The plots need the `shap` extra:

```sh
pip install "nice-shot[shap]"
```

---

## Search

The **Search** tab finds the shots that are most similar to one shot. It uses a nearest-neighbour search in feature space.

1. Click a shot on a scatter plot. The search starts automatically. Alternatively, type a shot ID and click **Find similar shots**.
2. Set **K**, the number of similar shots to find. The default is 10.
3. Change the features if necessary. The default is the projection features.

The Projection and Pairwise Scatter plots show the results as gold rings. The most similar shot has the brightest ring. The tab shows a table of the results with their distances, and the time traces of the similar shots.

Use the **Similar shots: ON/OFF** button in the left pane to hide the rings. This does not clear the results.

---

## Correlation

The **Correlation** tab shows a Pearson correlation matrix for a set of numeric columns.

- The default columns are the projection features. Select at least two columns.
- The matrix obeys the filters, so it shows only the shots that the filters keep.
- Blue is −1, white is 0 and red is +1. Each cell shows its `r` value.

---

## Lineage

The **Lineage** tab shows how one shot is different from the shots before it in the same reference lineage. It tells you what changed against the reference shot, and what the earlier shots in the lineage tried.

To use this tab, set [`reference_shot_col`](configuration.md#reference_shot_col) to the column that holds the reference (parent) shot ID of each shot:

```sh
nice-shot shot_stats.parquet --reference-shot-col reference__number
```

The tab is always in the tab bar. Until you set the column, the tab is off and tells you what is missing. You can also set the column in the [Configuration tab](configuration.md#configuration-tab), without a restart.

The tab uses the selected shot. If no shot is selected, the tab uses the shot with the highest shot ID.

Use the lineage control to select which shots the tab includes:

| Lineage | Shots included |
|---------|----------------|
| Ancestor chain | The shot, its reference shot, the reference shot of that shot, and so on. This is the default. |
| Connected | All shots that connect to the shot through reference links. |
| Siblings | All shots that have the same reference shot. |

A lineage holds a maximum of 100 shots. The tab always keeps the selected shot.

### The summary card

At the top, the tab shows a card with one row for each variable in the table. Each row gives the old value, the new value, the size of the change, and a bar. The largest change comes first, in either direction. Scroll the card to see the smaller changes and then the variables that did not change. The card covers all the variables in the table, not only the variables you select below it.

Use the **Rank by** control to select the measure that sorts the card and gives the value in the change column:

| Rank by | Description |
|---------|-------------|
| z-scored | The change divided by the spread of that column across all shots. This is the default. Use it to compare variables that have different units. |
| percentage | The change as a percentage of the value of the reference shot. The card gives the percentage in its own column, and the percentage sets the length of the bar. The change column then gives the change in the units of the variable. The tab cannot calculate a percentage if the value of the reference shot is 0. |

### The history table

The **History** view shows one row for each shot in the lineage. The newest shot is the first row. The `rel` column gives the position of each shot relative to the selected shot.

The table gives one column for each variable you select. The colour of a cell shows the change from the previous shot in the lineage. Red is an increase. Blue is a decrease. The text in a cell is always the raw value. Point at a cell to see the change.

Use the colour control to select the measure:

| Measure | Description |
|---------|-------------|
| z-scored change | The change divided by the spread of that column across all shots. This is the default. It makes the colours comparable between variables that use different units. |
| percent change | The change as a percentage of the previous value. The tab cannot calculate this value if the previous value is 0. |
| absolute change | The change in the units of the variable. The colours are not comparable between variables. |

### The other views

| View | Description |
|------|-------------|
| Notes | The text fields for each shot in the lineage. The tab shows all the columns that it cannot compare as numbers. The comment, objective and scenario fields come first. |
| Tree | The reference links between the shots in the lineage. Click a node to select that shot. |
| Sparklines | One small graph for each variable, in shot order. Use the **Visualise** control to select the variables. The view starts with the first 12 variables from the summary card, in the same order, and follows the **Rank by** control. Select **Top 12** to set this selection again. |

The tab builds the **Sparklines** view only while you look at it. It shows a spinner over a view while the view loads. A wide table, or a lineage that holds many shots, can need some seconds.

### Variables

A shot table can hold many variables. The variable control therefore searches the column names as you type, and shows a maximum of 200 names at a time.

The tab selects all the projection features: the `umap_features` columns, or all the numeric columns if you do not set `umap_features`. These are the variables that give each shot its position on the scatter plot.

Select **Top changed** to select the 20 variables that changed most instead. Select **Projection features** to select the projection features again. The tab keeps your selection when you click a different shot.

### Filters and the lineage

The Lineage tab does not use the filters. A hidden shot would make the change values incorrect, because the tab would then compare two shots that are not linked.

Select **Mark filtered shots** to show which shots the filters hide. The tab keeps these shots in the table, but shows them in grey. The change values stay correct.

---

## Configuration

The **Configuration** tab changes most settings while the app runs. Refer to [Configuration tab](configuration.md#configuration-tab).

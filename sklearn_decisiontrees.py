import numpy as np
from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
import joblib
import plotly.colors as colors
import plotly.express as px
from plotly.subplots import make_subplots


### This code shows how to use KFold to do cross_validation.
### This is just one of many ways to manage training and test sets in sklearn.

wine = load_wine()
X, y = wine.data, wine.target
scores = []
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X) :
    X_train, X_test, y_train, y_test = \
        (X[train_index], X[test_index], y[train_index], y[test_index])
    # clf = tree.DecisionTreeClassifier()
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, criterion='gini')
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

print(scores)

n_estimators_list = [10, 25, 50]
criteria = ['gini', 'entropy']

results = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for criterion in criteria:
    for n_estimators in n_estimators_list:
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, random_state=42)
            clf.fit(X_train, y_train)
            scores.append(clf.score(X_test, y_test))
        results.append({
            'Criterion': criterion,
            'Estimators': n_estimators,
            'Mean Accuracy': np.mean(scores),
            'Std Accuracy': np.std(scores)
        })

results_df = pd.DataFrame(results)
fig = px.bar(
    results_df,
    x="Estimators",
    y="Mean Accuracy",
    color="Criterion",
    barmode="group",
    error_y="Std Accuracy",
    title="Random Forest Mean Accuracy by Number of Estimators and Criterion",
    labels={"Estimators": "Number of Estimators", "Mean Accuracy": "Accuracy"}
)
fig.show()

print(results_df)

## Part 2. This code (from https://scikit-learn.org/1.5/auto_examples/ensemble/plot_forest_hist_grad_boosting_comparison.html)
## shows how to use GridSearchCV to do a hyperparameter search to compare two techniques.
from sklearn.datasets import load_breast_cancer

X,y = load_breast_cancer(return_X_y=True, as_frame=True)

N_CORES = joblib.cpu_count(only_physical_cores=True)
print(f"Number of physical cores: {N_CORES}")

models = {
    "Random Forest": RandomForestClassifier(
        min_samples_leaf=5, random_state=0, n_jobs=N_CORES
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_leaf_nodes=15, random_state=0, early_stopping=False
    ),
}
param_grids = {
    "Random Forest": {"n_estimators": [5, 10, 15, 20]},
    "Hist Gradient Boosting": {"max_iter": [25, 50, 75, 100]},
}
cv = KFold(n_splits=5, shuffle=True, random_state=0)

results = []
for name, model in models.items():
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        return_train_score=True,
        cv=cv,
    ).fit(X, y)
    result = {"model": name, "cv_results": pd.DataFrame(grid_search.cv_results_)}
    results.append(result)

print(results)

#### Part 3: This shows how to generate a scatter plot of your results

fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=["Train time vs score", "Predict time vs score"],
)
model_names = [result["model"] for result in results]
colors_list = colors.qualitative.Plotly * (
    len(model_names) // len(colors.qualitative.Plotly) + 1
)

for idx, result in enumerate(results):
    cv_results = result["cv_results"].round(3)
    model_name = result["model"]
    param_name = list(param_grids[model_name].keys())[0]
    cv_results[param_name] = cv_results["param_" + param_name]
    cv_results["model"] = model_name

    scatter_fig = px.scatter(
        cv_results,
        x="mean_fit_time",
        y="mean_test_score",
        error_x="std_fit_time",
        error_y="std_test_score",
        hover_data=param_name,
        color="model",
    )
    line_fig = px.line(
        cv_results,
        x="mean_fit_time",
        y="mean_test_score",
    )

    scatter_trace = scatter_fig["data"][0]
    line_trace = line_fig["data"][0]
    scatter_trace.update(marker=dict(color=colors_list[idx]))
    line_trace.update(line=dict(color=colors_list[idx]))
    fig.add_trace(scatter_trace, row=1, col=1)
    fig.add_trace(line_trace, row=1, col=1)

    scatter_fig = px.scatter(
        cv_results,
        x="mean_score_time",
        y="mean_test_score",
        error_x="std_score_time",
        error_y="std_test_score",
        hover_data=param_name,
    )
    line_fig = px.line(
        cv_results,
        x="mean_score_time",
        y="mean_test_score",
    )

    scatter_trace = scatter_fig["data"][0]
    line_trace = line_fig["data"][0]
    scatter_trace.update(marker=dict(color=colors_list[idx]))
    line_trace.update(line=dict(color=colors_list[idx]))
    fig.add_trace(scatter_trace, row=1, col=2)
    fig.add_trace(line_trace, row=1, col=2)

fig.update_layout(
    xaxis=dict(title="Train time (s) - lower is better"),
    yaxis=dict(title="Test R2 score - higher is better"),
    xaxis2=dict(title="Predict time (s) - lower is better"),
    legend=dict(x=0.72, y=0.05, traceorder="normal", borderwidth=1),
    title=dict(x=0.5, text="Speed-score trade-off of tree-based ensembles"),
)
fig.show()
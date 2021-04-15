import pandas as pd
import plotly.graph_objects as go


def best_params(grid):
    print("The best parameters are %s with a score of %0.3f" % (grid.best_params_, grid.best_score_))

    grid_results = pd.concat([
        pd.DataFrame(grid.cv_results_["params"]),
        pd.DataFrame(grid.cv_results_["mean_test_score"],
        columns=["Accuracy"])
    ], axis=1)

    grid_results.head()

    grid_contour = grid_results.groupby(['max_features', 'n_estimators']).mean()

    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'n_estimators', 'Accuracy']
    grid_pivot = grid_reset.pivot('max_features', 'n_estimators')

    print('grid_pivot')
    print(grid_pivot)

    return grid_pivot.columns.levels[1].values, grid_pivot.index.values, grid_pivot.values


def dt_best_params(grid):
    print("The best parameters are %s with a score of %0.3f" % (grid.best_params_, grid.best_score_))

    grid_results = pd.concat([
        pd.DataFrame(grid.cv_results_["params"]),
        pd.DataFrame(grid.cv_results_["mean_test_score"],
        columns=["Accuracy"])
    ], axis=1)

    grid_results.head()

    grid_contour = grid_results.groupby(['max_features', 'max_depth']).mean()

    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'max_depth', 'Accuracy']
    grid_pivot = grid_reset.pivot('max_features', 'max_depth')

    print('grid_pivot')
    print(grid_pivot)

    return grid_pivot.columns.levels[1].values, grid_pivot.index.values, grid_pivot.values


def visualise(grid):

    x, y, z = best_params(grid)
    # X and Y axes labels
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='n_estimators')
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='max_features')
        ))

    fig = go.Figure(data=[go.Contour(z=z, x=x, y=y)], layout=layout)

    fig.update_layout(title='Hyperparameter tuning', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.show()

    fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
    fig.update_layout(title='Hyperparameter tuning',
                      scene=dict(
                          xaxis_title='n_estimators',
                          yaxis_title='max_features',
                          zaxis_title='Accuracy'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()


def dt_visualise(grid):

    x, y, z = dt_best_params(grid)
    # X and Y axes labels
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='max_depth')
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='max_features')
        ))

    fig = go.Figure(data=[go.Contour(z=z, x=x, y=y)], layout=layout)

    fig.update_layout(title='Hyperparameter tuning', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.show()

    fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
    fig.update_layout(title='Hyperparameter tuning',
                      scene=dict(
                          xaxis_title='max_depth',
                          yaxis_title='max_features',
                          zaxis_title='Accuracy'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

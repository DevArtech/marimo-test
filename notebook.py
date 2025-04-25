

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    return (
        ConfusionMatrixDisplay,
        LogisticRegression,
        confusion_matrix,
        make_classification,
        plt,
        train_test_split,
    )


@app.cell
def _(make_classification):
    X, y = make_classification(
        n_samples=100,     
        n_features=2,      
        n_informative=2,   
        n_redundant=0,     
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(LogisticRegression, X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(X_test, model):
    y_pred = model.predict(X_test)
    return (y_pred,)


@app.cell
def _(confusion_matrix, y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    return (cm,)


@app.cell
def _(ConfusionMatrixDisplay, cm, plt):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

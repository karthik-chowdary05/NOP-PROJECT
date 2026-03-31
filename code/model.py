import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def run_models(X, y):

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    # Scale features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    pred_lr = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, pred_lr)

    print("Linear Regression MSE:", mse_lr)

    # Ridge Regression
    ridge = Ridge(alpha=50)
    ridge.fit(X_train, y_train)

    pred_ridge = ridge.predict(X_test)
    mse_ridge = mean_squared_error(y_test, pred_ridge)

    print("Ridge Regression MSE:", mse_ridge)

    # LASSO Regression
    lasso = Lasso(alpha=50)
    lasso.fit(X_train, y_train)

    pred_lasso = lasso.predict(X_test)
    mse_lasso = mean_squared_error(y_test, pred_lasso)

    print("LASSO MSE:", mse_lasso)

    # Dynamic LASSO optimization
    alphas = [100, 50, 10, 5, 1]

    best_mse = float("inf")
    best_alpha = None

    for alpha in alphas:

        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)

        print("Alpha:", alpha, "MSE:", mse)

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    print("Best Alpha:", best_alpha)
    print("Best Dynamic LASSO MSE:", best_mse)

    # Feature selection
    selected_features = np.sum(lasso.coef_ != 0)

    print("Number of Selected Features:", selected_features)

    return {
        "Linear": mse_lr,
        "Ridge": mse_ridge,
        "LASSO": mse_lasso,
        "Dynamic": best_mse
    }
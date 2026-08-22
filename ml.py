import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load data
X, y = fetch_california_housing(return_X_y=True)

# Always scale features before Ridge — the penalty is scale-sensitive
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compare OLS vs Ridge
ols = LinearRegression()
ridge = Ridge(alpha=10.0)  # alpha = λ in the formula

ols_scores = cross_val_score(ols, X_scaled, y, cv=5, scoring="r2")
ridge_scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring="r2")

print(f"OLS   R²: {ols_scores.mean():.4f} ± {ols_scores.std():.4f}")
print(f"Ridge R²: {ridge_scores.mean():.4f} ± {ridge_scores.std():.4f}")

# Check coefficient sizes — Ridge shrinks them
ols.fit(X_scaled, y)
ridge.fit(X_scaled, y)

print(f"\nOLS   coef range: [{ols.coef_.min():.3f}, {ols.coef_.max():.3f}]")
print(f"Ridge coef range: [{ridge.coef_.min():.3f}, {ridge.coef_.max():.3f}]")

from sklearn.linear_model import RidgeCV

# RidgeCV does cross-validation over a range of alphas for you
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge_cv.fit(X_scaled, y)

print(f"Best alpha: {ridge_cv.alpha_}")
print(f"R²: {ridge_cv.score(X_scaled, y):.4f}")

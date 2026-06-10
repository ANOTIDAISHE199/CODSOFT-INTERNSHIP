# ============================================================
# TASK 4 - SALES PREDICTION USING PYTHON
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# STEP 1:
df = pd.read_csv('adv.xls')
print("Dataset Shape:", df.shape)
print("\nDataset Overview:")
print(df.describe())
print("\nMissing Values:", df.isnull().sum().sum())

#STEP 2: DEFINE FEATURES & TARGET
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\nTraining: {len(X_train)} | Testing: {len(X_test)}")

#STEP 3: TRAIN MODELS
lr   = LinearRegression()

poly = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('reg',  LinearRegression())
])

rf   = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
gb   = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                  learning_rate=0.1, random_state=42)

models = {
    'Linear Regression' : lr,
    'Polynomial Reg'    : poly,
    'Random Forest'     : rf,
    'Gradient Boosting' : gb
}

kf      = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n── Model Results ──")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    cv     = cross_val_score(model, X, y, cv=kf, scoring='r2')
    results[name] = {
        'model': model, 'rmse': rmse, 'mae': mae,
        'r2': r2, 'cv_r2': cv.mean(),
        'cv_std': cv.std(), 'y_pred': y_pred
    }
    print(f"{name:22s} | RMSE={rmse:.4f}  MAE={mae:.4f}"
          f"  R²={r2:.4f}  CV={cv.mean():.4f}±{cv.std():.4f}")

best_name  = max(results, key=lambda k: results[k]['r2'])
best       = results[best_name]
print(f"\n✅ Best Model: {best_name}")

#STEP 4: VISUALIZATIONS
COLORS = ['#FFD700', '#FF6B6B', '#2EC4B6', '#9B72CF', '#6C8EBF']
CARD   = '#16162A'
BG     = '#0d0d1f'

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor(BG)
fig.suptitle('Sales Prediction Dashboard',
             fontsize=20, fontweight='bold', color='white')

def style(ax):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color('#2a2a4a')
    ax.tick_params(colors='white', labelsize=9)
    ax.title.set_color('white'); ax.title.set_fontweight('bold')
    try: ax.xaxis.label.set_color('white')
    except: pass
    try: ax.yaxis.label.set_color('white')
    except: pass

# Plot 1 – TV vs Sales
ax = axes[0][0]
ax.scatter(df['TV'], df['Sales'], color=COLORS[0], alpha=0.6, s=25)
m, b = np.polyfit(df['TV'], df['Sales'], 1)
xl = np.linspace(df['TV'].min(), df['TV'].max(), 100)
ax.plot(xl, m*xl+b, color=COLORS[2], lw=2)
ax.set_title('TV Budget vs Sales')
ax.set_xlabel('TV Budget'); ax.set_ylabel('Sales'); style(ax)

# Plot 2 – Radio vs Sales
ax = axes[0][1]
ax.scatter(df['Radio'], df['Sales'], color=COLORS[1], alpha=0.6, s=25)
m2, b2 = np.polyfit(df['Radio'], df['Sales'], 1)
xl2 = np.linspace(df['Radio'].min(), df['Radio'].max(), 100)
ax.plot(xl2, m2*xl2+b2, color=COLORS[2], lw=2)
ax.set_title('Radio Budget vs Sales')
ax.set_xlabel('Radio Budget'); ax.set_ylabel('Sales'); style(ax)

# Plot 3 – Model Comparison
ax = axes[0][2]
short = ['LR', 'Poly', 'RF', 'GB']
r2s   = [results[n]['r2'] for n in models]
b4    = ax.barh(short, r2s, color=COLORS[:4], edgecolor='none')
for bar, v in zip(b4, r2s):
    ax.text(v+0.005, bar.get_y()+bar.get_height()/2,
            f'{v:.3f}', va='center', color='white',
            fontsize=10, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.set_title('R² Score Comparison')
ax.set_xlabel('R²'); style(ax)

# Plot 4 – Actual vs Predicted
ax = axes[1][0]
ax.scatter(y_test, best['y_pred'], alpha=0.6, s=30,
           color=COLORS[1], edgecolors='none')
mn, mx = float(y_test.min()), float(y_test.max())
ax.plot([mn,mx],[mn,mx], '--', color=COLORS[2], lw=2, label='Perfect Fit')
ax.set_title(f'Actual vs Predicted ({best_name})')
ax.set_xlabel('Actual Sales'); ax.set_ylabel('Predicted')
ax.legend(facecolor=CARD, labelcolor='white', fontsize=8); style(ax)

# Plot 5 – Residuals
ax = axes[1][1]
residuals = y_test.values - best['y_pred']
ax.scatter(best['y_pred'], residuals, alpha=0.5, s=25,
           color=COLORS[0], edgecolors='none')
ax.axhline(0, color=COLORS[2], lw=2, linestyle='--')
ax.set_title('Residuals Plot')
ax.set_xlabel('Predicted Sales'); ax.set_ylabel('Residual'); style(ax)

# Plot 6 – Correlation Heatmap
ax = axes[1][2]
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, linewidths=2, linecolor=BG, cbar=False,
            annot_kws={'size': 12, 'weight': 'bold', 'color': 'white'})
ax.set_title('Feature Correlation Heatmap'); style(ax)

plt.tight_layout()
plt.savefig('sales_prediction_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.show()
print("\n✅ Dashboard saved as sales_prediction_dashboard.png")
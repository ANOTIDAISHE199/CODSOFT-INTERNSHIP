
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

#STEP 1: LOAD DATA
df = pd.read_csv('movie.csv', encoding='latin1')
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

#STEP 2: PREPROCESS
df['Year']     = df['Year'].str.extract(r'(\d{4})').astype(float)
df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(float)
df['Votes']    = pd.to_numeric(
    df['Votes'].astype(str).str.replace(',', ''), errors='coerce')

df = df.dropna(subset=['Rating'])
print(f"\nRows after dropping missing ratings: {len(df)}")

df['Duration'].fillna(df['Duration'].median(), inplace=True)
df['Votes'].fillna(0, inplace=True)
df['Year'].fillna(df['Year'].median(), inplace=True)
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col].fillna('Unknown', inplace=True)

# STEP 3: FEATURE ENGINEERING
df['Genre_primary'] = df['Genre'].str.split(',').str[0].str.strip()

global_mean = df['Rating'].mean()
for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Genre_primary']:
    means = df.groupby(col)['Rating'].mean()
    df[col + '_enc'] = df[col].map(means).fillna(global_mean)

df['LogVotes'] = np.log1p(df['Votes'])

FEATURES = ['Year', 'Duration', 'LogVotes',
            'Director_enc', 'Actor 1_enc',
            'Actor 2_enc', 'Actor 3_enc', 'Genre_primary_enc']

X = df[FEATURES].fillna(global_mean)
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\nTraining: {len(X_train)} | Testing: {len(X_test)}")

# ── STEP 4: TRAIN MODELS ───────────────────────────────────
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
gb = GradientBoostingRegressor(n_estimators=150, max_depth=5,
                                learning_rate=0.1, random_state=42)
rr = Ridge(alpha=1.0)

models  = {'Random Forest': rf, 'Gradient Boosting': gb, 'Ridge Regression': rr}
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
    results[name] = {'model': model, 'rmse': rmse, 'mae': mae,
                     'r2': r2, 'cv_r2': cv.mean(),
                     'cv_std': cv.std(), 'y_pred': y_pred}
    print(f"{name:22s} | RMSE={rmse:.4f} MAE={mae:.4f} "
          f"R²={r2:.4f} CV_R²={cv.mean():.4f}±{cv.std():.4f}")

best_name  = max(results, key=lambda k: results[k]['r2'])
best       = results[best_name]
print(f"\n✅ Best Model: {best_name}")

#STEP 5: VISUALIZATIONS
COLORS = ['#FFD700', '#FF6B6B', '#2EC4B6', '#9B72CF', '#6C8EBF']
CARD   = '#16162A'
BG     = '#0d0d1f'

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor(BG)
fig.suptitle('Movie Rating Prediction Dashboard',
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

# Plot 1 – Rating Distribution
ax = axes[0][0]
ax.hist(df['Rating'], bins=30, color=COLORS[0],
        edgecolor='none', alpha=0.85)
ax.set_title('Rating Distribution')
ax.set_xlabel('Rating'); ax.set_ylabel('Count'); style(ax)

# Plot 2 – Avg Rating by Genre
ax = axes[0][1]
top_g = df['Genre_primary'].value_counts().head(8).index
gm    = df[df['Genre_primary'].isin(top_g)].groupby(
    'Genre_primary')['Rating'].mean().sort_values()
ax.barh(gm.index, gm.values,
        color=[COLORS[i % 5] for i in range(len(gm))], edgecolor='none')
ax.set_title('Avg Rating by Genre')
ax.set_xlabel('Avg Rating'); style(ax)

# Plot 3 – Rating Over Years
ax = axes[0][2]
yr = df.groupby('Year')['Rating'].mean().reset_index()
yr = yr[(yr['Year'] >= 1970) & (yr['Year'] <= 2023)]
ax.plot(yr['Year'], yr['Rating'], color=COLORS[1], lw=2)
ax.fill_between(yr['Year'], yr['Rating'], alpha=0.2, color=COLORS[1])
ax.set_title('Avg Rating Over Years')
ax.set_xlabel('Year'); ax.set_ylabel('Rating'); style(ax)

# Plot 4 – Actual vs Predicted
ax = axes[1][0]
ax.scatter(y_test, best['y_pred'], alpha=0.3, s=12,
           color=COLORS[1], edgecolors='none')
mn, mx = float(y_test.min()), float(y_test.max())
ax.plot([mn, mx], [mn, mx], '--', color=COLORS[2], lw=2,
        label='Perfect Fit')
ax.set_title(f'Actual vs Predicted ({best_name})')
ax.set_xlabel('Actual Rating'); ax.set_ylabel('Predicted')
ax.legend(facecolor=CARD, labelcolor='white', fontsize=8); style(ax)

# Plot 5 – Residuals
ax = axes[1][1]
residuals = y_test.values - best['y_pred']
ax.scatter(best['y_pred'], residuals, alpha=0.3, s=12,
           color=COLORS[0], edgecolors='none')
ax.axhline(0, color=COLORS[2], lw=2, linestyle='--')
ax.set_title('Residuals Plot')
ax.set_xlabel('Predicted Rating'); ax.set_ylabel('Residual'); style(ax)

# Plot 6 – Feature Importances
ax = axes[1][2]
fi   = results['Random Forest']['model'].feature_importances_
fi_s = pd.Series(fi, index=FEATURES).sort_values()
ax.barh(fi_s.index, fi_s.values,
        color=[COLORS[0] if v >= fi_s.median() else COLORS[4]
               for v in fi_s.values], edgecolor='none')
ax.set_title('Feature Importances (RF)')
ax.set_xlabel('Importance'); style(ax)

plt.tight_layout()
plt.savefig('movie_rating_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.show()
print("\n✅ Dashboard saved as movie_rating_dashboard.png")
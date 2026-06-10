
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#STEP 1: LOAD DATA
df = pd.read_csv('Titanic.csv')
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

#STEP 2: PREPROCESS DATA
def preprocess(df):
    d = df.copy()

    # Fill missing Age by group median
    d['Age'] = d.groupby(['Pclass', 'Sex'])['Age'].transform(
        lambda x: x.fillna(x.median()))
    d['Age'].fillna(d['Age'].median(), inplace=True)

    # Fill other missing values
    d['Embarked'].fillna(d['Embarked'].mode()[0], inplace=True)
    d['Fare'].fillna(d['Fare'].median(), inplace=True)

    # Extract Title from Name
    d['Title'] = d['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    d['Title'] = d['Title'].replace(
        ['Lady','Countess','Capt','Col','Don',
         'Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    d['Title'] = d['Title'].replace(
        {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # Encode features
    d['Sex_enc']      = (d['Sex'] == 'male').astype(int)
    d['Embarked_enc'] = d['Embarked'].map({'S':0,'C':1,'Q':2}).fillna(0).astype(int)
    d['Title_enc']    = d['Title'].map(
        {'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4}).fillna(0).astype(int)

    # New features
    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1
    d['IsAlone']    = (d['FamilySize'] == 1).astype(int)
    d['AgeBand']    = pd.cut(d['Age'],
                             bins=[0,12,18,35,60,100],
                             labels=[0,1,2,3,4]).astype(float).fillna(2).astype(int)
    d['FareBand']   = pd.qcut(d['Fare'], 4,
                               labels=[0,1,2,3]).astype(float).fillna(0).astype(int)
    d['CabinKnown'] = d['Cabin'].notna().astype(int)

    return d

df = preprocess(df)
print("\nPreprocessing complete ✅")

# STEP 3: DEFINE FEATURES & SPLIT
FEATURES = ['Pclass','Sex_enc','AgeBand','FareBand','Embarked_enc',
            'FamilySize','IsAlone','Title_enc','CabinKnown','SibSp','Parch']

X = df[FEATURES].fillna(0)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# STEP 4: TRAIN MODELS
rf = RandomForestClassifier(n_estimators=200, max_depth=7,min_samples_split=4, random_state=42)
gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
lr = Pipeline([('sc', StandardScaler()),
               ('clf', LogisticRegression(max_iter=1000, random_state=42))])

models = {'Random Forest': rf, 'Gradient Boosting': gb, 'Logistic Regression': lr}
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n── Training Models ──")
for name, model in models.items():
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    y_pred    = model.predict(X_test)
    results[name] = {
        'model'  : model,
        'acc'    : accuracy_score(y_test, y_pred),
        'cv_mean': cv_scores.mean(),
        'cv_std' : cv_scores.std()
    }
    print(f"{name:25s} | Test: {results[name]['acc']:.4f} "
          f"| CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# STEP 5: BEST MODEL REPORT
best_name  = max(results, key=lambda k: results[k]['cv_mean'])
best_model = results[best_name]['model']
y_pred_b   = best_model.predict(X_test)

print(f"\n✅ Best Model: {best_name}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_b,
                             target_names=['Not Survived','Survived']))

#STEP 6: VISUALIZATIONS
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#0d0d1f')
fig.suptitle('Titanic Survival Prediction Dashboard',
             fontsize=20, fontweight='bold', color='white')

COLORS = ['#7b68ee','#43e8b8','#ff6b8a','#ffaa5a','#56b4e9']
CARD   = '#16162a'

def style(ax):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color('#2a2a4a')
    ax.tick_params(colors='white', labelsize=9)
    ax.title.set_color('white')
    ax.title.set_fontweight('bold')
    try: ax.xaxis.label.set_color('white')
    except: pass
    try: ax.yaxis.label.set_color('white')
    except: pass

# Plot 1 – Survival by Sex
ax = axes[0][0]
sg = df.groupby('Sex')['Survived'].mean()
b  = ax.bar(['Female','Male'], sg.values,
             color=[COLORS[2], COLORS[0]], width=0.5, edgecolor='none')
for bar, v in zip(b, sg.values):
    ax.text(bar.get_x()+0.25, v+0.02, f'{v:.0%}',
            ha='center', color='white', fontweight='bold')
ax.set_ylim(0, 1.1); ax.set_title('Survival Rate by Sex')
ax.set_ylabel('Rate', color='white'); style(ax)

# Plot 2 – Survival by Class
ax = axes[0][1]
cg = df.groupby('Pclass')['Survived'].mean()
b2 = ax.bar(['1st','2nd','3rd'], cg.values,
             color=[COLORS[1],COLORS[3],COLORS[2]], width=0.5, edgecolor='none')
for bar, v in zip(b2, cg.values):
    ax.text(bar.get_x()+0.25, v+0.02, f'{v:.0%}',
            ha='center', color='white', fontweight='bold')
ax.set_ylim(0, 1.1); ax.set_title('Survival Rate by Class')
ax.set_ylabel('Rate', color='white'); style(ax)

# Plot 3 – Age Distribution
ax = axes[0][2]
ax.hist(df[df['Survived']==0]['Age'], bins=25, alpha=0.7,
        color=COLORS[2], label='Died', edgecolor='none')
ax.hist(df[df['Survived']==1]['Age'], bins=25, alpha=0.7,
        color=COLORS[1], label='Survived', edgecolor='none')
ax.set_title('Age Distribution by Outcome')
ax.set_xlabel('Age', color='white'); ax.set_ylabel('Count', color='white')
ax.legend(facecolor=CARD, labelcolor='white', fontsize=8); style(ax)

# Plot 4 – Confusion Matrix
ax = axes[1][0]
cm = confusion_matrix(y_test, y_pred_b)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
            xticklabels=['Died','Survived'],
            yticklabels=['Died','Survived'],
            linewidths=2, linecolor='#0d0d1f', cbar=False,
            annot_kws={'size':16, 'weight':'bold', 'color':'white'})
ax.set_title(f'Confusion Matrix ({best_name})')
ax.set_xlabel('Predicted', color='white')
ax.set_ylabel('Actual', color='white'); style(ax)

# Plot 5 – ROC Curves
ax = axes[1][1]
for (name, res), col in zip(results.items(), COLORS):
    prob     = res['model'].predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, color=col, lw=2,
            label=f'{name[:2]} AUC={auc(fpr,tpr):.3f}')
ax.plot([0,1],[0,1],'--', color='gray', lw=1)
ax.set_title('ROC Curves'); ax.set_xlabel('FPR', color='white')
ax.set_ylabel('TPR', color='white')
ax.legend(facecolor=CARD, labelcolor='white', fontsize=8); style(ax)

# Plot 6 – Feature Importances
ax = axes[1][2]
fi   = results['Random Forest']['model'].feature_importances_
fi_s = pd.Series(fi, index=FEATURES).sort_values()
cols_fi = [COLORS[0] if v >= fi_s.median() else COLORS[4] for v in fi_s.value]

ax.barh(fi_s.index, fi_s.values, color=cols_fi, edgecolor='none')
ax.set_title('Feature Importances (RF)')
ax.set_xlabel('Importance', color='white'); style(ax)

plt.tight_layout()
plt.savefig('titanic_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#0d0d1f')
plt.show()
print("\n✅ Dashboard saved as titanic_dashboard.png")
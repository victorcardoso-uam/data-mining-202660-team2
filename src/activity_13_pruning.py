import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('data/processed/pruning_team2_semiconductor.csv')

X = df.drop(columns=['wafer_defect_flag'])
y = df['wafer_defect_flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

alphas = [0.001, 0.015, 0.080]

print(f"{'Alpha':<10} | {'Leaves (|T|)':<12} | {'Max Depth':<10} | {'Train Error':<12} | {'Test Error':<10} | {'Total Cost R_alpha(T)':<20}")
print("-" * 85)

for alpha in alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf.fit(X_train, y_train)
    
    leaves = clf.get_n_leaves()
    depth = clf.get_depth()
    
    train_error = 1.0 - clf.score(X_train, y_train)
    test_error = 1.0 - clf.score(X_test, y_test)
    
    total_cost = test_error + (alpha * leaves)
    
    print(f"{alpha:<10.3f} | {leaves:<12} | {depth:<10} | {train_error:<12.4f} | {test_error:<10.4f} | {total_cost:<20.4f}")
    
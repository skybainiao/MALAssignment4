import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans, DBSCAN


file_paths = {
    "alldata": "C:/Users/45527/Desktop/OneDrive_1_11-23-2023/alldata.xlsx",
    "drdata": "C:/Users/45527/Desktop/OneDrive_1_11-23-2023/drdata.xlsx",
    "drq": "C:/Users/45527/Desktop/OneDrive_1_11-23-2023/drq.xlsx",
    "tv2data": "C:/Users/45527/Desktop/OneDrive_1_11-23-2023/tv2data.xlsx",
    "tv2q": "C:/Users/45527/Desktop/OneDrive_1_11-23-2023/tv2q.xlsx",
    "electeddata": "C:/Users/45527/Desktop/OneDrive_1_11-23-2023/electeddata.xlsx"
}


dataframes = {name: pd.read_excel(path) for name, path in file_paths.items()}


file_paths = {
    "alldata": "/mnt/data/alldata.xlsx",
    "drdata": "/mnt/data/drdata.xlsx",
    "drq": "/mnt/data/drq.xlsx",
    "tv2data": "/mnt/data/tv2data.xlsx",
    "tv2q": "/mnt/data/tv2q.xlsx",
    "electeddata": "/mnt/data/electeddata.xlsx"
}

dataframes = {name: pd.read_excel(path) for name, path in file_paths.items()}

for df in dataframes.values():
    response_columns = df.columns[:-4]  # 假设最后四列是非响应数据
    new_column_names = {old: f"Q{index+1}" for index, old in enumerate(response_columns)}
    df.rename(columns=new_column_names, inplace=True)

for df in dataframes.values():
    if 'parti' in df.columns:
        df['parti'] = df['parti'].astype('category')
    if 'storkreds' in df.columns:
        df['storkreds'] = df['storkreds'].astype('category')

for name, df in dataframes.items():

    print(f"{name} - 缺失值数量: {df.isnull().sum().sum()}")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(dataframes['alldata'][numeric_features])  # 假设 numeric_features 是数值特征列
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title('PCA of Candidate Responses')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

std_dev_questions = dataframes['alldata'][numeric_features].std().sort_values(ascending=False)

average_positions = dataframes['alldata'].groupby('parti')[numeric_features].mean()
average_positions_transposed = average_positions.T
plt.figure(figsize=(15, 10))
sns.heatmap(average_positions_transposed, cmap='coolwarm', annot=False)
plt.title('Average Positions of Parties on Each Question')
plt.xlabel('Party')
plt.ylabel('Question')
plt.show()

average_age_per_party = dataframes['alldata'].groupby('parti')['alder'].mean()
plt.figure(figsize=(10, 6))
average_age_per_party.sort_values().plot(kind='bar')
plt.title('Average Age of Candidates by Party')
plt.xlabel('Party')
plt.ylabel('Average Age')
plt.show()

extreme_responses = dataframes['alldata'][numeric_features].isin([-2, 2])
proportion_extreme_responses = extreme_responses.mean(axis=1)
proportion_extreme_responses_with_names = pd.concat([dataframes['alldata']['navn'], proportion_extreme_responses],
                                                    axis=1)
proportion_extreme_responses_with_names.columns = ['Candidate Name', 'Proportion of Extreme Responses']
most_confident_candidates = proportion_extreme_responses_with_names.sort_values(by='Proportion of Extreme Responses',
                                                                                ascending=False)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, dataframes['alldata']['parti'], test_size=0.3,
                                                    random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
report_dt = classification_report(y_test, y_pred_dt)

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
report_gb = classification_report(y_test, y_pred_gb)

inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_scaled)


pca_elected = PCA(n_components=2)
principal_components_elected = pca_elected.fit_transform(dataframes['electeddata'][numeric_response_columns_elected])
pca_elected_df = pd.DataFrame(data=principal_components_elected, columns=['PC1', 'PC2'])
pca_elected_df['parti'] = dataframes['electeddata']['parti']
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='parti', data=pca_elected_df, palette='tab10', legend='full')
plt.title('Political Landscape of Elected Candidates')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

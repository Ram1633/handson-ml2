#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


import sklearn
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import pandas_profiling
from pandas_profiling import ProfileReport

import seaborn as sns


# # Get the data

# In[2]:


life = pd.read_csv("D:\OneDrive\Desktop\INSY 695 - Adv Topics in Enterprise ML\Assignment 1\Life Expectancy Data.csv")


# In[3]:


life.head()


# In[4]:


life.info()


# In[5]:


life.describe()


# In[6]:


life.hist(bins=50, figsize=(20,15))
plt.show()


# In[7]:


life["Status"].value_counts()


# Status of a country should be an important predictor. Since there is significant imbalance in proportion of developed and developing countries in the dataset, I have split the dataset according to the country status

# In[8]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(life, life["Status"]):
    strat_train_set = life.loc[train_index]
    strat_test_set = life.loc[test_index]


# In[9]:


strat_test_set["Status"].value_counts() / len(strat_test_set)


# In[10]:


life["Status"].value_counts() / len(life)


# In[11]:


def Status_proportions(data):
    return data["Status"].value_counts() / len(data)

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(life, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": Status_proportions(life),
    "Stratified": Status_proportions(strat_test_set),
    "Random": Status_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100


# In[12]:


compare_props


# # Discover and visualize the data to gain insights

# In[13]:


life_c = strat_train_set[pd.notnull(strat_train_set['exp'])].copy()


# Plotting all numerical attributes in the dataset

# In[14]:


from pandas.plotting import scatter_matrix

attributes = ["exp","Adult_Mortality","Schooling","Income composition of resources","Total_expenditure"]
#attributes = list(life_c.columns)[3:]
scatter_matrix(life_c[attributes], figsize=(30,30))


# Schooling is highly corelated with Life expectancy and is a promising predictor. Also, developed countries consistently had higher schooling and life expectancy

# In[15]:


plt.figure(figsize=(15,7))
sns.scatterplot('Schooling', 'exp', data=life_c, hue='Status')


# Life Expectancy had a consistent increasing trend between 2000 and 2014 for all countries

# In[16]:


plt.figure(figsize=(15,7))
plt.subplot(121)
plt.bar(life_c.groupby('Year')['Year'].count().index,life_c.groupby('Year')['exp'].mean(),color='blue',alpha=0.5)
plt.xlabel("Year",fontsize=12)
plt.ylabel("Avg Life_Expectancy",fontsize=12)
plt.title("Life_Expectancy w.r.t Year")

plt.subplot(122)
sns.lineplot('Year', 'exp', data=life_c, marker='o')
plt.title('Life Expectancy by Year')
plt.show()


# Here, inter-relationship between Schooling, Income composition and Adult Mortality has been investigated.

# In[17]:


fig, axes = plt.subplots(nrows=3, ncols=1)

life_c.plot(ax=axes[0],kind="scatter",x="Schooling",y="Income composition of resources",c='exp',cmap=plt.get_cmap("jet"),figsize=(10,20))
plt.title("Schooling vs Income composition of resources")


life_c.plot(ax=axes[1],kind="scatter",x="Schooling",y="Adult_Mortality",c='exp',cmap=plt.get_cmap("jet"),figsize=(10,20))
plt.title("Schooling vs Adult_Mortality")


life_c.plot(ax=axes[2],kind="scatter",x="Adult_Mortality",y="Income composition of resources",c='exp',cmap=plt.get_cmap("jet"),figsize=(10,20))
plt.title("Adult_Mortality vs Adult_Mortality")


# GDP and Income seems to be correlated with Life expectancy higher for observations having high GDP and Income

# In[18]:


plt.figure(figsize=(10,7))
sns.scatterplot('Income composition of resources', 'GDP', data=life_c, hue='exp')


# Developed countries were significantly smaller in size but had a higher life expectancy

# In[19]:


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.bar(life_c.groupby('Status')['Status'].count().index,life_c.groupby('Status')['exp'].mean())
plt.title('Average Life Expectancy by Country Status')
plt.xlabel('Country Status')
plt.ylabel('Average Life Expectancy')
plt.xticks(rotation=0)

plt.subplot(122)
life_c.Status.value_counts().plot(kind='pie', autopct='%.2f')
plt.ylabel('')
plt.title('Country Status Pie Chart')

plt.show()


# We can also see a significant number of outliers in the dataset for most of the attributes

# In[20]:


plt.figure(figsize=(15,10))
for i, col in enumerate(['Adult_Mortality', 'infant_deaths', 'BMI', 'under-five_deaths', 'GDP', 'Population'], start=1):
    plt.subplot(2, 3, i)
    life_c.boxplot(col)


# # Prepare the data for Machine Learning algorithms

# In[21]:


life_labels = life_c["exp"].copy()
life_c = life_c.drop(["exp","Country"], axis=1)


# In[22]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

life_num = life_c.drop(["Status"], axis=1)
life_cat = life_c[["Status"]]

imputer.fit(life_num)


# In[23]:


X = imputer.transform(life_num)

life_tr = pd.DataFrame(X, columns=life_num.columns,
                          index=life_num.index)


# In[24]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
stat_1hot = cat_encoder.fit_transform(life_cat)
stat_1hot


# In[25]:


stat_1hot.toarray()


# Pipeline for preprocessing the numerical attributes:

# In[26]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
life_num_tr = num_pipeline.fit_transform(life_num)


# In[27]:


life_num_tr


# In[28]:


from sklearn.compose import ColumnTransformer

num_attribs = list(life_c.drop(["Status"], axis=1))
cat_attribs = ["Status"]

full_pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), cat_attribs),
        ("num", num_pipeline, num_attribs),
    ])

life_prepared = full_pipeline.fit_transform(life_c)


# In[29]:


life_prepared


# # Select and train a model 

# In[30]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(life_prepared, life_labels)


# In[31]:


some_data = life_c.iloc[:5]
some_labels = life_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))


# In[32]:


from sklearn.metrics import mean_squared_error

life_predictions = lin_reg.predict(life_prepared)
lin_mse = mean_squared_error(life_labels, life_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[33]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(life_prepared, life_labels)


# In[34]:


life_predictions = tree_reg.predict(life_prepared)
tree_mse = mean_squared_error(life_labels, life_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# # Fine-tune your model

# In[35]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, life_prepared, life_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[36]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# In[37]:


lin_scores = cross_val_score(lin_reg, life_prepared, life_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[38]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(life_prepared, life_labels)


# In[39]:


life_predictions = forest_reg.predict(life_prepared)
forest_mse = mean_squared_error(life_labels, life_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[40]:


from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, life_prepared, life_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[41]:


scores = cross_val_score(lin_reg, life_prepared, life_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()


# In[42]:


from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(life_prepared, life_labels)
life_predictions = svm_reg.predict(life_prepared)
svm_mse = mean_squared_error(life_labels, life_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse


# In[43]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(life_prepared, life_labels)


# In[44]:


grid_search.best_params_


# In[45]:


grid_search.best_estimator_


# In[46]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[47]:


pd.DataFrame(grid_search.cv_results_)


# In[48]:


feature_importances = grid_search.best_estimator_.feature_importances_


# In[49]:


feature_importances


# In[50]:


cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# In[51]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(life_prepared, life_labels)


# In[52]:


rnd_search.best_params_


# In[53]:


cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[54]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[55]:


cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# In[56]:


final_model = rnd_search.best_estimator_

X_test = strat_test_set.drop(["exp","Country"], axis=1)
y_test = strat_test_set["exp"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_rmse


# In[57]:


from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))


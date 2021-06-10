# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
# %%
df_orig_train = pd.read_csv('./data/train.csv')
df_orig_test = pd.read_csv('./data/test.csv')
# %%
print(df_orig_train.columns)
# %%
print(df_orig_train['조식메뉴'][0])
# %%
df_orig_train.head()
# %%
df_orig_test.head()

# %%
date_mapper = {'월': 'Monday', '화':'Tuesday', '수':'Wednesday', '목':'Thursday', '금':'Friday', '토':'Saturday', '일':'Sunday'}
df_orig_train['요일'] = df_orig_train['요일'].map(date_mapper)
df_orig_train.head()
# %%
import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x='요일', y='중식계', data=df_orig_train, ax=ax)
plt.show()
# %%
import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x='요일', y='석식계', data=df_orig_train, ax=ax)
plt.show()
# %%
import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x='요일', y='본사출장자수', data=df_orig_train, ax=ax)
plt.show()
# %%
import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 10))
sns.boxplot(x='요일', y='본사시간외근무명령서승인건수', data=df_orig_train, ax=ax)
plt.show()
# %%

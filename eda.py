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
df_orig_train['조식메뉴'][0]
# %%
df_orig_train['중식메뉴'][0]
# %%
sample_text = df_orig_train['중식메뉴'][0]
print(sample_text)

# %%
sample_text.split('/')
# %%
import re
processed_1 = re.subn(r"[(<]\s*(\w+[,]*)*\w*\s*[:]*\s*[/]*(\w+[,]*)*\s*[:]*\s*(\w+[,]*)*\s*[)>]", " ", sample_text)[0]
results = re.sub("\s+", " ", processed_1)
print(results)
# %%

df_orig_train[['중식메뉴', '석식메뉴']]
# %%
df_orig_train[['중식메뉴']].to_csv('lunch.tsv', index=False, sep='\t')
df_orig_train[['석식메뉴']].to_csv('dinner.tsv', index=False, sep='\t')
# %%

# %%
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)

# %%
df_train = pd.read_csv("./preprocessed_data/preprocessed_train.csv")
df_test = pd.read_csv("./preprocessed_data/preprocessed_test.csv")
submission = pd.read_csv('./data/sample_submission.csv')

# %%
from sklearn.model_selection import train_test_split

train_set, dev_set = train_test_split(df_train, train_size=0.3)
# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rf_model_lunch = RandomForestRegressor(n_jobs=-1, n_estimators=50, random_state=1004)

feature_mask = ['식사대상자', '요일_lunch', '본사시간외근무명령서승인건수']

rf_model_lunch.fit(train_set[feature_mask], train_set['중식계'])

print(mean_absolute_error(rf_model_lunch.predict(train_set[feature_mask]), train_set['중식계']))
print(mean_absolute_error(rf_model_lunch.predict(dev_set[feature_mask]), dev_set['중식계']))

rf_model_dinner = RandomForestRegressor(n_jobs=-1, n_estimators=50, random_state=1004)

feature_mask = ['식사대상자', '요일_dinner', '본사시간외근무명령서승인건수']

rf_model_dinner.fit(train_set.loc[train_set['석식계'] != 0, feature_mask], train_set.loc[train_set['석식계'] != 0, '석식계'])
print(mean_absolute_error(rf_model_dinner.predict(train_set.loc[train_set['석식계'] != 0, feature_mask]), train_set.loc[train_set['석식계'] != 0, '중식계']))
print(mean_absolute_error(rf_model_dinner.predict(dev_set.loc[dev_set['석식계'] != 0, feature_mask]), dev_set.loc[dev_set['석식계'] != 0, '중식계']))

# %%

lunch_pred = rf_model_lunch.predict(df_test[['식사대상자', '요일_lunch', '본사시간외근무명령서승인건수']])
dinner_pred = rf_model_dinner.predict(df_test[['식사대상자', '요일_dinner', '본사시간외근무명령서승인건수']])

submission['중식계'] = lunch_pred
submission['석식계'] = dinner_pred

submission.to_csv('rf_model_add_feature.csv', index=False)
# %%

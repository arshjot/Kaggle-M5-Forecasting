import numpy as np
import pandas as pd
import category_encoders as ce
import pickle as pkl
pd.options.mode.chained_assignment = None


def read_data(input_data_dir='../../data/', output_dir='./'):
    train_data = pd.read_csv(f'{input_data_dir}/sales_train_validation.csv')
    sell_prices = pd.read_csv(f'{input_data_dir}/sell_prices.csv')
    calendar = pd.read_csv(f'{input_data_dir}/calendar.csv')

    # ---- process calendar features ---- #
    print('* Processing calendar features')

    calendar.date = pd.to_datetime(calendar.date)
    calendar['relative_year'] = 2016 - calendar.year

    # convert month, day and weekday to cyclic encodings
    calendar['month_sin'] = np.sin(2 * np.pi * calendar.month/12.0)
    calendar['month_cos'] = np.cos(2 * np.pi * calendar.month/12.0)
    calendar['day_sin'] = np.sin(2 * np.pi * calendar.date.dt.day/calendar.date.dt.days_in_month)
    calendar['day_cos'] = np.cos(2 * np.pi * calendar.date.dt.day/calendar.date.dt.days_in_month)
    calendar['weekday_sin'] = np.sin(2 * np.pi * calendar.wday/7.0)
    calendar['weekday_cos'] = np.cos(2 * np.pi * calendar.wday/7.0)

    one_day_events = ['SuperBowl', 'ValentinesDay', 'PresidentsDay', 'StPatricksDay',
                      'OrthodoxEaster', 'Cinco De Mayo', "Mother's day", 'MemorialDay',
                      "Father's day", 'IndependenceDay', 'Eid al-Fitr', 'LaborDay',
                      'ColumbusDay', 'Halloween', 'EidAlAdha', 'VeteransDay',
                      'Thanksgiving', 'Christmas', 'NewYear', 'OrthodoxChristmas',
                      'MartinLutherKingDay', 'Easter']
    multi_day_events = ['LentStart', 'LentWeek2', 'Purim End', 'Pesach End',
                        'NBAFinalsStart', 'NBAFinalsEnd', 'Ramadan starts', 'Chanukah End']

    # create separate columns for each event
    for event in one_day_events:
        calendar[event] = [1 if val == event else 0 for val in calendar.event_name_1]
        calendar.loc[calendar.event_name_2 == event, event] = 1

    calendar['Lent'] = [1 if val == 'LentStart' else 0 for val in calendar.event_name_1]
    calendar.loc[calendar.event_name_2 == 'LentStart', 'Lent'] = 1
    calendar['Purim'] = [1 if val == 'Purim End' else 0 for val in calendar.event_name_1]
    calendar.loc[calendar.event_name_2 == 'Purim End', 'Purim'] = 1
    calendar['Pesach'] = [1 if val == 'Pesach End' else 0 for val in calendar.event_name_1]
    calendar.loc[calendar.event_name_2 == 'Pesach End', 'Pesach'] = 1
    calendar['Ramadan'] = [1 if val == 'Ramadan starts' else 0 for val in calendar.event_name_1]
    calendar.loc[calendar.event_name_2 == 'Ramadan starts', 'Ramadan'] = 1
    calendar['Chanukah'] = [1 if val == 'Chanukah End' else 0 for val in calendar.event_name_1]
    calendar.loc[calendar.event_name_2 == 'Chanukah End', 'Chanukah'] = 1

    calendar['NBAFinals'] = [1 if (val == 'NBAFinalsStart') else None for val in calendar.event_name_1]
    calendar.loc[(calendar.event_name_2 == 'NBAFinalsStart'), 'NBAFinals'] = 1
    calendar.loc[
        (calendar.event_name_1 == 'NBAFinalsEnd') | (calendar.event_name_2 == 'NBAFinalsEnd'), 'NBAFinals'] = 0

    # for multi-day events, fill value as 1 from start to end
    # Lent ends approx 6 weeks from the start
    calendar['Lent'] = calendar['Lent'].rolling(min_periods=1, window=7 * 6).sum()
    # Purim lasts just 2 days
    calendar['Purim'] = calendar['Purim'].shift(-1).rolling(min_periods=1, window=2).sum()
    # Purim usually lasts for 9 days
    calendar['Pesach'] = calendar['Pesach'].shift(-8).rolling(min_periods=1, window=9).sum()
    # both start and end dates for NBA Finals have been given
    calendar['NBAFinals'] = calendar['NBAFinals'].fillna(method='ffill').fillna(0)
    calendar.loc[
        (calendar.event_name_1 == 'NBAFinalsEnd') | (calendar.event_name_2 == 'NBAFinalsEnd'), 'NBAFinals'] = 1
    # Ramadan ends approx 30 days from the start
    calendar['Ramadan'] = calendar['Ramadan'].rolling(min_periods=1, window=30).sum()
    # Chanukah lasts for 9 days
    calendar['Chanukah'] = calendar['Chanukah'].shift(-8).rolling(min_periods=1, window=9).sum()

    calendar_df = calendar[['wm_yr_wk', 'd', 'snap_CA', 'snap_TX', 'snap_WI', 'relative_year',
                            'month_sin', 'month_cos', 'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos',
                            'SuperBowl', 'ValentinesDay', 'PresidentsDay', 'StPatricksDay', 'OrthodoxEaster',
                            'Cinco De Mayo', "Mother's day", 'MemorialDay', "Father's day", 'IndependenceDay',
                            'Eid al-Fitr', 'LaborDay', 'ColumbusDay', 'Halloween', 'EidAlAdha', 'VeteransDay',
                            'Thanksgiving', 'Christmas', 'NewYear', 'OrthodoxChristmas', 'MartinLutherKingDay',
                            'Easter', 'Lent', 'Purim', 'Pesach', 'Ramadan', 'Chanukah', 'NBAFinals']]

    # ---- Merge all dfs, keep calender_df features separate and just concat them for each batch ---- #
    train_data.id = train_data.id.str[:-11]
    sell_prices['id'] = sell_prices['item_id'] + '_' + sell_prices['store_id']

    # add empty columns for future data
    train_data = pd.concat([train_data, pd.DataFrame(columns=['d_'+str(i) for i in range(1914, 1970)])])

    # Encode categorical features using either one-hot or label encoding (for embeddings)
    print('* Encoding categorical features')
    one_hot = ['cat_id', 'state_id']
    label = ['item_id', 'dept_id', 'store_id']
    label_encoded_cols = [str(i)+'_enc' for i in label]

    train_data[[str(i)+'_enc' for i in one_hot]] = train_data[one_hot]
    one_hot_encoder = ce.OneHotEncoder(cols=[str(i)+'_enc' for i in one_hot], use_cat_names=True)
    one_hot_encoder.fit(train_data)
    train_data = one_hot_encoder.transform(train_data)

    train_data[label_encoded_cols] = train_data[label]
    label_encoder = ce.OrdinalEncoder(cols=[str(i)+'_enc' for i in label])
    label_encoder.fit(train_data)
    train_data = label_encoder.transform(train_data)

    # subtract one from label encoded as pytorch uses 0-indexing
    for col in label_encoded_cols:
        train_data[col] = train_data[col] - 1

    # Reshape, change dtypes and add previous day sales
    print('* Add previous day sales and merge sell prices')
    data_df = pd.melt(train_data, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                                           'cat_id_enc_HOBBIES', 'cat_id_enc_HOUSEHOLD', 'cat_id_enc_FOODS',
                                           'state_id_enc_CA', 'state_id_enc_TX', 'state_id_enc_WI',
                                           'item_id_enc', 'dept_id_enc', 'store_id_enc'],
                      var_name='d', value_vars=['d_'+str(i) for i in range(1, 1970)], value_name='sales')

    # change dtypes to reduce memory usage
    data_df[['sales']] = data_df[['sales']].fillna(-2).astype(np.int16)  # fill future sales as -2
    calendar_df[one_day_events + ['Lent', 'Purim', 'Pesach', 'Ramadan', 'Chanukah', 'NBAFinals',
                                  'snap_CA', 'snap_TX', 'snap_WI', 'relative_year']] = calendar_df[
        one_day_events + ['Lent', 'Purim', 'Pesach', 'Ramadan', 'Chanukah', 'NBAFinals',
                          'snap_CA', 'snap_TX', 'snap_WI', 'relative_year']].astype(np.int8)

    data_df[['cat_id_enc_HOBBIES', 'cat_id_enc_HOUSEHOLD', 'cat_id_enc_FOODS', 'state_id_enc_CA',
             'state_id_enc_TX', 'state_id_enc_WI']] = data_df[['cat_id_enc_HOBBIES', 'cat_id_enc_HOUSEHOLD',
                                                               'cat_id_enc_FOODS', 'state_id_enc_CA',
                                                               'state_id_enc_TX', 'state_id_enc_WI']].astype(np.int8)

    data_df[['item_id_enc', 'dept_id_enc', 'store_id_enc']] = data_df[['item_id_enc', 'dept_id_enc',
                                                                       'store_id_enc']].astype(np.int16)

    # merge sell prices
    data_df = data_df.merge(right=calendar_df[['d', 'wm_yr_wk']], on=['d'], how='left')
    data_df = data_df.merge(right=sell_prices[['id', 'wm_yr_wk', 'sell_price']], on=['id', 'wm_yr_wk'], how='left')

    data_df.sell_price = data_df.sell_price.fillna(0.0)
    data_df['prev_day_sales'] = data_df.groupby(['id'])['sales'].shift(1)

    # remove data for d_1
    data_df.dropna(axis=0, inplace=True)
    calendar_df = calendar_df[calendar_df.d != 'd_1']

    # change dtypes
    data_df[['prev_day_sales']] = data_df[['prev_day_sales']].astype(np.int16)

    # ---- Add previous day totals of aggregated series as features ---- #
    print('* Add previous day totals of aggregated series as features')
    # total
    data_df = data_df.merge(right=
                            data_df.groupby(['d'])[['prev_day_sales']].sum().astype(
                                np.int32).add_suffix('_all').reset_index(),
                            on=['d'], how='left')
    # category level
    data_df = data_df.merge(right=data_df.groupby(['d', 'cat_id'])[['prev_day_sales']].sum().astype(
                                np.int32).reset_index().pivot(
                                index='d', columns='cat_id', values='prev_day_sales').add_prefix('prev_d_cat_'),
                            on=['d'], how='left')
    # state level
    data_df = data_df.merge(right=
                            data_df.groupby(['d', 'state_id'])[['prev_day_sales']].sum().astype(
                                np.int32).reset_index().pivot(
                                index='d', columns='state_id', values='prev_day_sales').add_prefix('prev_d_state_'),
                            on=['d'], how='left')
    # store level
    data_df = data_df.merge(right=
                            data_df.groupby(['d', 'store_id'])[['prev_day_sales']].sum().astype(
                                np.int32).reset_index().pivot(
                                index='d', columns='store_id', values='prev_day_sales').add_prefix('prev_d_store_'),
                            on=['d'], how='left')
    # department level
    data_df = data_df.merge(right=
                            data_df.groupby(['d', 'dept_id'])[['prev_day_sales']].sum().astype(
                                np.int32).reset_index().pivot(
                                index='d', columns='dept_id', values='prev_day_sales').add_prefix('prev_d_dept_'),
                            on=['d'], how='left')

    # remove category columns
    del data_df['wm_yr_wk']
    del data_df['item_id']
    del data_df['dept_id']
    del data_df['cat_id']
    del data_df['store_id']
    del data_df['state_id']

    num_samples = data_df.id.nunique()
    num_timesteps = data_df.d.nunique()
    data_df = data_df.set_index(['id', 'd'])
    
    ids = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    enc_dec_feats = ['sell_price', 'cat_id_enc_HOBBIES', 'cat_id_enc_HOUSEHOLD', 'cat_id_enc_FOODS', 'state_id_enc_CA',
                     'state_id_enc_TX', 'state_id_enc_WI', 'item_id_enc', 'dept_id_enc', 'store_id_enc']
    enc_only_feats = data_df.columns.difference(['sales', 'sell_price', 'prev_day_sales'] + enc_dec_feats)

    sales_data_ids = train_data[ids].values
    Y = data_df.sales.values.reshape(num_timesteps, num_samples).T
    X_enc_only_feats = np.array(data_df[enc_only_feats]).reshape(num_timesteps, num_samples, -1)
    X_enc_dec_feats = np.array(data_df[enc_dec_feats]).reshape(num_timesteps, num_samples, -1)
    X_prev_day_sales = data_df.prev_day_sales.values.reshape(num_timesteps, num_samples)
    calendar_index = calendar_df.d
    X_calendar = np.array(calendar_df.iloc[:, 2:])
    X_calendar_cols = list(calendar_df.columns[2:])

    # for prev_day_sales and sales (y), set value as -1 for the period the product was not actively sold
    for idx, first_non_zero_idx in enumerate((X_prev_day_sales != 0).argmax(axis=0)):
        X_prev_day_sales[:first_non_zero_idx, idx] = -1
    for idx, first_non_zero_idx in enumerate((Y != 0).argmax(axis=1)):
        Y[idx, :first_non_zero_idx] = -1

    # ---- Save processed data ---- #
    print('* Save processed data')
    data_dict = {'sales_data_ids': sales_data_ids, 'calendar_index': calendar_index,
                 'X_prev_day_sales': X_prev_day_sales,
                 'X_enc_only_feats': X_enc_only_feats, 'X_enc_dec_feats' : X_enc_dec_feats,
                 'enc_dec_feat_names': enc_dec_feats, 'enc_only_feat_names': enc_only_feats,
                 'X_calendar': X_calendar, 'X_calendar_cols': X_calendar_cols,
                 'Y': Y,
                 'one_hot_encoder': one_hot_encoder, 'label_encoder': label_encoder}

    # pickle data
    with open(f'{output_dir}/data.pickle', 'wb') as f:
        pkl.dump(data_dict, f, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print('Processing Data:\n')
    read_data()
    print('\nCompleted')

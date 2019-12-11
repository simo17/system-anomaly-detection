import pandas as pd
import re
from collections import Counter
from sklearn import preprocessing
from utils import train_test_split

def compute_label(df):
    if df['Count'] > 0:
        return 1
    else:
        return 0

def create_feature_mat(seq_label_df):
    feature_mat = pd.DataFrame()
    feature_mat['Label'] = seq_label_df['Label']
    for idx, row in seq_label_df.iterrows():
        event_seq = row['EventId'].split(',')
        count_map = Counter(event_seq)
        for event in count_map:
            feature_mat.at[idx, event] = count_map[event]
    feature_mat = feature_mat.fillna(0)
    return feature_mat

def process_data(log_file, label_file, window_size, normalization=True):
    # read log and label
    log_data = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True).filter(
        ['Date', 'Time', 'Content', 'EventId'], axis=1)

    label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
    label_data = label_data.set_index('BlockId')
    label_dict = label_data['Label'].to_dict()

    # format data\etime
    log_data = log_data.applymap(str)
    log_data['Datetime'] = '200' + log_data['Date'] + log_data['Time']
    log_data['Datetime'] = pd.to_datetime(log_data['Datetime'], format='%Y%m%d%H%M%S')
    log_data = log_data.drop(['Date', 'Time'], 1)

    # label log
    log_data['Label'] = 1
    for idx, row in log_data.iterrows():
        content = row['Content']
        blk = re.findall(r'(blk_-?\d+)', content)[0]
        if label_dict[blk] == 'Anomaly':
            log_data.at[idx, 'Label'] = 1
        else:
            log_data.at[idx, 'Label'] = 0

    log_data = log_data.sort_values('Datetime')
    log_data = log_data.reset_index(drop=True)

    time_series_data = log_data.groupby('Datetime')['Label'].sum()
    time_series_data.columns = ['Datetime', 'Count']
    time_series_data_df = pd.DataFrame({'Datetime': time_series_data.index, 'Count': time_series_data.values})

    time_series_data_df['Label'] = time_series_data_df.apply(compute_label, axis=1)
    # set datetime as index
    time_series_data_df = time_series_data_df.set_index('Datetime')
    # use data from 2008-11-09～2008-11-11
    time_series_data_df = time_series_data_df.loc['2008-11-09':'2008-11-11']

    # generate feature map
    mask = (log_data['Datetime'] >= '2008-11-09') & (log_data['Datetime'] <= '2008-11-12')
    log_data_use = log_data.loc[mask]

    log_data_use = log_data_use.reset_index(drop=True)
    log_data_use = log_data_use.drop('Content', 1)

    event_seq = log_data_use.groupby('Datetime')['EventId'].apply(','.join).reset_index()

    assert event_seq.shape[0] == time_series_data_df.shape[0]

    event_seq = event_seq.set_index('Datetime')
    seq_label_df = pd.concat([event_seq, time_series_data_df], axis=1)

    # consecutive = seq_label_df.loc['2008-11-10 03:00:00':'2008-11-10 23:59:59']

    window = str(window_size) + 's'
    event_seq = seq_label_df.groupby(pd.Grouper(freq=window))['EventId'].apply(','.join).reset_index()
    event_seq = event_seq.set_index('Datetime')
    label = seq_label_df.groupby(pd.Grouper(freq=window))['Label'].sum().reset_index()
    label.loc[label['Label'] > 0, 'Label'] = 1
    label = label.set_index('Datetime')
    seq_label_df = pd.concat([event_seq, label], axis=1)
    feature_mat = create_feature_mat(seq_label_df)
    df_x = feature_mat.drop('Label', 1)
    X = df_x.values
    Y = feature_mat['Label'].values

    (x_train, y_train), (x_test, y_test) = train_test_split(X, Y, 0.8)
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)

    print('Window size is ' + str(window_size) + 's.')
    print('Train: {} instances, {} anomaly, {} normal'.format(num_train, num_train_pos, num_train - num_train_pos))
    print('Percentage of Anomaly in Train: {:.3f}%'.format(num_train_pos * 100 / num_train))
    print('Test: {} instances, {} anomaly, {} normal'.format(num_test, num_test_pos, num_test - num_test_pos))
    print('Percentage of Anomaly in Test: {:.3f}%'.format(num_test_pos * 100 / num_test))

    if normalization:
        std_scale = preprocessing.StandardScaler().fit(x_train)
        x_train = std_scale.transform(x_train)
        x_test = std_scale.transform(x_test)

    return (x_train, y_train), (x_test, y_test)




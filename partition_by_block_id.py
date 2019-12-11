from collections import Counter
import pandas as pd
import numpy as np
import re
from collections import OrderedDict
from sklearn.utils import resample
from utils import train_test_split

class FeatureExtractor(object):
    def __init__(self):
        self.mean_vec = None
        self.events = None
        self.empty_events = None

    def transform_train(self, x_train):

        count = []
        for i in range(x_train.shape[0]):
            event_counts = Counter(x_train[i])
            count.append(event_counts)

        x_df = pd.DataFrame(count)
        x_df = x_df.fillna(0)

        self.events = x_df.columns
        x = x_df.values

        num_instance, num_event = x.shape

        mean_vec = x.mean(axis=0)
        self.mean_vec = mean_vec.reshape(1, num_event)
        x = x - np.tile(self.mean_vec, (num_instance, 1))


        return x

    def transform_test(self, x_test):

        count = []
        for i in range(x_test.shape[0]):
            event_counts = Counter(x_test[i])
            count.append(event_counts)
        x_df = pd.DataFrame(count)
        x_df = x_df.fillna(0)

        self.empty_events = set(self.events) - set(x_df.columns)

        for event in self.empty_events:
            x_df[event] = [0] * len(x_df)
        x = x_df[self.events].values

        num_instance, num_event = x.shape

        x = x - np.tile(self.mean_vec, (num_instance, 1))

        return x

    def find_unused_events(self, log_templates):
        templates = pd.read_csv(log_templates)
        all_events = templates['EventId'].values
        self.unused_events = set(self.events) - set(all_events)

        return None

def load_data(log_file, label_file, train_ratio, oversampling):
    struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
    data_dict = OrderedDict()
    for idx, row in struct_log.iterrows():
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = []
            data_dict[blk_Id].append(row['EventId'])
    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

    label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
    label_data = label_data.set_index('BlockId')
    label_dict = label_data['Label'].to_dict()
    data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

    # Split train and test data
    (x_train, y_train), (x_test, y_test) = train_test_split(data_df['EventSequence'].values, data_df['Label'].values, train_ratio)

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)

    print('Train: {} instances, {} anomaly, {} normal'.format(num_train, num_train_pos, num_train - num_train_pos))
    print('Percentage of Anomaly in Train: {}%\n'.format(num_train_pos * 100 / num_train))

    print('Test: {} instances, {} anomaly, {} normal'.format(num_test, num_test_pos, num_test - num_test_pos))
    print('Percentage of Anomaly in Test: {}%\n'.format(num_test_pos * 100 / num_test))

    if oversampling:
        # upsample
        # https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18
        X = pd.DataFrame({'EventSequence': x_train, 'Label': y_train}, columns=['EventSequence', 'Label'])
        normal = X.loc[X['Label'] == 0]
        abnormal = X.loc[X['Label'] == 1]
        abnormal_upsampled = resample(abnormal,
                                      replace=True,  # sample with replacement
                                      n_samples=len(normal))  # match number in majority class
        upsampled = pd.concat([normal, abnormal_upsampled])

        upsampled_x_train = upsampled['EventSequence'].values
        upsampled_y_train = upsampled['Label'].values

        num_upsampled_pos = sum(upsampled_y_train)
        num_upsampled = upsampled_x_train.shape[0]

        print('Upsampled Train: {} instances, {} anomaly, {} normal'.format(num_upsampled, num_upsampled_pos, num_upsampled - num_upsampled_pos))
        print('Percentage of Anomaly in Upsampled Train: {} %\n'.format(num_upsampled_pos * 100 / num_upsampled))



        return (upsampled_x_train, upsampled_y_train), (x_test, y_test)
    else:
        return (x_train, y_train), (x_test, y_test)


def process_data(log_file, label_file, train_ratio=0.8, oversampling=False):
    (x_train, y_train), (x_test, y_test) = load_data(log_file, label_file, train_ratio, oversampling)
    feature_extractor = FeatureExtractor()

    x_train = feature_extractor.transform_train(x_train)
    x_test = feature_extractor.transform_test(x_test)

    return (x_train, y_train), (x_test, y_test)
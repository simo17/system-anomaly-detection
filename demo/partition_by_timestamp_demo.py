from partition_by_timestamp import process_data
from models.random_forest import RandomForest

log_file = '../sample_data/Small_HDFS_parsed.csv'
label_file = '../label/anomaly_label.csv'

(x_train_std, y_train), (x_test_std, y_test) = process_data(log_file, label_file, window_size = 2, normalization=True)

model = RandomForest(n_estimators=100, max_depth=10, bootstrap=True, oob_score=True)
model.fit(x_train_std, y_train)

print('Train:')
model.evaluate(x_train_std, y_train)

print('Test:')
model.evaluate(x_test_std, y_test)
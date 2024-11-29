import ast

import pandas as pd

domains = ['beauty', 'cd', 'movie', 'music', 'sport']
for domain in domains:
    file_prefix = './dataset/amazon_' + domain
    df = pd.read_csv(file_prefix + '/user_seq.csv')
    train_users, train_items, train_ratings, train_times = [], [], [], []
    valid_users, valid_items, valid_ratings, valid_times = [], [], [], []
    test_users, test_items, test_ratings, test_times = [], [], [], []

    for i in range(len(df)):
        user = df['reviewerID'][i] - 1
        item_seq = ast.literal_eval(df['asin'][i])
        item_seq = [item - 2 for item in item_seq]
        rate_seq = ast.literal_eval(df['overall'][i])
        rate_seq = [rate / 5.0 for rate in rate_seq]
        time_seq = ast.literal_eval(df['unixReviewTime'][i])

        train_users.extend([user for _ in range(len(item_seq) - 2)])
        valid_users.append(user)
        test_users.append(user)

        train_items.extend(item_seq[:-2])
        valid_items.append(item_seq[-2])
        test_items.append(item_seq[-1])

        train_ratings.extend(rate_seq[:-2])
        valid_ratings.append(rate_seq[-2])
        test_ratings.append(rate_seq[-1])

        train_times.extend(time_seq[:-2])
        valid_times.append(time_seq[-2])
        test_times.append(time_seq[-1])

    print(max(train_users + valid_users + test_users), min(train_users + valid_users + test_users))
    print(max(train_items + valid_items + test_items), min(train_items + valid_items + test_items))

    new_df = pd.DataFrame()
    new_df['user'] = train_users
    new_df['item'] = train_items
    new_df['rating'] = train_ratings
    new_df['time'] = train_times
    new_df.to_csv(file_prefix + '/amazon_' + domain + '.train', header=True, index=False, sep='\t')

    new_df = pd.DataFrame()
    new_df['user'] = valid_users
    new_df['item'] = valid_items
    new_df['rating'] = valid_ratings
    new_df['time'] = valid_times
    new_df.to_csv(file_prefix + '/amazon_' + domain + '.valid', header=True, index=False, sep='\t')

    new_df = pd.DataFrame()
    new_df['user'] = test_users
    new_df['item'] = test_items
    new_df['rating'] = test_ratings
    new_df['time'] = test_times
    new_df['last_item'] = valid_items
    new_df.to_csv(file_prefix + '/amazon_' + domain + '.test', header=True, index=False, sep='\t')
    pass
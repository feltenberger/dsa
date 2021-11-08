import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import metrics, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, STATUS_OK, Trials, STATUS_FAIL  # , hp
from tensorflow.keras.layers import Dense, Dropout, Embedding, Concatenate, Flatten, BatchNormalization


pd.options.display.max_rows = None
pd.options.display.max_columns = None

_EVALS = 5
_EPOCHS = 50
_SPLIT = 0.75
_BATCH_SIZE = 2000
_OPTIMIZING = False
_BATCH_NORM = False
_EARLY_STOPPING = False
_PLOT_HISTORY = (not _OPTIMIZING) and True
_MIN_COS_DATE = datetime(1985, 1, 1)
_CATEGORICALS = ['StateHoliday', 'DayOfWeek', 'StoreType', 'Assortment']
_DROP_COLUMNS = ['Customers', 'Date', 'PromoInterval', 'CompetitionOpenSince', 'Promo2SinceWeek', 'Promo2SinceYear', 'Promo2Since']
# despite best attmepts this feature still does more harm than good
_DROP_COLUMNS.append('CompetitionOpenSinceRelative')

default_params = {'dropout': 0.4404292830003049,
                  'layers': 6,
                  'optimizer': 'rmsprop',
                  'store_embedding': 8,
                  'width_0': 128,
                  'width_1': 128,
                  'width_2': 160,
                  'width_3': 160,
                  'width_4': 128,
                  'width_5': 128,  # was tuned to 160 but suspect that was error. it's a wash in testing
                  'lr': 0.0002,  # 0.001 seems to produce best val_mae on 30 epochs but is unstable. 0.0001 is slow but stable; going for 0.0002 with 50 epochs
                  }


space = {'lr': 0.0001,          # hp.choice('lr', [0.1, 0.01, 0.001, 0.0001]),
         'dropout': 0.44,       # hp.choice('dropout', [0.44, 0.44]),
         'layers': 6,           # hp.choice('layers', [6]),
         'width_0': 128,        # hp.choice('width_0', [80, 128, 160]),
         'width_1': 128,        # hp.choice('width_1', [80, 128, 160]),
         'width_2': 160,        # hp.choice('width_2', [80, 128, 160]),
         'width_3': 160,        # hp.choice('width_3', [80, 128, 160]),
         'width_4': 128,        # hp.choice('width_4', [80, 128, 160]),
         'width_5': 128,        # hp.choice('width_5', [80, 128, 160]),
         'store_embedding': 8,  # hp.choice('store_embedding', [2, 4, 8, 16, 32]),
         'optimizer': 'rmsprop'}  # TODO: this is currently ignored


def _clean_competition_open_since(df):
    df['day'] = 1
    df = df.rename(index=str, columns={'CompetitionOpenSinceYear': 'year', 'CompetitionOpenSinceMonth': 'month'})
    df['CompetitionOpenSince'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.drop(['day', 'month', 'year'], axis='columns')

    df['CompetitionOpenSince'] = df['CompetitionOpenSince'].apply(lambda x: x if x > _MIN_COS_DATE or pd.isnull(x) else _MIN_COS_DATE)

    return df


def _read_clean_and_merge_data():
    df = pd.read_csv('data/train.csv', low_memory=False)

    store_df = pd.read_csv('data/store.csv')
    store_df = _clean_competition_open_since(store_df)

    # double merge trick to preserve order per https://bit.ly/2XTMa4b
    df = df.merge(df.merge(right=store_df, on='Store', sort=False))

    return df


def _split_data(df):
    split_point = int(_SPLIT * len(df))

    train_df = df.iloc[0:split_point]
    test_df = df.iloc[split_point:]

    return train_df, test_df


def _get_data():
    df = _read_clean_and_merge_data()

    # reverse order, since dataframe is reversed
    df = df.reindex(index=df.index[::-1])
    df = pd.get_dummies(df, columns=_CATEGORICALS)

    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    # TODO: this is hacky - we're doing it here because we can't do it when processing competitionopensince (which is performed on loading stores.csv, before merge)
    df = _encode_relative_date(df, 'CompetitionOpenSince')

    y_df = df['Sales']
    x_df = df
    del x_df['Sales']

    return _split_data(x_df), _split_data(y_df)


# note that git history includes a version which calculates based on a daily mean
def _print_baseline_mae(y_train, y_test):
    baseline_mae = np.mean(np.abs(y_test - np.mean(y_train)))
    print('baseline mae: {}'.format(baseline_mae))


def _plot_history(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def _extract_categoricals_for_embedding(x_df):
    stores_s = x_df['Store']
    del x_df['Store']

    return {'store': stores_s, 'other': x_df}


# TODO: clean this up
#       - extract plotting of history
#       - extract creation of dense layers and/or model
#
def _fit_and_predict(params, x_train, y_train_df, x_test, y_test_df):
    inputs = []

    input_store = Input(shape=[1])
    inputs.append(input_store)
    _STORE_DIM = params['store_embedding']
    assert(len(x_train['store'].unique()) == len(x_test['store'].unique()))  # nosec
    store_embedding = Embedding(len(x_train['store'].unique()) + 1, _STORE_DIM, input_length=1)(input_store)
    store_embedding = Flatten()(store_embedding)

    input_shape = x_train['other'].shape[1]
    input_tensor = Input(shape=(input_shape,))
    inputs.append(input_tensor)

    x = Concatenate()([store_embedding, input_tensor])

    for layer_index in range(params['layers']):
        width = params['width_' + str(layer_index)]
        # TODO: adding this messy if as saw strange things when setting use_bias=not _BATCH_NORM (maybe spurious but unknown)
        import pdb; pdb.set_trace()
        if _BATCH_NORM:
            # TODO: 90% sure this should be false but may want to google batchnormalization and use_bias to be sure
            x = Dense(width, input_shape=(input_shape,), activation='relu', use_bias=False)(x)
        else:
            x = Dense(width, input_shape=(input_shape,), activation='relu')(x)
        # trying swapping order (doesn't seem to make much diff)
        x = Dropout(params['dropout'])(x)
        if _BATCH_NORM:
            x = BatchNormalization()(x)
        input_shape = width

    x = Dense(1, input_shape=(input_shape,))(x)

    model = Model(inputs, x)

    optimizer = RMSprop(lr=params['lr'])  # params['optimizer']
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[metrics.MAE])
    print(model.summary())

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=16, min_delta=20)
    callbacks = [es] if _EARLY_STOPPING else []

    # import pdb; pdb.set_trace()
    history = model.fit([x_train['store'].values, x_train['other'].values], y_train_df.values, batch_size=_BATCH_SIZE, epochs=_EPOCHS, verbose=1,
                        validation_data=([x_test['store'].values, x_test['other'].values], y_test_df.values), callbacks=callbacks)
    score = model.evaluate([x_test['store'].values, x_test['other'].values], y_test_df.values, batch_size=_BATCH_SIZE, verbose=1)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    if _PLOT_HISTORY:
        _plot_history(history, 'loss')
        _plot_history(history, 'mean_absolute_error')

    return {'loss': score[0], 'status': STATUS_OK if not np.isnan(score[0]) else STATUS_FAIL, 'mae': score[1]}


def optimize(x_train, y_train_df, x_test, y_test_df):
    def objective(params):
        return _fit_and_predict(params, x_train, y_train_df, x_test, y_test_df)

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=_EVALS)

    print('best: ')
    print(best)
    print('best_trial: ')
    print(trials.best_trial)


def evaluate(x_train, y_train_df, x_test, y_test_df):
    results = _fit_and_predict(default_params, x_train, y_train_df, x_test, y_test_df)

    print('val_los: {}'.format(results['loss']))
    print('val_mae: {}'.format(results['mae']))


def _convert_date_to_relative_days(df, column):
    df[column] = df['Date'] - df[column]
    df[column] = df[column].apply(lambda x: x.days)

    return df


def _transform_competition_open_since(x_train_df, x_test_df):
    x_train_df = _convert_date_to_relative_days(x_train_df, 'CompetitionOpenSince')
    x_test_df = _convert_date_to_relative_days(x_test_df, 'CompetitionOpenSince')

    min_train_cos = np.min(x_train_df['CompetitionOpenSince'])
    # TODO: hoist into a function called on train and test separately (probably including _convert_date call, above)
    x_train_df['CompetitionOpenSince'] = x_train_df['CompetitionOpenSince'] - min_train_cos
    x_test_df['CompetitionOpenSince'] = x_test_df['CompetitionOpenSince'] - min_train_cos

    x_train_df['CompetitionOpenSince'] = np.log(1 + x_train_df['CompetitionOpenSince'])
    x_test_df['CompetitionOpenSince'] = np.log(1 + x_test_df['CompetitionOpenSince'])

    mean_train_cos = x_train_df['CompetitionOpenSince'].mean()
    std_train_cos = x_train_df['CompetitionOpenSince'].std()

    x_train_df['CompetitionOpenSince'] = (x_train_df['CompetitionOpenSince'] - mean_train_cos) / std_train_cos
    x_test_df['CompetitionOpenSince'] = (x_test_df['CompetitionOpenSince'] - mean_train_cos) / std_train_cos

    x_train_df['CosIsNa'] = x_train_df['CompetitionOpenSince'].isna()
    x_test_df['CosIsNa'] = x_test_df['CompetitionOpenSince'].isna()

    stdized_mean_train_cos = x_train_df['CompetitionOpenSince'].mean()

    # TODO: this isn't ideal. see note in evernote on better solutions
    x_train_df['CompetitionOpenSince'] = x_train_df['CompetitionOpenSince'].fillna(stdized_mean_train_cos)
    # TODO: this is a particularly troublesome line, since it causes a huge spike at the mean value of a different distribution!
    x_test_df['CompetitionOpenSince'] = x_test_df['CompetitionOpenSince'].fillna(stdized_mean_train_cos)

    return x_train_df, x_test_df


# TODO: HACK: this leaks data. clean it up to lose the relative name, split train and test and use _convert_date_to_relative_days
def _encode_relative_date(df, column):
    relative_column = column + 'Relative'
    df[relative_column] = df['Date'] - df[column]
    df[relative_column] = df[relative_column].apply(lambda x: x.days)
    df[relative_column] = df[relative_column].fillna(df[relative_column].median())
    df[relative_column] = (df[relative_column] - df[relative_column].mean()) / df[relative_column].std()

    return df


# TODO: do we really want to set NaT to mean - it really screws the distribution (ask Patrick)
def _transform_promo2_since(df):
    # https://stackoverflow.com/questions/45436873/pandas-how-to-create-a-datetime-object-from-week-and-year
    df['Promo2Since'] = pd.to_datetime(df.Promo2SinceYear.map('{:.0f}'.format).astype(str), format='%Y') + \
        pd.to_timedelta(df.Promo2SinceWeek.fillna(0).mul(7).map('{:.0f}'.format).astype(str) + ' days')
    df = _encode_relative_date(df, 'Promo2Since')

    return df


# TODO: HACK: this leaks data. need to split out train and test and use e.g. just train mean, std, etc.
def _transform_competition_distance(df):
    # df['CompetitionDistance'] = df['CompetitionDistance'].apply(lambda x: max(-1000, min(x, 2000)))
    df['CompetitionDistance'] = np.log(df['CompetitionDistance'])

    mean_train_cd = df['CompetitionDistance'].mean()
    df['CompetitionDistance'] = (df['CompetitionDistance'] - mean_train_cd) / df['CompetitionDistance'].std()

    df['CdIsNa'] = df['CompetitionDistance'].isna()

    stdized_mean_train_cd = df['CompetitionDistance'].mean()
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(stdized_mean_train_cd)

    return df


def _transform_data(x_train_df, x_test_df):
    x_train_df, x_test_df = _transform_competition_open_since(x_train_df, x_test_df)

    x_train_df = _transform_promo2_since(x_train_df)
    x_test_df = _transform_promo2_since(x_test_df)

    x_train_df = _transform_competition_distance(x_train_df)
    x_test_df = _transform_competition_distance(x_test_df)

    x_train_df = x_train_df.drop(_DROP_COLUMNS, axis='columns')
    x_test_df = x_test_df.drop(_DROP_COLUMNS, axis='columns')

    return x_train_df, x_test_df


def main():
    # TODO: this is a little idiosyncratic in unpacking and order
    (x_train_df, x_test_df), (y_train_df, y_test_df) = _get_data()
    x_train_df, x_test_df = _transform_data(x_train_df, x_test_df)

    x_train = _extract_categoricals_for_embedding(x_train_df)
    x_test = _extract_categoricals_for_embedding(x_test_df)

    _print_baseline_mae(y_train_df.values, y_test_df.values)

    if _OPTIMIZING:
        optimize(x_train, y_train_df, x_test, y_test_df)
    else:
        evaluate(x_train, y_train_df, x_test, y_test_df)


if __name__ == '__main__':
    main()

from sklearn.metrics import roc_auc_score, accuracy_score

from pyspikelib import TrainNormalizeTransform
from pyspikelib import TsfreshVectorizeTransform
from pyspikelib import TsfreshFeaturePreprocessorPipeline
from pyspikelib.utils import simple_undersampling


def tsfresh_fit_predict(model, X_train, X_test, y_train, y_test, config):
    X_train, y_train = tsfresh_vectorize(X_train, y_train, config)
    X_test, y_test = tsfresh_vectorize(X_test, y_test, config)
    preprocessing = TsfreshFeaturePreprocessorPipeline(
        do_scaling=config.scale, remove_low_variance=config.remove_low_variance
    ).construct_pipeline()
    preprocessing.fit(X_train)
    X_train = preprocessing.transform(X_train)
    X_test = preprocessing.transform(X_test)
    print('Dataset size: train {}, test {}'.format(X_train.shape, X_test.shape))
    for _ in range(config.trials):
        eval_classifier_scores(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            train_subsample_factor=config.train_subsample_factor,
            test_subsample_factor=config.test_subsample_factor,
        )
    feature_names = [
        'abs_energy',
        'mean',
        'median',
        'minimum',
        'maximum',
        'standard_deviation',
    ]
    simple_baseline_features = ['value__' + name for name in feature_names]
    X_train = X_train.loc[:, simple_baseline_features]
    X_test = X_test.loc[:, simple_baseline_features]
    print('SIMPLE BASELINE:')
    for _ in range(config.trials):
        eval_classifier_scores(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            train_subsample_factor=config.train_subsample_factor,
            test_subsample_factor=config.test_subsample_factor,
        )


def tsfresh_vectorize(X, y, config):
    normalizer = TrainNormalizeTransform(window=config.window, step=config.step)
    vectorizer = TsfreshVectorizeTransform(feature_set='no_entropy')
    X, y = normalizer.transform(X, y, delimiter=config.delimiter)
    X = vectorizer.transform(X)
    return X, y


def eval_classifier_scores(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    train_subsample_factor=0.7,
    test_subsample_factor=0.7,
):
    X_train, y_train = simple_undersampling(
        X_train, y_train, subsample_size=train_subsample_factor
    )
    X_test, y_test = simple_undersampling(
        X_test, y_test, subsample_size=test_subsample_factor
    )
    model.fit(X_train, y_train)
    acc_score = accuracy_score(y_test, model.predict(X_test))
    auc_roc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print('Accuracy & AUC-ROC scores of RF: {}, {}'.format(acc_score, auc_roc_score))
    return {'accuracy': acc_score, 'auc-roc': auc_roc_score}

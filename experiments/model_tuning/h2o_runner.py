import os.path as osp
from pathlib import Path

import h2o
import pandas as pd
from h2o.automl import H2OAutoML


class H2ORunner:
    def __init__(self, nthreads=-1, max_mem_size=12, logdir=None):
        h2o.init(nthreads=nthreads, max_mem_size=max_mem_size)
        self.h2o_frames = {}
        self.logdir = './' if logdir is None else logdir
        Path(self.logdir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def create_h2o_frame(X, y):
        frame = pd.concat(
            [X.reset_index(drop=True), pd.Series(y, name='target')], axis='columns'
        )
        frame.columns = [str(name) for name in frame.columns]
        X_y_h = h2o.H2OFrame(frame)
        X_y_h['target'] = X_y_h['target'].asfactor()
        return X_y_h

    def set_data(self, train_data, valid_data, test_data):
        self.h2o_frames['train'] = self.create_h2o_frame(*train_data)
        self.h2o_frames['valid'] = self.create_h2o_frame(*valid_data)
        self.h2o_frames['test'] = self.create_h2o_frame(*test_data)

    def run(self, max_runtime_secs=10000, max_models=None):
        self.aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs, max_models=max_models,
            nfolds=0, seed=42
        )
        self.aml.train(
            y='target',
            training_frame=self.h2o_frames['train'],
            validation_frame=self.h2o_frames['valid'],
            leaderboard_frame=self.h2o_frames['test'],
        )

        self.leaderboard = self.aml.leaderboard.as_data_frame()
        self.dump_results()

    def dump_results(self):
        model_ids = list(self.aml.leaderboard['model_id'].as_data_frame().iloc[:, 0])

        for m_id in model_ids:
            mdl = h2o.get_model(m_id)
            h2o.save_model(model=mdl, path=self.logdir, force=True)

        h2o.export_file(
            self.aml.leaderboard,
            osp.join(self.logdir, 'aml_leaderboard.h2o'),
            force=True,
        )

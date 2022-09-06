from glob import glob

import chika
import pandas as pd
import pylab as plt
import seaborn as sns
from spikebench.plotting import beautify_mpl, prettify

beautify_mpl()

@chika.config
class Config:
    metric_name: str = 'cohen_kappa'
    export_format: str = 'png'
    csv_pattern = 'retina*imbalanced'
    include_models = ('raw_knn_k1_isi', 'raw_knn_k1_victor_purpura')
    xticks = ('1NN (ISI)', '1NN (Victor-Purpura)')
    ylabel = 'Geometric mean on the imbalanced test set'
    title = ''
    outfile = 'retina_imbalanced.png'


@chika.main(cfg_cls=Config)
def main(cfg: Config):
    df = pd.DataFrame()
    for csv_file_path in glob(f'./csv/{cfg.csv_pattern}.csv'):
        df = pd.concat([df, pd.read_csv(csv_file_path)])

    df = df[df.model_name.isin(cfg.include_models)]

    grouped = df.loc[:,['model_name', cfg.metric_name]] \
        .groupby(['model_name']) \
        .median() \
        .sort_values(by=cfg.metric_name)

    ax = sns.boxplot(x='model_name', y=cfg.metric_name,
        data=df, order=grouped.index)

    for box in ax.artists:
        box.set_facecolor((255/255, 229/255, 217/255, 0.7))

    plt.xticks(range(len(cfg.xticks)), cfg.xticks, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('')
    plt.ylabel(cfg.ylabel, fontsize=18)
    plt.title(cfg.title, fontsize=20)

    prettify()
    plt.savefig(f'./figures/{cfg.outfile}',
        bbox_inches='tight', transparent=False)


if __name__ == '__main__':
    main()

import numpy as np
import pandas

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn

results = pandas.read_csv('cross_validation_results.csv')
results['train_size'] = results['train_size'].astype(int)
results['cv_name'] = results['cv_name'].map({
                            'loo': 'LOO',
                            '50 splits': '50 splits\n20% test'})
# Limit to results with 100 samples
results = results[results['train_size'] == 100]

plt.close('all')
#plt.figure(figsize=(3.4, 3.5))
plt.rcParams['ytick.major.pad'] = 2.
plt.rcParams['ytick.labelsize'] = 10.
plt.rcParams['xtick.major.pad'] = 2.
plt.rcParams['xtick.labelsize'] = 10.


seaborn.set_style("ticks",
                  {"xtick.color": '0',
                   "ytick.color": '0',
                   "text.color": '0',
                   "grid.color": '.8',
                   "axes.edgecolor": '0'})

seaborn.set_context(rc={"lines.linewidth": 2})
box = seaborn.pairplot(data=results, x_vars=['score_error'],
                      y_vars=['score_sem'],
                      hue='cv_name',
                      size=4,
                      plot_kws=dict(s=10, edgecolor='none', alpha=.4),
                      #hue='cv_name',
                      #palette=[(1, 1, .2), (.8, .8, 0)],
                      )

seaborn.despine(top=False, bottom=False, left=False, right=False)
plt.axvline(0, color='.7', lw=1, zorder=0)

ax = plt.gca()

plt.xlim(-.19, .19)
plt.ylim(0, .055)

box.fig.legends[0].remove()

plt.legend(loc=(-.01, .45), handlelength=1, markerscale=2,
           handletextpad=.2, labelspacing=3, fontsize=11)


def formatter(value, pos):
    sign = " "
    if value < 0:
        sign = "-"
    elif value > 0:
        sign = "+"
    return "%s%i%%" % (sign, np.round(abs(100 * value)))

ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))


def time_100_formatter(value, pos):
    return "%.0f" % (np.round(abs(100 * value)))

ax.yaxis.set_major_formatter(plt.FuncFormatter(time_100_formatter))

plt.tight_layout(rect=(-.03, -.03, 1.03, 1.03))


plt.xlabel('Estimation error on the prediction accuracy',
           size=12.7)
plt.ylabel('Standard error of the mean across the folds  ', size=12.7)

plt.savefig('error_vs_sem.pdf')
plt.show()


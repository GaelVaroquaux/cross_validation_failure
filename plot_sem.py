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

# Compute confidence intervals with the SEM:
#results['sem_error_bar'] = 1.96 * results['score_sem']
results['sem_error_bar'] = 1.64 * results['score_sem']

# Now compute the empirical 95% confidence interval
results['empirical_error_bar'] = results.groupby(
                ['cv_name', 'train_size']
            )['score_error'].transform(lambda x: x.quantile(.95))

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
box = seaborn.pairplot(data=results, x_vars=['empirical_error_bar'],
                      y_vars=['sem_error_bar'],
                      hue='cv_name',
                      size=3,
                      aspect=1.05,
                      plot_kws=dict(s=7, edgecolor='none', alpha=.3),
                      )

seaborn.despine(top=True, bottom=False, left=False, right=True)
#plt.axvline(0, color='.7', lw=1, zorder=0)

ax = plt.gca()

plt.xlim(0, .2)
plt.ylim(0, .2)

box.fig.legends[0].remove()

mean_results = results.groupby(['cv_name', 'train_size']).mean().reset_index()
mean_results = mean_results.sort_values('cv_name', ascending=False)

handles = list()
for cv_name, df in mean_results.groupby('cv_name', sort=False):
    handles.extend(plt.plot(df['empirical_error_bar'], df['sem_error_bar'],
                            label=cv_name))

plt.rcParams['font.size'] = 10

plt.legend(handles, results.cv_name.unique(),
           loc=(.01, .45), handlelength=.8, markerscale=2,
           handletextpad=.2, labelspacing=.8, fontsize=10,
           title='Cross-validation\nstrategy')

plt.text(.22, .98, "SEM over-estimates error bars",
         transform=ax.transAxes, size=8)
plt.text(.99, .9, "SEM under-estimates error bars",
         transform=ax.transAxes, size=8, rotation=90)
plt.plot([0, .2], [0, .2], color='.5', lw=1)


def formatter(value, pos):
    sign = " "
    if value < 0:
        sign = "-"
    elif value > 0:
        sign = "+"
    return "%s%i%%" % (sign, np.round(abs(100 * value)))

ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))
ax.yaxis.set_major_formatter(plt.FuncFormatter(formatter))

plt.tight_layout(rect=(-.03, -.03, 1.04, 1.04))


plt.xlabel('95 percentile on the observed error          ',
           size=12.7)
plt.ylabel('95% confidence limit from SEM ', size=12.7)

plt.savefig('sem_vs_error.pdf')
plt.show()


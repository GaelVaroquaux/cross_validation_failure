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

plt.close('all')
plt.figure(figsize=(3.4, 4))
plt.rcParams['ytick.major.pad'] = 2.
plt.rcParams['ytick.labelsize'] = 11.


seaborn.set_style("whitegrid",
                  {"xtick.color": '0',
                   "ytick.color": '0',
                   "text.color": '0',
                   "grid.color": '.7',
                   "axes.edgecolor": '.7'})

seaborn.set_context(rc={"lines.linewidth": 2})
box = seaborn.boxplot(data=results, x='score_error', y='train_size',
                      orient='h', hue='cv_name',
                      whis=[5, 95], width=.45, fliersize=0,
                      palette=[(1, 1, .2), (.8, .8, 0)],
                      )


seaborn.set_context(rc={"lines.linewidth": .5})
voilin = seaborn.violinplot(data=results, x='score_error', y='train_size',
                            orient='h', hue='cv_name', split=True,
                            inner=None, legend=False,
                            palette=[(1, 1, .2), (.8, .8, 0)],
                            )

seaborn.despine(top=True, bottom=True, left=True, right=True)

plt.axhspan(.5, 1.5, facecolor='.9', edgecolor='none', zorder=-1)
plt.axhspan(2.5, 3.5, facecolor='.9', edgecolor='none', zorder=-1)
plt.axvline(0, color='.7', lw=3, zorder=0)

ax = plt.gca()
plt.xlim(-.33, .33)
plt.ylim(3.5, -.5)

ax.add_artist(mpatches.Rectangle((.052, 1 / 3. + .03), .4, .13,
                                 facecolor='1', edgecolor='none'))
ax.add_artist(mpatches.Rectangle((.1515, -1 / 3. - .14), .2, .13,
                                 facecolor='1', edgecolor='none'))
ax.add_artist(mpatches.Rectangle((.035, 3.9 / 3. + .06), .4, .13,
                                 facecolor='.9', edgecolor='none'))
ax.add_artist(mpatches.Rectangle((.1515, 1.95 / 3. - .12), .2, .13,
                                 facecolor='.9', edgecolor='none'))
ax.add_artist(mpatches.Rectangle((.035, 6.9 / 3. + .06), .4, .13,
                                 facecolor='1', edgecolor='none'))
ax.add_artist(mpatches.Rectangle((.1515, 4.95 / 3. - .12), .2, .13,
                                 facecolor='1', edgecolor='none'))
ax.add_artist(mpatches.Rectangle((.035, 9.9 / 3. + .06), .4, .13,
                                 facecolor='.9', edgecolor='none'))
ax.add_artist(mpatches.Rectangle((.1515, 7.95 / 3. - .12), .2, .13,
                                 facecolor='.9', edgecolor='none'))


legends = list()
for i in range(4):
    l = plt.legend(handles=[mpatches.Patch(facecolor=(1, 1, .2), label='LOO'),
                        mpatches.Patch(facecolor=(.8, .8, 0),
                                    label='50 splits, 20% test')],
            loc=(.51, i / 3.99 - .0135), handlelength=1,
            handletextpad=.4, labelspacing=5.2, fontsize=8.5,
            markerfirst=False)
    legends.append(l)

for l in legends:
    ax.add_artist(l)



def formatter(value, pos):
    sign = " "
    if value < 0:
        sign = "-"
    elif value > 0:
        sign = "+"
    return "%s%i%%" % (sign, np.round(abs(100 * value)))

ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))

plt.xticks([-.3, -.15, 0, .15, .3])


def float_formatter(value, pos):
    sign = " "
    if value < 0:
        sign = "-"
    elif value > 0:
        sign = "+"
    return "%s%i%%" % (sign, round(abs(100 * value)))


# Add text for the 5% and 95%
for (train_size, cv_name), values in results.groupby(('train_size', 'cv_name')):
    values = values.score_error
    shift = .2 + ([30, 100, 300, 1000].index(train_size)
             - .3 * (cv_name == 'LOO'))
    lower_quantile = values.quantile(.05)
    plt.text(lower_quantile - .01, shift,
            float_formatter(lower_quantile, 0),
            size=10, ha='right')
    top_quantile = values.quantile(.95)
    plt.text(top_quantile + .01, shift,
            float_formatter(top_quantile, 0),
            size=10)


plt.tight_layout(rect=(.02, -.04, 1.03, 1.04))


plt.xlabel('Estimation error on the prediction accuracy             ',
           size=12.7)
plt.ylabel('Number of available samples   ', size=12.7)

plt.savefig('error_vs_train_size.pdf')
plt.show()


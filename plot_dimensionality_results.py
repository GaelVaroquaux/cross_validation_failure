import numpy as np
import pandas

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn

results = pandas.read_csv('dimensionality_results.csv')
results['train_size'] = results['train_size'].astype(int)
results['dim'] = results['dim'].astype(int)
results['cv_name'] = results['cv_name'].map({
                            'loo': 'LOO',
                            '50 splits': '50 splits\n20% left out'})

plt.close('all')
plt.figure(figsize=(4, 5.33))

seaborn.set_style("whitegrid",
                  {"xtick.color": '0',
                   "ytick.color": '0',
                   "text.color": '0',
                   "grid.color": '.7',
                   "axes.edgecolor": '.7'})

seaborn.set_context(rc={"lines.linewidth": 2})
box = seaborn.boxplot(data=results, x='score_error', y='dim',
                      orient='h', hue='cv_name',
                      whis=[5, 95], width=.45, fliersize=0,
                      palette=[(1, 1, .2), (.8, .8, 0)],
                      )
ax = plt.gca()
for a in ax.artists:
    a.remove()
    a.set_facecolor('none')
    a.set_edgecolor('none')



seaborn.set_context(rc={"lines.linewidth": .5})
voilin = seaborn.violinplot(data=results, x='score_error', y='dim',
                            orient='h', hue='cv_name', split=True,
                            inner=None, legend=False,
                            palette=[(1, 1, .2), (.8, .8, 0)],
                            )

seaborn.despine(top=True, bottom=True, left=True, right=True)

plt.axhspan(.5, 1.5, facecolor='.9', edgecolor='none', zorder=-1)
plt.axhspan(2.5, 3.5, facecolor='.9', edgecolor='none', zorder=-1)
plt.axvline(0, color='.7', lw=3, zorder=0)

ax = plt.gca()
plt.xlim(-.199, .199)

ax.add_artist(mpatches.Rectangle((.035, 1 / 3. - .018), .4, .13,
                                 facecolor='1', edgecolor='none'))
ax.add_artist(mpatches.Rectangle((.1315, -1 / 3. - .11), .2, .13,
                                 facecolor='1', edgecolor='none'))
ax.add_artist(mpatches.Rectangle((.035, 3.9 / 3. + .008), .4, .13,
                                 facecolor='.9', edgecolor='none'))
ax.add_artist(mpatches.Rectangle((.1315, 1.95 / 3. - .1), .2, .13,
                                 facecolor='.9', edgecolor='none'))
ax.add_artist(mpatches.Rectangle((.035, 6.9 / 3. + .008), .4, .13,
                                 facecolor='1', edgecolor='none'))
ax.add_artist(mpatches.Rectangle((.1315, 4.95 / 3. - .1), .2, .13,
                                 facecolor='1', edgecolor='none'))


legends = list()
for i in range(3):
    l = plt.legend(handles=[mpatches.Patch(facecolor=(1, 1, .2), label='LOO'),
                        mpatches.Patch(facecolor=(.8, .8, 0),
                                    label='50 splits, 20% left out')],
            loc=(.58, i / 3. + .005), handlelength=1,
            handletextpad=.4, labelspacing=5.9, fontsize=8.5,
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


def float_formatter(value, pos):
    sign = " "
    if value < 0:
        sign = "-"
    elif value > 0:
        sign = "+"
    return "%s%i%%" % (sign, round(abs(100 * value)))


# Add text for the 5% and 95%
for (dim, cv_name), values in results.groupby(('dim', 'cv_name')):
    values = values.score_error
    shift = .2 + ([1, 10, 300, 10000].index(dim)
             - .3 * (cv_name == 'LOO'))
    lower_quantile = values.quantile(.05)
    plt.text(lower_quantile - .01, shift,
            float_formatter(lower_quantile, 0),
            size=10, ha='right')
    top_quantile = values.quantile(.95)
    plt.text(top_quantile + .01, shift,
            float_formatter(top_quantile, 0),
            size=10)


plt.tight_layout(rect=(-.02, 0, 1.01, 1))


plt.xlabel('Error on the estimation of prediction accuracy        ',
           size=12.7)
plt.ylabel('Number of features', size=12.7)

plt.savefig('dimensionality_results.pdf')
plt.show()


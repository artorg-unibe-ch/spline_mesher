import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# https://matplotlib.org/3.1.3/tutorials/introductory/customizing.html


# rcParams
f = 1
plt.rcParams['figure.figsize'] = [10 * f, 10 * f]
plt.rcParams['font.size'] = 15
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# sns styles
sns.set(font_scale=1.5)
sns.set_style('white')
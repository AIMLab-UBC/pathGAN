import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from statannot import add_stat_annotation


def gan_eval_boxplots(acc, kappa, auc, a_acc):
    print('--------- p-value --------------')
    print(r'\begin{longtable}{cccc} \toprule')
    print(r'{Splits} & {W. Acc.} & {Kappa} & {AUC} & {A. Acc.}  \\ \midrule')
    print('Split vs. Split + Fake  & {:.4f} & {:.4f}& {:.4f} & {:.4f}  \\\\'.format(scipy.stats.ttest_rel(
        acc[:10], acc[-10:]).pvalue, scipy.stats.ttest_rel(kappa[:10], kappa[-10:]).pvalue, scipy.stats.ttest_rel(auc[:10], auc[-10:]).pvalue, scipy.stats.ttest_rel(a_acc[:10], a_acc[-10:]).pvalue))
    print('Split vs. Split + Real  & {:.4f} & {:.4f}& {:.4f} & {:.4f}  \\\\'.format(scipy.stats.ttest_rel(
        acc[:10], acc[10:20]).pvalue, scipy.stats.ttest_rel(kappa[:10], kappa[10:20]).pvalue, scipy.stats.ttest_rel(auc[:10], auc[10:20]).pvalue, scipy.stats.ttest_rel(a_acc[:10], a_acc[10:20]).pvalue))
    print('Split + Real vs. Split + Fake  & {:.4f} & {:.4f} & {:.4f} & {:.4f}  \\\\'.format(scipy.stats.ttest_rel(
        acc[10:20], acc[-10:]).pvalue, scipy.stats.ttest_rel(kappa[10:20], kappa[-10:]).pvalue, scipy.stats.ttest_rel(auc[10:20], auc[-10:]).pvalue, scipy.stats.ttest_rel(a_acc[10:20], a_acc[-10:]).pvalue))
    print(r'\bottomrule')

    baseline_acc = np.asarray(acc[:10])
    baseline_kappa = np.asarray(kappa[:10])
    baseline_auc = np.asarray(auc[:10])
    baseline_a_acc = np.asarray(a_acc[:10])
    fake_acc = np.asarray(acc[-10:])
    fake_kappa = np.asarray(kappa[-10:])
    fake_auc = np.asarray(auc[-10:])
    fake_a_acc = np.asarray(a_acc[-10:])
    real_acc = np.asarray(acc[10:20])
    real_kappa = np.asarray(kappa[10:20])
    real_auc = np.asarray(auc[10:20])
    real_a_acc = np.asarray(a_acc[10:20])

    def _printer(acc, kappa, auc, a_acc, prefix):
        print('{} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\'.format(
            prefix, acc.mean(), acc.std(), np.median(acc), kappa.mean(), kappa.std(), np.median(kappa), auc.mean(), auc.std(), np.median(auc), a_acc.mean(), a_acc.std(), np.median(a_acc)))

    print('-------- mean and std ----------')
    print(r'\begin{longtable}{ccccc} \toprule')
    print(r'{Splits} & {W. Acc. Mean} & {W. Acc. STD} & {W. Acc. Median} & {Kappa Mean}  & {Kappa STD} & {Kappa Median} & {AUC Mean} & {AUC STD} & {AUC Median} & {A. Acc. Mean} & {A. Acc. STD} & {A. Acc. Median} \\ \midrule')
    _printer(baseline_acc, baseline_kappa,
             baseline_auc, baseline_a_acc, 'Splits')
    _printer(fake_acc, fake_kappa, fake_auc, fake_a_acc, 'Splits + Synthetic')
    _printer(real_acc, real_kappa, real_auc, real_a_acc, 'Splits + Real')
    print(r'\bottomrule')


def gan_eval_analyze_table(latex_table):
    splits = latex_table.split(r'\midrule')
    acc = np.zeros(30).tolist()
    kappa = np.zeros(30).tolist()
    auc = np.zeros(30).tolist()
    a_acc = np.zeros(30).tolist()
    for split_idx in range(10):
        cur_split = splits[split_idx].split('Split {}'.format(split_idx))
        cur_split = [s.strip() for s in cur_split]
        cur_split = [s for s in cur_split if s != '']
        for combination_idx in range(3):
            result = cur_split[combination_idx].split('&')
            result = [r.strip() for r in result]
            result = [r for r in result if r != '']
            cur_acc = result[-4].replace(r'\%', '')
            cur_kappa = result[-3]
            cur_auc = result[-2]
            cur_a_acc = result[-1].replace(r'\%', '').replace('\\', '')
            cur_acc = cur_acc.replace(r'extbf{', '').replace(r'}', '')
            cur_kappa = cur_kappa.replace(r'extbf{', '').replace(r'}', '')
            cur_auc = cur_auc.replace(r'extbf{', '').replace(r'}', '')
            cur_a_acc = cur_a_acc.replace(r'extbf{', '').replace(r'}', '')
            acc[10 * combination_idx + split_idx] = float(cur_acc) / 100
            kappa[10 * combination_idx + split_idx] = float(cur_kappa)
            auc[10 * combination_idx + split_idx] = float(cur_auc)
            a_acc[10 * combination_idx + split_idx] = float(cur_a_acc) / 100
    return acc, kappa, auc, a_acc


ovcare_table = '''
Split 0& 94.44\% & 82.71\% & 91.71\% & 77.88\% & 36.91\% & 63.76\% & 0.5363 & 0.9067 & 76.73\%\\
Split 0 + Real& 83.95\% & 90.98\% & 93.95\% & 21.25\% & 62.52\% & 69.69\% & 0.5685 & 0.9275 & 70.53\% \\
Split 0 + Fake& 98.25\% & 81.20\% & 91.62\% & 53.17\% & 80.16\% & 82.75\% & 0.7500 & 0.9548 & \textbf{80.88\%} \\
\midrule
Split 1& 15.85\% & 37.50\% & 99.07\% & 4.65\% & 94.69\% & 86.47\% & 0.7647 & 0.8123 & 50.35\% \\
Split 1 + Real& 9.26\% & 56.25\% & 86.62\% & 0.00\% & 86.13\% & 77.15\% & 0.6002 & 0.7264 & 47.65\% \\
Split 1 + Fake& 44.13\% & 64.06\% & 99.14\% & 1.16\% & 99.57\% & 92.58\% & 0.8667 & 0.8803 & \textbf{61.61\%} \\
\midrule
Split 2& 90.56\% & 77.18\% & 92.71\% & 15.45\% & 92.98\% & 82.65\% & 0.7635 & 0.9179 & 73.78\% \\
Split 2 + Real& 97.73\% & 95.30\% & 90.12\% & 36.63\% & 64.76\% & 77.52\% & 0.7067 & 0.9395 & 76.91\% \\
Split 2 + Fake& 97.29\% & 98.22\% & 79.60\% & 68.70\% & 44.15\% & 70.81\% & 0.6333 & 0.9278 & \textbf{77.59\%} \\
\midrule
Split 3& 95.48\% & 39.10\% & 71.15\% & 26.42\% & 53.31\% & 64.08\% & 0.5240 & 0.8479 & 57.09\% \\
Split 3 + Real& 91.64\% & 69.92\% & 63.98\% & 22.74\% & 59.78\% & 61.60\% & 0.4898 & 0.8634 & 61.61\% \\
Split 3 + Fake& 94.34\% & 93.98\% & 87.15\% & 91.80\% & 37.89\% & 78.53\% & 0.7133 & 0.9480 & \textbf{81.03\%} \\
\midrule
Split 4& 97.51\% & 96.46\% & 98.09\% & 8.89\% & 66.23\% & 75.02\% & 0.6515 & 0.8998 & \textbf{73.44\%} \\
Split 4 + Real& 89.24\% & 69.03\% & 89.69\% & 5.65\% & 69.96\% & 71.67\% & 0.5990 & 0.8845 & 64.71\% \\
Split 4 + Fake& 86.75\% & 38.99\% & 98.82\% & 76.64\% & 35.93\% & 68.09\% & 0.5842 & 0.9000 & 67.43\% \\
\midrule
Split 5& 6.59\% & 98.32\% & 96.59\% & 1.97\% & 85.96\% & 57.96\% & 0.4445 & 0.8016 & 57.89\% \\
Split 5 + Real& 40.52\% & 66.98\% & 95.86\% & 18.62\% & 47.30\% & 53.17\% & 0.3823 & 0.8247 & 53.85\% \\
Split 5 + Fake& 45.40\% & 92.54\% & 98.13\% & 76.53\% & 34.35\% & 65.72\% & 0.5581 & 0.9050 & \textbf{69.39\%} \\
\midrule
Split 6& 0.13\% & 71.00\% & 94.84\% & 58.88\% & 47.60\% & 56.45\% & 0.4501 & 0.8152 & 54.49\% \\
Split 6 + Real& 48.19\% & 96.52\% & 84.98\% & 32.03\% & 85.54\% & 80.68\% & 0.7202 & 0.9148 & 69.45\% \\
Split 6 + Fake& 29.87\% & 98.16\% & 86.90\% & 48.53\% & 84.22\% & 80.24\% & 0.7196 & 0.9262 & \textbf{69.53\%} \\
\midrule
Split 7& 73.26\% & 42.11\% & 77.93\% & 16.81\% & 68.89\% & 67.06\% & 0.4985 & 0.8329 & 55.80\% \\
Split 7 + Real& 84.11\% & 72.18\% & 76.24\% & 80.40\% & 50.62\% & 61.64\% & 0.4601 & 0.9037 & \textbf{72.71\%} \\
Split 7 + Fake& 52.63\% & 82.71\% & 95.90\% & 77.53\% & 31.73\% & 53.21\% & 0.4066 & 0.9074 & 68.10\% \\
\midrule
Split 8& 17.44\% & 87.87\% & 91.70\% & 30.76\% & 84.92\% & 74.92\% & 0.6374 & 0.8507 & 62.54\% \\
Split 8 + Real& 8.62\% & 82.84\% & 98.17\% & 68.37\% & 87.82\% & 83.10\% & 0.7444 & 0.9424 & \textbf{69.16\%} \\
Split 8 + Fake& 16.80\% & 81.90\% & 89.01\% & 40.63\% & 94.03\% & 77.76\% & 0.6751 & 0.8934 & 64.48\% \\
\midrule
Split 9& 10.02\% & 91.57\% & 88.66\% & 0.00\% & 78.29\% & 71.44\% & 0.6236 & 0.7914 & 53.71\% \\
Split 9 + Real& 8.24\% & 94.30\% & 94.07\% & 15.70\% & 85.29\% & 76.85\% & 0.6841 & 0.8894 & 59.52\% \\
Split 9 + Fake& 29.99\% & 98.08\% & 86.84\% & 35.54\% & 92.68\% & 81.81\% & 0.7540 & 0.9340 & \textbf{68.63\%} \\
'''

tcga_results = '''
 Split 0 & 93.38\% & 93.57\% & 97.77\% & 99.24\% & 89.14\% & 94.54\% & 0.9300 & 0.9970 & 94.62\% \\
 Split 0 + Real & 95.84\% & 100.00\% & 52.62\% & 99.87\% & 89.14\% & 91.28\% & 0.8873 & 0.9790 & 87.49\% \\
 Split 0 + Fake & 98.51\% & 99.74\% & 71.24\% & 99.79\% & 91.72\% & 94.71\% & 0.9316 & 0.9936 & 92.20\% \\
 \midrule
 Split 1 & 89.03\% & 90.45\% & 98.83\% & 98.24\% & 98.20\% & 93.77\% & 0.9192 & 0.9882 & 94.95\% \\
 Split 1 + Real & 84.71\% & 90.74\% & 22.14\% & 90.72\% & 99.64\% & 87.09\% & 0.8296 & 0.9549 & 77.59\% \\
 Split 1 + Fake & 72.32\% & 90.83\% & 30.99\% & 84.48\% & 99.77\% & 83.21\% & 0.7775 & 0.9546 & 75.68\% \\
 \midrule
 Split 2 & 73.19\% & 97.74\% & 100.00\% & 90.86\% & 99.78\% & 90.34\% & 0.8721 & 0.9966 & 92.31\% \\
 Split 2 + Real & 90.82\% & 73.07\% & 100.00\% & 86.18\% & 95.80\% & 85.21\% & 0.8081 & 0.9901 & 89.17\% \\
 Split 2 + Fake & 84.67\% & 91.75\% & 100.00\% & 95.44\% & 92.44\% & 91.35\% & 0.8869 & 0.9936 & 92.86\% \\
 \midrule
 Split 3 & 94.30\% & 57.79\% & 98.70\% & 95.20\% & 94.97\% & 86.92\% & 0.8226 & 0.9803 & 88.19\% \\
 Split 3 + Real & 97.35\% & 47.86\% & 100.00\% & 96.16\% & 98.96\% & 87.22\% & 0.8260 & 0.9985 & 88.07\% \\
 Split 3 + Fake & 99.21\% & 96.04\% & 100.00\% & 78.46\% & 96.28\% & 96.12\% & 0.9459 & 0.9952 & 94.00\% \\
 \midrule
 Split 4 & 99.92\% & 84.25\% & 90.97\% & 98.48\% & 81.10\% & 89.42\% & 0.8575 & 0.9874 & 90.95\% \\
 Split 4 + Real & 99.71\% & 97.92\% & 94.61\% & 100.00\% & 98.04\% & 98.43\% & 0.9786 & 0.9995 & 98.06\% \\
 Split 4 + Fake & 95.40\% & 92.37\% & 98.12\% & 99.77\% & 98.82\% & 96.48\% & 0.9521 & 0.9988 & 96.90\% \\
 \midrule
 Split 5 & 91.95\% & 87.70\% & 81.15\% & 82.44\% & 99.49\% & 90.75\% & 0.8783 & 0.9757 & 88.55\% \\
 Split 5 + Real & 94.84\% & 70.43\% & 65.43\% & 96.06\% & 99.84\% & 86.69\% & 0.8277 & 0.9681 & 85.32\% \\
 Split 5 + Fake & 94.58\% & 91.11\% & 82.62\% & 95.71\% & 99.93\% & 94.27\% & 0.9249 & 0.9911 & 92.79\% \\
 \midrule
 Split 6 & 95.28\% & 76.16\% & 91.78\% & 98.57\% & 94.74\% & 91.72\% & 0.8930 & 0.9699 & 91.30\% \\
 Split 6 + Real & 97.47\% & 71.29\% & 99.84\% & 92.40\% & 98.16\% & 91.33\% & 0.8881 & 0.9823 & 91.83\% \\
 Split 6 + Fake & 99.71\% & 55.73\% & 99.51\% & 100.00\% & 95.89\% & 89.80\% & 0.8679 & 0.9910 & 90.17\% \\
 \midrule
 Split 7 & 67.25\% & 77.23\% & 57.09\% & 98.23\% & 99.80\% & 81.41\% & 0.7559 & 0.9706 & 79.92\% \\
 Split 7 + Real & 94.11\% & 78.35\% & 89.15\% & 80.35\% & 99.80\% & 89.47\% & 0.8628 & 0.9856 & 88.35\% \\
 Split 7 + Fake & 75.69\% & 64.31\% & 97.45\% & 99.44\% & 99.91\% & 84.50\% & 0.8002 & 0.9812 & 87.36\% \\
 \midrule
 Split 8 & 93.28\% & 94.03\% & 87.80\% & 99.82\% & 99.11\% & 94.64\% & 0.9298 & 0.9956 & 94.81\% \\
 Split 8 + Real & 93.15\% & 99.72\% & 48.00\% & 99.45\% & 98.87\% & 90.63\% & 0.8769 & 0.9960 & 87.84\% \\
 Split 8 + Fake & 99.32\% & 96.23\% & 44.52\% & 97.91\% & 99.35\% & 91.64\% & 0.8894 & 0.9837 & 87.47\% \\
 \midrule
 Split 9 & 88.23\% & 97.62\% & 58.56\% & 96.84\% & 97.85\% & 91.03\% & 0.8843 & 0.9913 & 87.82\% \\
 Split 9 + Real & 95.95\% & 99.78\% & 64.44\% & 97.05\% & 99.71\% & 94.34\% & 0.9271 & 0.9929 & 91.39\% \\
 Split 9 + Fake & 98.68\% & 99.01\% & 95.81\% & 97.00\% & 99.75\% & 98.13\% & 0.9759 & 0.9994 & 98.05\% \\
'''


def tcga_parser(w_acc, kappa, auc, a_acc):

    def _parser(table):
        tcga_results = table.split('\n')
        tcga_results = [x.strip() for x in tcga_results if x.strip() != '']
        tcga_results = tcga_results[1:]
        tcga_data = np.zeros(30).tolist()
        for combination_idx, tcga_result in enumerate(tcga_results):
            _res = tcga_result.split('|')
            _res = [x.strip() for x in _res if x.strip() != '']
            tcga_data[combination_idx] = float(_res[-1])
            tcga_data[combination_idx + 10] = float(_res[-2])
            tcga_data[combination_idx + 20] = float(_res[-3])
        return tcga_data

    return _parser(w_acc), _parser(kappa), _parser(auc), _parser(a_acc)


ovcare_w_acc, ovcare_kappa, ovcare_auc, ovcare_a_acc = gan_eval_analyze_table(
    ovcare_table)
#tcga_w_acc, tcga_kappa, tcga_auc, tcga_a_acc = tcga_parser(tcga_acc, tcga_kappa, tcga_auc, tcga_a_acc)
tcga_w_acc, tcga_kappa, tcga_auc, tcga_a_acc = gan_eval_analyze_table(
    tcga_results)

# metric = 'Average Accuracy'
# tcga_data = tcga_a_acc
# ovcare_metric = ovcare_a_acc
# ylim = (0.4, 1)

metric = 'Weighted Accuracy'
tcga_data = tcga_w_acc
ovcare_metric = ovcare_w_acc
ylim = (0.4, 1)

# metric = 'AUC'
# tcga_data = tcga_auc
# ovcare_metric = ovcare_auc
# ylim = (0.7, 1)

# metric = 'Kappa'
# tcga_data = tcga_kappa
# ovcare_metric = ovcare_kappa
# ylim = (0.2, 1)

print('----- TCGA -------')
gan_eval_boxplots(tcga_w_acc, tcga_kappa, tcga_auc, tcga_a_acc)
print('----- OVCARE -------')
gan_eval_boxplots(ovcare_w_acc, ovcare_kappa, ovcare_auc, ovcare_a_acc)


plt.style.use('seaborn-darkgrid')
rc('text', usetex=True)
# colors = ['#f8dadb', '#ee848b', '#de425b']
colors = ['#ffffff', '#9a9a9a', '#ffffff']
hatches = ['', '', '///', '', '', '///']
labels = ['Baseline', 'Baseline + Real', 'Baseline + Synthetic']

cls_list = ['Baseline'] * 10 + ['Baseline + Real'] * \
    10 + ['Baseline + Synthetic'] * 10

data = {'OVCARE': ovcare_metric,
        'TCGA': tcga_data,
        'Splits': cls_list}

df = pd.DataFrame(data)

df_melt = df.melt(id_vars='Splits',
                  value_vars=['OVCARE', 'TCGA'],
                  var_name='Datasets')


ax = sns.boxplot(data=df_melt,
                 hue='Splits',
                 palette=colors,
                 x='Datasets',
                 y='value',
                 order=['OVCARE', 'TCGA'], width=0.8, linewidth=0.6)

for hatch, patch in zip(hatches, ax.artists):
    patch.set_hatch(hatch)

leg_artists = []
for i in range(3):
    p = matplotlib.patches.Patch(
        facecolor=colors[i], hatch=hatches[i], label=labels[i])
    leg_artists.append(p)

l = ax.legend(handles=leg_artists, loc='best')
l.set_title('')
ax.set_ylim(*ylim)

plt.ylabel(metric)
plt.savefig('bw_pattern_weighted_acc.pdf', dpi=300)

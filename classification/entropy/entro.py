import numpy as np
import h5py
from PIL import Image
import random
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import pandas as pd
import seaborn as sns
import scipy
from statannot import add_stat_annotation

seed = 1234

def entropy(rgb_image):
    '''
    function returns entropy of a image
    signal must be a 1-D numpy array
    '''
    signal=np.asarray(rgb_image.convert('L')).flatten()
    lensig=signal.size
    symset=list(set(signal))
    numsym=len(symset)
    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
    ent=np.sum([p*np.log2(1.0/p) for p in propab])
    return ent

def select_images_from_dir(img_dir, k=5000):
    if not isinstance(img_dir, list):
        images = glob.glob(os.path.join(img_dir, '*.png'))
    else:
        images = img_dir
    random.seed(seed)
    selected = random.sample(images, k=k)
    images = []
    for s in tqdm(selected):
        images += [Image.open(s).convert('RGB').resize((1024, 1024), Image.LANCZOS)]
    return images

def compute_entropy(a_list):
    e = []
    for a in tqdm(a_list):
        e += [entropy(a)]
    return e

if not os.path.isfile('./vae.npy'):
    vae = select_images_from_dir('/projects/ovcare/jpeng/gan_paper/media/molinux02_data/Jason/005_gan_paper/data/vae_hgsc')
    e_vae = compute_entropy(vae)
    temp = np.asarray(e_vae)
    np.save('vae.npy', e_vae)
else:
    e_vae = np.load('./vae.npy').tolist()

if not os.path.isfile('./esr.npy'):
    esr = select_images_from_dir('/projects/ovcare/jpeng/gan_paper/media/molinux02_data/Jason/005_gan_paper/data/esrgan/ground_truth_HGSC_images')
    e_esr = compute_entropy(esr)
    temp = np.asarray(e_esr)
    np.save('esr.npy', e_esr)
else:
    e_esr = np.load('./esr.npy').tolist()

if not os.path.isfile('./dst.npy'):
    dst = select_images_from_dir(glob.glob('/projects/ovcare/jpeng/gan_paper/dataSyn_256_40x/HGSC/**/*.png'))
    e_dst = compute_entropy(dst)
    temp = np.asarray(e_dst)
    np.save('dst.npy', e_dst)
else:
    e_dst = np.load('./dst.npy').tolist()

if not os.path.isfile('./pro.npy'):
    pro = select_images_from_dir(glob.glob('/projects/ovcare/classification/1024_fake_sorted/Tumor/HGSC/FAKE_WSI/*'))
    e_pro = compute_entropy(pro)
    temp = np.asarray(e_pro)
    np.save('pro.npy', e_pro)
else:
    e_pro = np.load('./pro.npy').tolist()

if not os.path.isfile('./real.npy'):
    rea = select_images_from_dir(glob.glob('/projects/ovcare/classification/1024_Patches/HGSC/**/Tumor/*'))
    e_rea = compute_entropy(rea)
    temp = np.asarray(e_rea)
    np.save('real.npy', e_rea)
else:
    e_rea = np.load('./real.npy').tolist()

entropys = e_vae + e_esr + e_dst + e_pro + e_rea
colors = ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600']
labels = ['VAE', 'ESRGAN', 'DST', 'PROGAN', 'REAL']
dict_entropy = {
    'VAE': e_vae,
    'ESRGAN': e_esr,
    'DST': e_dst,
    'PROGAN': e_pro,
    'REAL': e_rea
}
cls_list = ['VAE'] * 5000 + ['ESRGAN'] * 5000 + ['DST'] * 5000 + ['PROGAN'] * 5000 + ['REAL'] * 5000

plt.style.use('seaborn-darkgrid')
#sns.set_style(style='white')
rc('text', usetex=True)

data = {'Entropy': entropys,
        'Source' : cls_list}

df = pd.DataFrame(data)

sns.set_style(style='white')
ax = sns.boxplot(data = df,
                palette=colors,
                x = 'Source',
                y = 'Entropy', width=0.8, linewidth=0.6, showfliers = False)

#l.set_title('')
#ax.set_ylim(*ylim)

#plt.title('Boxplot grouped by Splits') # You can change the title here
plt.ylabel('Entropy')
#plt.show()
plt.savefig('entropy.pdf', dpi=600)

print('==== Std and Mean =======')
for label in labels:
    d = np.asarray(dict_entropy[label])
    print('{}, Std {}, Mean {}'.format(label, d.std(), d.mean()))

print('==== Pair-wise t-test =====')
for index, out in enumerate(labels):
    for inner in labels[index:]:
        if out != inner:
            print('{} vs {}: {}'.format(out, inner, scipy.stats.ttest_ind(dict_entropy[out], dict_entropy[inner]).pvalue))
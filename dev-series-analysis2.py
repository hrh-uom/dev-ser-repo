import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.animation as animation
import glob
from scipy.spatial import Delaunay
from PIL import Image
import os
from IPython.display import HTML
from skimage.draw import line
import alphashape
from descartes import PolygonPatch
from shapely.geometry import Point
import itertools
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.stats import ks_1samp

atom=False

plt.style.use('~/dbox/4-Thesis/stylesheet.mplstyle')

pxsize1=200/411 #13kx
pxsize2=500/379.5 #4800x
days=np.array([1, 2, 3, 4, 5, 6,8, 9, 10, 14,16, 18, 21, 25,28, 30, 35, 42, 49, 56 ])
lin_days=np.linspace(days[0], days[-1], 101)
dir_out='./plots-for-thesis/'
N=len(days)

def import_from_cellprofiler(days):
    sampleimagepath=np.random.choice(glob.glob(f"../EM/*/day*/output/*tiff"))
    nx, ny  = Image.open(sampleimagepath).size
    fibril_dfs=[] #need the first element empty so fibril_dfs can be indexed from 1 not 0, as data is labelled day 1-10
    border_dfs=[] #same as fibrildfs, but with the border fibrils included
    image_mmts_dfs=[] #Measurements such as FVF
    cells_dfs=[] #Properties of masked regions
    maskpathnames=[]
    for i in days:
        fpattern_g1=f'../EM/*/day{str(i).zfill(2)}/output/'
        fpattern_g2=f'../EM/group2*/analysed-final/day{str(i).zfill(2)}/'
        fpattern=fpattern_g1 if i<=10 else fpattern_g2
        f_fibs='*FibrilFiltered*.csv'
        f_borders='*FilteredWithBorders*.csv'
        f_image='*Image*.csv'
        f_cells='*Cells*.csv'
        f_masks='*mask*'

        try:
            maskpathnames.append(glob.glob(fpattern+f_masks))
        except:
            maskpathnames.append(0)


        try:
            fibril_dfs.append(pd.read_csv(glob.glob(fpattern+f_fibs)[0]))
        except:
            fibril_dfs.append(0)

        try:
            border_dfs.append(pd.read_csv(glob.glob(fpattern+f_borders)[0]))
        except:
            border_dfs.append(0)

        try:
            image_mmts_dfs.append(pd.read_csv(glob.glob(fpattern+f_image)[0]))
        except:
            image_mmts_dfs.append(0)
        try:
            cells_dfs.append(pd.read_csv(glob.glob(fpattern+f_cells)[0]))
        except:
            cells_dfs.append(0)
    return nx, ny, fibril_dfs, border_dfs, image_mmts_dfs, cells_dfs, maskpathnames
nx, ny, fibril_dfs, border_dfs, image_mmts_dfs, cells_dfs, maskpathnames=import_from_cellprofiler(days)
#%%------------------------------FIBRIL VOLUME FRACTION --------------------------------

def import_selected_images(filepath='./images-no-cells-list.csv'):
    selected_df=pd.read_csv(filepath )
    filenames=list(map(lambda s: s.lower(), list(map(os.path.basename,     selected_df.file.to_list())))) #finds filename and makes it lowercase
    tempInumber=list(np.array(list(map(lambda s: s.split('_', 3), filenames)))[:, -1]) #Removes the hui part, leaves only number and .tiff
    imID=list(np.array(list(map(lambda s: s.split('_mask', 1),tempInumber)))[:, 0]) #Matches Metadata_imageNumber column now
    return pd.concat((selected_df.timepoint, pd.DataFrame(imID, columns=['imageID'])), axis=1)
def find_timepoint_index(t):
    """
    Finds the index in days which corresponds to timepoint t
    """
    return np.argwhere(days==t)[0,0]
def measure_cell_coverage(t):
    """
    Measures cell coverage in image, based on manual masks
    IN PIXELS
    """
    # t=16

    i=find_timepoint_index(t)
    df=cells_dfs[i]

    imgs_cells=np.unique(df.ImageNumber) #IMages that actually have cells in!!
    allims=np.unique(fibril_dfs[i].ImageNumber) #allimages
    firsIm=allims[0] ; lastim = allims[-1]
    cell_coverage_arr=np.zeros(allims.shape) ; k=0
    for j in allims: #J goes through image numbers, k ticks through from 0
        if j in imgs_cells:
            cell_coverage_arr[k]=np.sum(df[df.ImageNumber==j].AreaShape_Area)
        k+=1

    return cell_coverage_arr

def calculate_FVF():
    FVF_list=[]
    for t in days:
        i=find_timepoint_index(t)
        if np.any(image_mmts_dfs[i]): #tests if image file exists
            fibril_coverage=image_mmts_dfs[i].AreaOccupied_AreaOccupied_FilteredWithBorders.to_numpy() #An array of length the number of images for timepoint t
            cell_coverage=measure_cell_coverage(t)
            # print(f' i {i} t {t} len cell coverage {cell_coverage.shape} len fibril_coverage {fibril_coverage.shape}')
            FVF=(fibril_coverage -cell_coverage)/(np.full( len(fibril_coverage), nx*ny))
            FVF_list.append(FVF)
    means=  [np.mean    (FVF_list[i])                                                   for i in np.arange(N)]
    stds=   [0 if np.any(FVF_list[i])==False else np.std     (FVF_list[i], ddof=1)      for i in np.arange(N)]
    l =     [len(FVF_list[i]) for i in np.arange(N)]
    return means, stds, l
def find_FVFs_best_image():
    """
    Reads FVFs from the selected images which contain no cells
    """
    FVFs=[]
    for t in days:
        selected_df=import_selected_images()
        imagestr=selected_df[selected_df.timepoint==t].imageID.item()
        df=image_mmts_dfs[find_timepoint_index(t)]
        area_occupied=(df[df['URL_Original'].str.contains(imagestr)].AreaOccupied_AreaOccupied_FilteredWithBorders).to_numpy()[0]
        FVF=area_occupied/(nx*ny)
        FVFs.append(FVF)
    return FVFs

def plateau(x, M, a):
    """
    Plateau function. M= height, a=how fast it gets there
    """
    return M*(1-np.exp(-x/a))
    # return M-b*np.exp(-x/a)

def plot_FVF():
    FVFs, e, l=calculate_FVF()
    fig,ax=plt.subplots()
    # ax.set_title("Evolution of Fibril Volume Fraction")
    pars, cov=curve_fit(plateau, days, FVFs)

    ax.errorbar(days,FVFs, yerr=e,fmt='none',color='k', capsize=6, capthick=1.5,zorder=2)
    ax.scatter(days, FVFs, label='Mean, all images', zorder=3)
    textstr=f'$M$ = {pars[0]:.3f}     \n$T$ = {pars[1]:.3} days'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)    # these are matplotlib.patch.Patch properties
    ax.text(0.75, 0.2, textstr, transform=ax.transAxes, fontsize=18,verticalalignment='top', bbox=props)
    ax.plot(lin_days, plateau(lin_days, *pars), 'r', zorder=1)

    # ax.plot(days,find_FVFs_best_image(), 'r', label='Decel select image')
    ax.set_xticks(ticks=np.arange(0, 57, 7))
    ax.set_xlabel('Day postnatal')
    ax.set_xlim(0, days[-1]+2)

    ax.set_ylabel('Fibril volume fraction')
    # for i in np.arange(N):
    #     ax.annotate(f"n={l[i]}", (days[i]+0.05, FVFs[i]+0.02))
    # plt.legend()
    plt.savefig(dir_out+'FVF')
    if atom:
        plt.show()
# calculate_FVF()
plot_FVF()

#%%.......................................................................................
# ------------------------------FIBRIL DIAMETER DISTRIBUTION --------------------------------
#.......................................................................................

def MFD_histogram_grid():
    rows, cols = 5,4; i=0 #Ticks through days
    fig, ax = plt.subplots(rows, cols,sharex='all', figsize=(20,18))
    # fig.suptitle("Evolution of Fibril Diameters (MFD)")


    for r in range(rows):
        for c in range(cols):
            pxsize=pxsize1 if i<find_timepoint_index(14) else pxsize2
            x=fibril_dfs[i].AreaShape_MinFeretDiameter*pxsize
            ax[r,c].hist(x, bins=np.arange(0, 350, 10), density=True)
            # ax[r,c].title.set_text(f'{i}')
            ax[r,c].xaxis.set_tick_params(which='both', labelbottom=True)
            ax[r,c].text(0.8, 0.9, f'Day {days[i]}', fontsize=18, horizontalalignment='center',verticalalignment='center', transform=ax[r,c].transAxes)
            # ax[r,c].set_xlabel('Minimum Feret diameter (nm)')
            i+=1
            if i>=N:
                break
    fig.add_subplot(111, frameon=False)# Adding common axis  https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
    plt.grid(False)
    plt.tick_params(labelcolor='none', pad=20, which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Probability density");    plt.xlabel("Minimum Feret diameter (nm)")
    fig.tight_layout(); plt.savefig(dir_out+'MFDgrid');
    if atom:
        plt.show()

# MFD_histogram_grid()

#%%ANIMATION

def create_MFD_animation():
    # Fixing bin edges
    HIST_BINS = np.linspace(0, 400, 50)

    # histogram our data with numpy
    i=0
    data = fibril_dfs[i].AreaShape_MinFeretDiameter*pxsize1
    n, _ = np.histogram(data, HIST_BINS, density=True)
    fig, ax1 = plt.subplots()
    fig.suptitle("Evolution of fibril diameters (MFD)")
    ax1.set_title(f"Day{0}")
    ax1.set_xlabel('Fibril Diameter (Minimum Feret Diameter) (nm)')
    ax1.set_ylabel('Density')
    ax1.set_xlim(0, 400)
    # ax.text(0.5, 1.100, "y=sin(x)", bbox={'facecolor': 'red',
    def prepare_animation(bar_container):
        def animate(i):
            # simulate new data coming in
            pxsize=pxsize1 if i<find_timepoint_index(14) else pxsize2
            data = fibril_dfs[i].AreaShape_MinFeretDiameter*pxsize
            ax1.set_title(f"Day{days[i]}")
            n, _ = np.histogram(data, HIST_BINS, density=True)
            ax1.set_ylim(top=1.1*np.max(n))  # set safe limit to ensure that all data is visible.

            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)
            return bar_container.patches
        return animate

    _, _, bar_container = ax1.hist(data, HIST_BINS, lw=1, density=True,
                                  ec="yellow", fc="green", alpha=0.5)




    ani = animation.FuncAnimation(fig, prepare_animation(bar_container), N,
                                  repeat=False, blit=True, interval=800)
    plt.close()
    return ani

# ani=create_MFD_animation()
# ani.save(dir_out+'MFD.mp4')
# if atom:
#     HTML(ani.to_html5_video())

#%%MFD Mean/STD

def violinwidths():
    widths_=np.zeros(N)

    for i in range (N):
        widths_[i]=5
        if days[i] < 35:
            widths_[i]=2
        if days[i] < 14:
            widths_[i]=1
    return widths_
pxsize_arr=[pxsize1 if i<find_timepoint_index(14) else pxsize2 for i in range(N)]


MFDs        =   [    (fibril_dfs[i].AreaShape_MinFeretDiameter*pxsize_arr[i])           for i in range(N)]
meanMFDs    =   [np.mean(i) for i in MFDs]
maxMFDs     =   np.array([np.max(i) for i in MFDs])
MFDs_Q3     =   [np.quantile(i, .75) for i in MFDs]
MFDs_Q3     =   [np.quantile(i, .25) for i in MFDs]
minMFDs     =   np.array([np.min(i) for i in MFDs])
rangeMFDs   =   maxMFDs-minMFDs

def plot_MFDs(even=False):

    if not even:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.violinplot(MFDs, positions=days, showmeans=True, widths=violinwidths())
        ax.set_xticks(ticks=np.arange(0, 63, 7))
        ax.set_ylabel('Minimum Feret Diameter (nm)')
        ax.set_xlabel('Day postnatal')
        plt.savefig(dir_out+'MFD')

    #EVEN SPACING
    else:
        fig2, ax2 = plt.subplots(figsize=(15, 8))
        ax2.violinplot(MFDs, showmeans=True, widths=0.9)
        ax2.set_xticks(np.arange(1, N+1))
        ax2.set_xlim(0.5, N+1)
        ax2.set_xticklabels(days)
        ax2.set_ylabel('Minimum Feret Diameter (nm)')
        ax2.set_xlabel('Day postnatal')
        plt.savefig(dir_out+'MFD_even')
        if atom:
            plt.show()
    return meanMFDs

plot_MFDs(even=True)

def linear(x, m, c):
    return m*x + c

def plot_MFD_range():
    fig, ax=plt.subplots()
    ax.set_ylabel('Range of fibril MFD (nm)')
    ax.set_xlabel('Day postnatal')
    ax.scatter(days, rangeMFDs)
    pars, cov = curve_fit(linear, days, rangeMFDs)
    ax.plot(lin_days,linear(lin_days, pars[0], pars[1]), 'r')
    ax.set_xticks(ticks=np.arange(0, 57, 7))

    textstr=f'$g$ = {pars[0]:.1f} nm/day\n$y_0$ = {pars[1]:.3} nm'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)    # these are matplotlib.patch.Patch properties
    ax.text(0.65, 0.2, textstr, transform=ax.transAxes, fontsize=18,verticalalignment='top', bbox=props)
    plt.savefig(dir_out+'mfd_range')
    if atom:
        plt.show()

plot_MFD_range()


#%% make bimodal fits df

def make_multimodal_df():
    list=[['day']]+[[f'unimodal_guess_u{i}',f'unimodal_guess_s{i}'] for i in range(1)]+[[f'bimodal_guess_u{i}',f'bimodal_guess_s{i}'] for i in range(2)]+[['bimodal_guess_w']]+[[f'trimodal_guess_u{i}',f'trimodal_guess_s{i}'] for i in range(3)]+[[f'trimodal_guess_w{i}'] for i in range(3)]+[[f'unimodal_fitted_u{i}',f'unimodal_fitted_s{i}'] for i in range(1)]+[[f'bimodal_fitted_u{i}',f'bimodal_fitted_s{i}'] for i in range(2)]+[['bimodal_fitted_W']]+[[f'trimodal_fitted_u{i}',f'trimodal_fitted_s{i}'] for i in range(3)]+[[f'trimodal_fitted_w{i}'] for i in range(3)]+[['unimodal_KS','bimodal_KS','trimodal_KS']]
    newlist = [item for list in list for item in list]

    df=pd.DataFrame(columns=newlist)
    df.day=days
    df.to_csv(dir_out+'multimodalfits.csv')


#%%Fit multimodal
def normal_pdf(x, u0, s0):
    """
    Normal distribution
    """
    A   =   1       /          (s0 * np.sqrt(2*np.pi))
    B   =   0.5     *          ((x-u0)/s0)**2
    return A * np.exp (-B)
def bi_pdf(x, u0,  s0, u1, s1 , w):
    """
    bimodal Normal distribution
    """
    return w * normal_pdf(x, u0, s0) + (1-w) * normal_pdf(x, u1, s1)
def tri_pdf(x, u0,s0, u1,s1, u2, s2, w0, w1, w2):
    """
    trimodal Normal distribution
    """
    return (w0 * normal_pdf(x, u0, s0) + w1 * normal_pdf(x, u1, s1)+ w2 * normal_pdf(x, u2, s2))/(w0+w1+w2)
def normal_cdf(x, u0, s0):
    X=erf((x-u0)/(s0*np.sqrt(2)))
    return 0.5 * (1 + X)
def bi_cdf(x, u0,  s0, u1, s1 , w):
    return w * normal_cdf(x, u0, s0) + (1-w) * normal_cdf(x, u1, s1)
def tri_cdf(x, u0,s0, u1,s1, u2, s2, w0, w1, w2):
    """
    trimodal Normal distribution
    """
    return (w0 * normal_cdf(x, u0, s0) + w1 * normal_cdf(x, u1, s1)+ w2 * normal_cdf(x, u2, s2))/(w0+w1+w2)
def fit_multimodal(t, pvalue):

    multimodal_df=pd.read_csv(dir_out+'multimodal/multimodalfits.csv', index_col=0)
    cols=multimodal_df.columns.to_list()
    i=find_timepoint_index(t)

    xmax=np.ceil(np.max(MFDs[i])/100)*100 #Round up to the nearest 100
    binsize=10 ;bins_=np.linspace(0, xmax, 40) ;  xlin=np.linspace(0, xmax, 1001)

    #Initial guesses
    uni_p0s=multimodal_df[[x for x in cols if x.startswith('unimodal_guess')]].iloc[i].to_list()
    bi_p0s=multimodal_df[[x for x in cols if x.startswith('bimodal_guess')]].iloc[i].to_list()
    tri_p0s=multimodal_df[[x for x in cols if x.startswith('trimodal_guess')]].iloc[i].to_list()

    #Fit cols
    uni_fit_cols=[x for x in cols if x.startswith('unimodal_fitted')]
    bi_fit_cols=[x for x in cols if x.startswith('bimodal_fitted')]
    tri_fit_cols=[x for x in cols if x.startswith('trimodal_fitted')]


    fig, ax=plt.subplots()

    kde=gaussian_kde(MFDs[i])
    yfit=kde(xlin)


    try:
        pars_uni, cov_uni=curve_fit(normal_pdf, xlin, yfit, p0=uni_p0s)
        index=[multimodal_df.columns.get_loc(j) for j in uni_fit_cols];multimodal_df.iloc[i, index]=pars_uni

    except:
        print("cannot fit unimodal")
        index=[multimodal_df.columns.get_loc(j) for j in uni_fit_cols];multimodal_df.iloc[i, index]=0
    try:
        pars_bi, cov_bi=curve_fit(bi_pdf, xlin, yfit, p0=bi_p0s)
        index=[multimodal_df.columns.get_loc(j) for j in bi_fit_cols];multimodal_df.iloc[i, index]=pars_bi

    except:
        print("cannot fit bimodal")
        index=[multimodal_df.columns.get_loc(j) for j in bi_fit_cols];multimodal_df.iloc[i, index]=0
    try:
        pars_tri, cov_tri=curve_fit(tri_pdf, xlin, yfit, p0=tri_p0s)
        ws=pars_tri[-3:]/np.sum(pars_tri[-3:]) ; pars_tri[-3:]=ws
        index=[multimodal_df.columns.get_loc(j) for j in tri_fit_cols];multimodal_df.iloc[i, index]=pars_tri

    except:
        print("cannot fit trimodal")
        index=[multimodal_df.columns.get_loc(j) for j in tri_fit_cols];multimodal_df.iloc[i, index]=0
    ax.plot(xlin, yfit, '--k', label='KDE')
    ax.set_title(f'Day {t}')
    ax.hist(MFDs[i], density=True, bins=bins_, histtype='bar', edgecolor='black', alpha=0.2)
    ax.set_xlim(0, xmax+10)
    ax.set_ylabel('Probability density')
    ax.set_xlabel('Minimum Feret Diameter (nm)')

    ks_pvalues=np.zeros(3)

    ks_pvalues[0]=ks_1samp(MFDs[i], normal_cdf, args=tuple(pars_uni))[1]
    multimodal_df.iat[i,multimodal_df.columns.get_loc('unimodal_KS')]=ks_pvalues[0]

    ks_pvalues[1]=ks_1samp(MFDs[i], bi_cdf, args=tuple(pars_bi))[1]
    multimodal_df.iat[i,multimodal_df.columns.get_loc('bimodal_KS')]=ks_pvalues[1]


    ks_pvalues[2]=ks_1samp(MFDs[i], tri_cdf, args=tuple(pars_tri))[1]
    multimodal_df.iat[i,multimodal_df.columns.get_loc('trimodal_KS')]=ks_pvalues[2]

    labels=['Normal ', 'Bimodal ', 'Trimodal ']
    for i in range(3):
        if ks_pvalues[i]>pvalue:
            labels[i]=labels[i]+'*'
        # if i==np.argmax(ks_pvalues):
        #     labels[i]=labels[i]+' \u25B2'

    ax.plot(xlin, normal_pdf(xlin, *pars_uni), 'r', label=labels[0])
    ax.plot(xlin, bi_pdf(xlin, *pars_bi), 'g', label=labels[1])
    ax.plot(xlin, tri_pdf(xlin, *pars_tri), '-b', label=labels[2])

    ax.legend(); plt.savefig(dir_out+f'multimodal/mmodal_day{t}')
    multimodal_df.to_csv(dir_out+'multimodal/multimodalfits.csv')
    if atom:
        plt.show()
    # print(ks_1samp(MFDs[i], normal_cdf, args=tuple(pars_uni)))
    # print(pars_uni)
    # print(ks_1samp(MFDs[i], bi_cdf, args=tuple(pars_bi)))
    # print(pars_bi)
    # # print(ks_1samp(MFDs[i], tri_cdf, args=tuple(pars_tri)))
    # print(pars_tri)
[fit_multimodal(ii, 0.05) for ii in days]
#%%Multimodal plots
def multimodal_plots():
    multimodal_df=pd.read_csv(dir_out+'multimodal/multimodalfits.csv', index_col=0)

    uni_fit_cols=[x for x in cols if x.startswith('unimodal_fitted')]
    bi_fit_cols=[x for x in cols if x.startswith('bimodal_fitted')]
    tri_fit_cols=[x for x in cols if x.startswith('trimodal_fitted')]

    fig, ax=plt.subplots()

    for i in range(2):
        ax.scatter(days, multimodal_df[bi_fit_cols[2*i]], label=f'Peak {i+1}')
    ax.set_xlabel('Day postnatal'); ax.set_ylabel('Minimum Feret diameter (nm)')
    ax.set_xticks(ticks=np.arange(0, 57, 7))

    ax.legend() ; plt.savefig(dir_out+'multimodal/bimodal_peaks');plt.show()

    fig, ax=plt.subplots()

    for i in range(3):
        ax.scatter(days, multimodal_df[tri_fit_cols[2*i]], label=f'Peak {i+1}')

    ax.set_xlabel('Day postnatal'); ax.set_ylabel('Minimum Feret diameter (nm)')
    ax.set_xticks(ticks=np.arange(0, 57, 7))

    ax.legend() ; plt.savefig(dir_out+'multimodal/trimodal_peaks');plt.show()
multimodal_plots()

#%%............................FIBRIL AREA .............................................

def area_histogram_grid():
    rows, cols = 5,4; i=0 #Ticks through days
    fig, ax = plt.subplots(rows, cols,sharex='all', figsize=(20,18))
    # fig.suptitle("Evolution of Fibril Diameters (MFD)")


    for r in range(rows):
        for c in range(cols):
            pxsize=pxsize1 if i<find_timepoint_index(14) else pxsize2
            x=fibril_dfs[i].AreaShape_Area*pxsize**2/(1000*1000)
            ax[r,c].hist(x, density=True)
            ax[r,c].xaxis.set_tick_params(which='both', labelbottom=True)
            ax[r,c].text(0.8, 0.9, f'Day {days[i]}', fontsize=18, horizontalalignment='center',verticalalignment='center', transform=ax[r,c].transAxes)
            i+=1
            if i>=N:
                break
    fig.add_subplot(111, frameon=False)# Adding common axis  https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
    plt.grid(False)
    plt.tick_params(labelcolor='none', pad=20, which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Probability density");    plt.xlabel("Fibril Area ($\mu$m$^2$)")
    fig.tight_layout(); plt.savefig(dir_out+'areagrid');
    if atom:
        plt.show()

# area_histogram_grid()

def create_area_animation():
    # Fixing bin edges
    xscale=1000*1000
    xlim1, xlim2=0/xscale, 125000/xscale
    HIST_BINS = np.linspace(xlim1, xlim2, 50)

    # histogram our data with numpy
    i=0
    data = fibril_dfs[i].AreaShape_Area*pxsize1**2/xscale
    n, _ = np.histogram(data, HIST_BINS, density=True)
    fig, ax1 = plt.subplots()
    fig.suptitle("Evolution of Fibril Areas")
    ax1.set_title(f"Day{i}")
    ax1.set_xlabel('Area ($\mu$m$^2$)')
    ax1.set_ylabel('Density')
    ax1.set_xlim(xlim1  , xlim2)
    # ax.text(0.5, 1.100, "y=sin(x)", bbox={'facecolor': 'red',
    def prepare_animation(bar_container):
        def animate(i):
            # simulate new data coming in
            pxsize=pxsize1 if i<find_timepoint_index(14) else pxsize2

            data = fibril_dfs[i].AreaShape_Area*pxsize**2/xscale
            ax1.set_title(f"Day{i}")

            n, _ = np.histogram(data, HIST_BINS, density=True)
            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)

            ax1.set_ylim(top=1.1*np.max(n))  # set safe limit to ensure that all data is visible.

            return bar_container.patches
        return animate
    _, _, bar_container = ax1.hist(data, HIST_BINS, lw=1, density=True,
                                  ec="yellow", fc="green", alpha=0.5)
    ani = animation.FuncAnimation(fig, prepare_animation(bar_container), N,
                                  repeat=False, blit=True, interval=800)
    plt.close()
    return ani

# ani=create_area_animation()
# ani.save(dir_out+'area.mp4')
if atom:
    HTML(ani.to_html5_video())

#%%Mean Area

def plot_mean_area(even=False, scale=(1000**2)):
    area=  [ np.array(fibril_dfs[i].AreaShape_Area*pxsize_arr[i]**2)/scale   for i in range(N)]

    if not even:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.violinplot(area, positions=days, showmeans=True, widths=violinwidths())
        ax.set_xticks(ticks=np.arange(0, 63, 7))
        ax.set_ylabel('Mean fibril area ($\mu$m$^2$)')
        ax.set_xlabel('Day postnatal')
        plt.savefig(dir_out+'area')

    #EVEN SPACING
    else:
        fig2, ax2 = plt.subplots(figsize=(15, 8))
        ax2.violinplot(area, showmeans=True, widths=0.9)
        ax2.set_xticks(np.arange(1, N+1))
        ax2.set_xlim(0.5, N+1)
        ax2.set_xticklabels(days)
        ax2.set_ylabel('Mean fibril area ($\mu$m$^2$)')
        ax2.set_xlabel('Day postnatal')
        plt.savefig(dir_out+'area_even')
        if atom:
            plt.show()
# plot_mean_area(even=True)


#%%.---------------------------NETWORK ANALYSIS----------------------
def exportFilesToCsv():
    im_filenames_df=pd.concat([image_mmts_dfs[i].FileName_Original for i in range(N)], axis=1)
    im_filenames_df.columns=['day'+str(i).zfill(2) for i in days]
    im_filenames_df.to_csv(dir_out+'imagesused.csv', index=False)
def find_simplices(t):
    """
    Returns a list of points in xy (nm), with corresponding simplices, and the concave hull alpha shape, points in px, mfds (nm)
    """
    dayID=find_timepoint_index(t)
    imagestr=import_selected_images()[import_selected_images().timepoint==t].imageID.item()
    if t<14:
        df=border_dfs[dayID]
        field='Metadata_FileLocation'
    else:
        df=fibril_dfs[dayID]
        field='FileName_Original'


    df_select=df[df[field].str.contains(imagestr)]
    points_px=np.array([df_select.AreaShape_Center_X,df_select.AreaShape_Center_Y]).T
    MFDs_select=pxsize_arr[dayID]*np.array(df_select.AreaShape_MinFeretDiameter)
    points_px=np.delete(points_px, np.where(MFDs_select==0)[0], axis=0)
    MFDs_select=np.delete(MFDs_select, np.where(MFDs_select==0)[0], axis=0)
    points=points_px*pxsize_arr[dayID]

    tri = Delaunay(points, furthest_site=False)
    alpha_shape=alphashape.alphashape(points, 0.002)
    remove_lis=[]
    for i in range(len(points)):
        x=points[i, 0]
        y=points[i,1]
        point = Point(x,y) # analysis point
        if alpha_shape.contains(point) == False:
            remove_lis.append(i)

    #Removing simplices which are connected to the outer fibrils to prevent really long connections
    newsimplices=[]
    for j in range(len(tri.simplices)):
        if ~np.any(np.in1d(tri.simplices[j] , remove_lis)):
            newsimplices.append(tri.simplices[j])
    newsimplices=np.array(newsimplices)
    return points, newsimplices, alpha_shape, points_px, MFDs_select
def colormap():
    def rgb(a, b, c):
        return (a/255, b/255, c/255)
    def rgba(a, b, c, d):
        return (a/255, b/255, c/255, d)
    fibcol='white' #rgb(10, 82, 129)
    bgcol=rgba(0, 0, 0, 0.1)
    linecol=rgba(2, 86, 214, 0.72)
    centcol=rgba(255, 0, 0, 1)
    hullcolour=rgba(0, 0, 0, 0.2)

    cmap = matplotlib.colors.ListedColormap([bgcol, fibcol])
    return cmap, linecol, centcol
def display_Delaunay_mesh(t):
    print(f"delaunay {t}")
    """
    Shows a plot of a mesh for a particular image
    """
    dayID=find_timepoint_index(t)
    imagestr=import_selected_images().iloc[dayID].imageID.item()
    fig, ax=plt.subplots()
    imagepath = [s for s in maskpathnames[dayID] if s.__contains__(imagestr)][0]
    points, simplices, alpha_shape, _, _=find_simplices(t)

    def rgb(a, b, c):
        return (a/255, b/255, c/255)
    def rgba(a, b, c, d):
        return (a/255, b/255, c/255, d)
    fibcol='white' #rgb(10, 82, 129)
    bgcol=rgba(0, 0, 0, 0.1)
    linecol=rgba(2, 86, 214, 0.72)
    centcol=rgba(255, 0, 0, 1)
    hullcolour=rgba(0, 0, 0, 0.2)

    cmap, linecol, centcol = colormap()
    im_matrix=plt.imread(imagepath)
    ax.imshow(im_matrix, cmap=cmap, extent=(0, nx*pxsize_arr[dayID], 0, ny*pxsize_arr[dayID]),origin='lower')
    # ax.add_patch(PolygonPatch(alpha_shape, color=hullcolour, alpha=0.2))
    ax.set_title(f'Day {t}')

    ax.triplot(points[:,0], points[:,1], simplices, '-' , color=linecol,lw=1.5)
    ax.plot(points[:,0], points[:,1], 'o', color=centcol, ms=3)
    ax.grid(visible=None)
    ax.set_xlabel('nm')  ;  ax.set_ylabel('nm')
    plt.savefig(dir_out+f'delaunay/delaunay_{t}', dpi=100)
    if atom:
        plt.show()
[display_Delaunay_mesh(ii) for ii in days]
def angle_Cosinerule(a, b, c):
    """
    Cosine Rule - returns angle in degrees
    """
    return np.arccos((c**2 - b**2 - a**2)/(-2.0 * a * b))  * (180/np.pi)
def lengths_and_angles(t):
    """
    Distances and angles generated from Delaunay mesh for a particular TEM image
    Also lengths divided by mean MFD of fibrils at nodes joined by edge of length L.
    These should all be given in nm.
    """
    points, simplices, _, _, mfds=find_simplices(t)

    simplices_coords=points[simplices] #list of the coords of the matching triangles
    simplices_MFDs=mfds[simplices]
    lengths=[];angles=[]; length_over_mfd=[]

    for tID in range(len(simplices_coords)):#step through the simplices and measure distances and angles
        a=np.linalg.norm(simplices_coords[tID, 0]-simplices_coords[tID, 1])
        a_bar= a/(np.mean(simplices_MFDs[tID, 0:2]))
        b=np.linalg.norm(simplices_coords[tID, 1]-simplices_coords[tID, 2])
        b_bar= b/(np.mean(simplices_MFDs[tID, 1:]))
        c=np.linalg.norm(simplices_coords[tID, 2]-simplices_coords[tID, 0])
        c_bar= c/(np.mean(simplices_MFDs[tID, 0::2]))
        angA = angle_Cosinerule(a,b,c)
        angB = angle_Cosinerule(b,c,a)
        angC = angle_Cosinerule(c,a,b)
        lengths.extend([a,b,c])
        angles.extend([angA,angB,angC])
        length_over_mfd.extend([a_bar,b_bar,c_bar])
    return lengths, angles, length_over_mfd
def delaunay_plots():
    l_lis=[]; ang_lis=[]; lbar_lis=[]

    for t in days:
        lengths, angles, length_over_mfd=lengths_and_angles(t)
        l_lis.append(lengths); ang_lis.append(angles); lbar_lis.append(length_over_mfd)

    fig, ax=plt.subplots(figsize=(15, 8))
    ax.set_xticks(np.arange(1, N+1))
    ax.set_xticklabels(days)
    ax.set_xlabel('Day postnatal')
    ax.set_ylabel('Centre-to-centre distance (nm)')
    ax.violinplot(l_lis, showmeans=True, widths=0.9)
    plt.savefig(dir_out+'nearestneighbour_L')
    if atom:
        plt.show()


    fig1, ax1=plt.subplots(figsize=(15, 8))
    ax1.set_xticks(np.arange(1, N+1))
    ax1.set_xticklabels(days)
    ax1.set_xlabel('Day postnatal')
    ax1.set_ylabel('Centre-to-centre distance (relative to fibril size)')
    ax1.violinplot([lengths_and_angles(t)[2] for t in days], showmeans=True, widths=0.9)
    plt.savefig(dir_out+'nearestneighbour_LoverD')
    if atom:
        plt.show()



    fig2, ax2=plt.subplots(figsize=(15, 8))
    ax2.set_xticks(np.arange(1, N+1))
    ax2.set_xticklabels(days)
    ax2.set_xlabel('Day postnatal')
    ax2.set_ylabel('Angles between nearest neighbours ($\degree$)')
    ax2.violinplot(ang_lis, showmeans=True, widths=0.9)
    plt.savefig(dir_out+'nearestneighbour_theta')
    if atom:
        plt.show()
delaunay_plots()

#%% GAP SIZE

def calculate_gap_sizes():
    gaps=[]; gapfracs=[]
    for t in days:
        dayID=find_timepoint_index(t)
        imagestr=import_selected_images().iloc[dayID].imageID.item()
        _, simplices, alpha_shape, points_px,_=find_simplices(t)
        imagepath = [s for s in maskpathnames[dayID] if s.__contains__(imagestr)][0]
        points_px=np.round(points_px).astype(int)
        points_px=np.vstack((points_px[:,1], points_px[:,0])).T #RC CONVERSION
        im_matrix=(plt.imread(imagepath)/65535).astype(int)
        pairs=np.vstack(np.array([list(itertools.combinations(i, 2)) for i in simplices]))
        npairs=pairs.shape[0]
        gap_lens=np.zeros(npairs)
        gap_frac=np.zeros(npairs)
        for j in range(npairs):
            L=np.array(line(*np.reshape(np.array(points_px[pairs[j]], dtype='int'), 4))).T#Line joining two points
            linelength_px=np.linalg.norm((L[0]- L[-1]))
            npoints_online=L.shape[0]
            values_on_line=np.array([im_matrix[coord[0], coord[1]] for coord in L ])
            fraction_ofline_gap=(npoints_online-np.count_nonzero(values_on_line))/npoints_online
            linelength_px=np.linalg.norm((L[0]-L[-1]))
            gaplen_px=linelength_px*fraction_ofline_gap
            gap_lens[j]=gaplen_px
            gap_frac[j]=fraction_ofline_gap
        gaps.append(gap_lens*pxsize_arr[dayID])
        gapfracs.append(gap_frac)
    return gaps, gapfracs
def plot_gap_sizes():
    fig, ax=plt.subplots(figsize=(15, 8))

    ax.violinplot(calculate_gap_sizes()[0], showmeans=True, widths=0.9)
    ax.set_xticks(np.arange(1, N+1))
    ax.set_xticklabels(days)
    ax.set_xlabel('Day postnatal')
    ax.set_ylabel('Gap size (nm)')
    plt.savefig(dir_out+'gapsizes')
    if atom:

        plt.show()


    fig, ax=plt.subplots(figsize=(15, 8))

    ax.violinplot(calculate_gap_sizes()[1], showmeans=True, widths=0.9)
    ax.set_xticks(np.arange(1, N+1))
    ax.set_xticklabels(days)
    ax.set_xlabel('Day postnatal')
    ax.set_ylabel('Gapsize relative to fibrils')
    plt.savefig(dir_out+'gapfrac')
    if atom:

        plt.show()
plot_gap_sizes()

#%%==============================Mouse growth curve=================================================
def mousecurve():
    mouse_df=pd.read_csv('./mousegrowth.csv')
    mouse_df.shape
    means_mass=np.array([np.mean(mouse_df[['w1', 'w2', 'w3', 'w4']].iloc[i]) for i in range(mouse_df.shape[0])]) #grams
    std_mass=np.array([np.std(mouse_df[['w1', 'w2', 'w3', 'w4']].iloc[i]) for i in range(mouse_df.shape[0])])
    density=1 #g/cm^3

    mean_vol=means_mass /density #g/cm^3
    stdvol=std_mass/density
    fig, ax=plt.subplots()
    pars, cov=curve_fit(plateau, mouse_df.Day, means_mass)
    ax.set_xlabel('Day postnatal') ; ax.set_ylabel('Mass')
    textstr=f'$W$ = {pars[0]:.1f} g \n$T$ = {pars[1]:.1f} days \n$W_0$ = {pars[2]:.1f} days'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)    # these are matplotlib.patch.Patch properties
    ax.text(0.75, 0.25, textstr, transform=ax.transAxes, fontsize=16,verticalalignment='top', bbox=props)
    ax.errorbar(mouse_df.Day,mean_vol,stdvol, fmt='None',color='k', capsize=6, capthick=1.5,zorder=1)
    ax.scatter(mouse_df.Day,mean_vol)
    ax.set_xticks(ticks=np.arange(0, 57, 7))

    ax.plot(lin_days, plateau(lin_days, *pars), 'r')
    plt.savefig(dir_out+'mouse_growth_g')
    if atom:
        plt.show()
    return pars
mousepars=mousecurve()

#%%=================Number Density ==============================
def Decay(x, a,M, b):
    """
    Decay function. M= final value, a=how fast it gets there
    """
    return M+b*np.exp(-x/a)
def plot_numberdensity(relative=True):
    numberdensity_um=[]

    for i in range (N):
        firstIm=np.min(fibril_dfs[i].ImageNumber)
        lastIm=np.max(fibril_dfs[i].ImageNumber)
        cell_coverage=measure_cell_coverage(days[i])
        ndens_array=np.zeros(lastIm-firstIm) ; j=0
        for im in range(firstIm, lastIm):
            nfibs=np.max(fibril_dfs[i][fibril_dfs[i].ImageNumber==im].ObjectNumber)
            scale=plateau(days[i], 1, mousepars[1], 0)**(2/3) if relative else 1
            area=pxsize_arr[i]**2*(nx*ny-cell_coverage[j])*scale

            ndens_array[j]=10**6*nfibs/area
            j+=1
        numberdensity_um.append(ndens_array)

    numberdensity_means=[np.mean(numberdensity_um[i]) for i in range(N)]
    numberdensity_std=[np.std(numberdensity_um[i]) for i in range(N)]
    pars, cov=curve_fit(Decay, days, numberdensity_means)

    fig, ax=plt.subplots()
    ax.set_xlabel('Day postnatal')
    ylabel_='Number density (relative)' if relative else 'Number density ($\mu$m$^{-2}$)'
    ax.set_ylabel(ylabel_)
    y=Decay(lin_days, *pars)/pars[-1] if relative else Decay(lin_days, *pars)
    ax.plot(lin_days, y, '-r')
    textstr=f'$T$ = {pars[0]:.1f} days'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)    # these are matplotlib.patch.Patch properties
    ax.text(0.75, 0.25, textstr, transform=ax.transAxes, fontsize=16,verticalalignment='top', bbox=props)
    ax.set_xticks(ticks=np.arange(0, 57, 7))
    y=numberdensity_means/pars[-1] if relative else numberdensity_means
    ax.scatter(days,y)
    yerr=numberdensity_std/pars[-1] if relative else numberdensity_std
    ax.errorbar(days,y,yerr, fmt='None',color='k', capsize=6, capthick=1.5,zorder=1)
    str='relative'  if relative else 'um-2'
    plt.savefig(dir_out+'numberdensity_'+str)
    if atom:
        plt.show()

plot_numberdensity()
#%%
display_Delaunay_mesh(14)

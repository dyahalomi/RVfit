"""
Analyze the results of an MCMC run.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from funcs import RV_model
from funcs import readObservations
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import csv

# current parameters for the model and their order

#labels HIRES or TRES
labels = ['$P$ (days)', '$t_{tran}$ (days)', '$\sqrt{e} cos\omega$', '$\sqrt{e} sin\omega$',
          '$K_1$ (km/s)', '$\gamma$ (km/s)']

numSpec = 7
for i in range(1, numSpec):
    labels.append('$\gamma_{os,' + str(i) + '} (km/s)$')

for i in range(0, numSpec):
    labels.append('$\sigma^2_{j,' + str(i+1) +'} (km/s)^2$')


# the file with the MCMC chain results
infile = './HD102509/chain_100000.txt'


# after the burn in, only use every thin amount for speed
nthin = 1

# output the median and 1-sigma error results to a TeX file
# use None if not desired
texout = './HD102509/chain_100000.tex'

foldername = './HD102509/'
RVfigname = 'RVfit_100000.jpg'
cornerFigname = 'corner_100000.jpg'
chainFigname = 'chainPlot_100000.jpg'



# iteration where burn-in stops
burnin = 4000
# make the triangle plot
maketriangle = True

#########################

nparams = len(labels)

x = np.loadtxt(infile)
print 'File loaded'

# split the metadata from the chain results
iteration = x[:, 0]
walkers = x[:, 1]
uwalkers = np.unique(walkers)
loglike = x[:, 2]
x = x[:, 3:]

# thin the file if we want to speed things up
thin = np.arange(0, iteration.max(), nthin)
good = np.in1d(iteration, thin)
x = x[good, :]
iteration = iteration[good]
walkers = walkers[good]
loglike = loglike[good]

# plot the value of each chain for each parameter as well as its log likelihood
plt.figure(figsize = (24, 18))
plt.clf()
for ii in np.arange(nparams+1):
    # use 3 columns of plots
    ax = plt.subplot(np.ceil((nparams+1)/3.), 3, ii+1)
    for jj in uwalkers:
        this = np.where(walkers == jj)[0]
        if ii < nparams:
            # if this chain is really long, cut down on plotting time by only
            # plotting every tenth element
            if len(iteration[this]) > 5000:
                plt.plot(iteration[this][::10],
                         x[this, ii].reshape((-1,))[::10])
            else:
                plt.plot(iteration[this], x[this, ii].reshape((-1,)))
        # plot the likelihood
        else:
            if len(iteration[this]) > 5000:
                plt.plot(iteration[this][::10], loglike[this][::10])
            else:
                plt.plot(iteration[this], loglike[this])
    # show the burnin location
    plt.plot([burnin, burnin], plt.ylim(), lw=2)
    # add the labels
    if ii < nparams:
        plt.ylabel(labels[ii])
    else:
        plt.ylabel('Log Likelihood')
        plt.xlabel('Iterations')
    ax.ticklabel_format(useOffset=False)

plt.savefig(foldername + chainFigname)

# now remove the burnin phase
pastburn = np.where(iteration > burnin)[0]
iteration = iteration[pastburn]
walkers = walkers[pastburn]
loglike = loglike[pastburn]
x = x[pastburn, :]

# sort the results by likelihood for the triangle plot
lsort = np.argsort(loglike)
lsort = lsort[::-1]
iteration = iteration[lsort]
walkers = walkers[lsort]
loglike = loglike[lsort]
x = x[lsort, :]

        

if maketriangle:
    plt.figure(figsize = (18, 18))
    plt.clf()
    # set unrealistic default mins and maxes
    maxes = np.zeros(len(x[0, :])) - 9e9
    mins = np.zeros(len(x[0, :])) + 9e9
    nbins = 50
    # go through each combination of parameters
    for jj in np.arange(len(x[0, :])):
        for kk in np.arange(len(x[0, :])):
            # only handle each combination once
            if kk < jj:
                # pick the right subplot
                ax = plt.subplot(len(x[0, :]), len(x[0, :]),
                                 jj * len(x[0, :]) + kk + 1)
                # 3, 2, and 1 sigma levels
                sigmas = np.array([0.9973002, 0.9544997, 0.6826895])
                # put each sample into 2D bins
                hist2d, xedge, yedge = np.histogram2d(x[:, jj], x[:, kk],
                                                      bins=[nbins, nbins],
                                                      normed=False)
                # convert the bins to frequency
                hist2d /= len(x[:, jj])
                flat = hist2d.flatten()
                # get descending bin frequency
                fargs = flat.argsort()[::-1]
                flat = flat[fargs]
                # cumulative fraction up to each bin
                cums = np.cumsum(flat)
                levels = []
                # figure out where each sigma cutoff bin is
                for ii in np.arange(len(sigmas)):
                        above = np.where(cums > sigmas[ii])[0][0]
                        levels.append(flat[above])
                levels.append(1.)
                # figure out the min and max range needed for this plot
                # then see if this is beyond the range of previous plots.
                # this is necessary so that we can have a common axis
                # range for each row/column
                above = np.where(hist2d > levels[0])
                thismin = xedge[above[0]].min()
                if thismin < mins[jj]:
                    mins[jj] = thismin
                thismax = xedge[above[0]].max()
                if thismax > maxes[jj]:
                    maxes[jj] = thismax
                thismin = yedge[above[1]].min()
                if thismin < mins[kk]:
                    mins[kk] = thismin
                thismax = yedge[above[1]].max()
                if thismax > maxes[kk]:
                    maxes[kk] = thismax
                # make the contour plot for these two parameters
                plt.contourf(yedge[1:]-np.diff(yedge)/2.,
                             xedge[1:]-np.diff(xedge)/2., hist2d,
                             levels=levels,
                             colors=('k', '#444444', '#888888'))
            # plot the distribution of each parameter
            if jj == kk:
                ax = plt.subplot(len(x[0, :]), len(x[0, :]),
                                 jj*len(x[0, :]) + kk + 1)
                plt.hist(x[:, jj], bins=nbins, facecolor='k')

    # allow for some empty space on the sides
    diffs = maxes - mins
    mins -= 0.05*diffs
    maxes += 0.05*diffs
    # go back through each figure and clean it up
    for jj in np.arange(len(x[0, :])):
        for kk in np.arange(len(x[0, :])):
            if kk < jj or jj == kk:
                ax = plt.subplot(len(x[0, :]), len(x[0, :]),
                                 jj*len(x[0, :]) + kk + 1)
                # set the proper limits
                if kk < jj:
                    ax.set_ylim(mins[jj], maxes[jj])
                ax.set_xlim(mins[kk], maxes[kk])
                # make sure tick labels don't overlap between subplots
                ax.yaxis.set_major_locator(plticker.MaxNLocator(nbins=4,
                                                                prune='both'))
                # only show tick labels on the edges
                if kk != 0 or jj == 0:
                    ax.set_yticklabels([])
                else:
                    # tweak the formatting
                    plt.ylabel(labels[jj])
                    locs, labs = plt.yticks()
                    plt.setp(labs, rotation=0, va='center')
                    yformatter = plticker.ScalarFormatter(useOffset=False)
                    ax.yaxis.set_major_formatter(yformatter)
                # do the same with the x-axis ticks
                ax.xaxis.set_major_locator(plticker.MaxNLocator(nbins=4,
                                                                prune='both'))
                if jj != len(x[0, :])-1:
                    ax.set_xticklabels([])
                else:
                    plt.xlabel(labels[kk])
                    locs, labs = plt.xticks()
                    plt.setp(labs, rotation=90, ha='center')
                    yformatter = plticker.ScalarFormatter(useOffset=False)
                    ax.xaxis.set_major_formatter(yformatter)
    # remove the space between plots
    plt.subplots_adjust(hspace=0.0, wspace=0.0)


# the best, median, and standard deviation of the input parameters
# used to feed back to model_funcs for initrange, and plotting the best fit
# model for publication figures in mcmc_run
best = x[0, :]
meds = np.median(x, axis=0)
devs = np.std(x, axis=0)
print 'Best model parameters: '
print best

print 'Median model parameters: '
print meds

# put the MCMC results into a TeX table
if texout is not None:

    # calculate eccentricity and add it to the list of parameters
    e = (np.sqrt(x[:, 2]**2. + x[:, 3]**2.)).reshape((len(x[:, 0]), 1))
    x = np.concatenate((x, e), axis=1)
    labels.append('$e$')

    # add omega to the list
    omega = np.arctan2(x[:, 3], x[:, 2]).reshape((len(x[:, 0]), 1))*180./np.pi
    x = np.concatenate((x, omega), axis=1)
    labels.append('$\omega$ (deg)')

    # what are the median and 1-sigma limits of each parameter we care about
    stds = [15.87, 50., 84.13]
    neg1, med, plus1 = np.percentile(x, stds, axis=0)



    # get ready to write them out
    ofile = open(texout, 'w')
    ofile.write('\\documentclass{article}\n\\usepackage{graphicx}\n\n\\begin{document}\n\n')
    ofile.write('\\begin{table}\n\\centering\n')
    ofile.write('\\begin{tabular}{| c | c |}\n\\hline\n')

    # what decimal place the error bar is at in each direction
    sigfigslow = np.floor(np.log10(np.abs(plus1-med)))
    sigfigshigh = np.floor(np.log10(np.abs(med-neg1)))
    sigfigs = sigfigslow * 1
    # take the smallest of the two sides of the error bar
    lower = np.where(sigfigshigh < sigfigs)[0]
    sigfigs[lower] = sigfigshigh[lower]
    # go one digit farther
    sigfigs -= 1
    # switch from powers of ten to number of decimal places
    sigfigs *= -1.
    sigfigs = sigfigs.astype(int)

    # go through each parameter
    for ii in np.arange(len(labels)):
      
        # if we're rounding to certain decimal places, do it
        if sigfigs[ii] >= 0:
            val = '%.'+str(sigfigs[ii])+'f'
        else:
            val = '%.0f'
        # do the rounding to proper decimal place and write the result
        ostr = labels[ii]+' & $'
        ostr += str(val % np.around(med[ii], decimals=sigfigs[ii]))
        ostr += '^{+' + str(val % np.around(plus1[ii]-med[ii],
                                            decimals=sigfigs[ii]))
        ostr += '}_{-' + str(val % np.around(med[ii]-neg1[ii],
                                             decimals=sigfigs[ii]))
        ostr += '}$ \\\\\n\\hline\n'
        ofile.write(ostr)

    ofile.write('\\end{tabular}\n\\end{table}\n\n')


    ofile.write('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\textwidth]{' + str(RVfigname) + '}\n\\caption{RV fit to median MCMC parameters}\n\\end{figure}\n\n')
    ofile.write('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\textwidth]{' + str(cornerFigname) + '}\n\\caption{Corner plot for MCMC model}\n\\end{figure}\n\n')
    ofile.write('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\textwidth]{' + str(chainFigname) + '}\n\\caption{MCMC chains for all 50 walkers. Green line is burnout}\n\\end{figure}\n\n')

    ofile.write('\\end{document}')
    ofile.close()


#save corner plot
plt.savefig(foldername + cornerFigname)
plt.show()

def plot_RV(p, t, rv,rvErr):
    '''
    Plot the RV data against RV model

    '''
    # Define all parameters except gammaOffsets and jitters
    (period, ttran, ecosomega, esinomega, K, gamma) = p[0:6]

    # Define a list of gamma offset parameters
    gammaOffset = p[6:6+numSpec-1]

    # Define a list of jitter squared parameters
    jitterSqrd = p[6+numSpec-1:len(p)]

    colors = [
    '#800000', '#9A5324', '#808000', '#469990', '#000075', '#e6194B', '#f58231', 
    '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4', '#f032e6'
    ]

    plt.figure(figsize=(15,10))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], wspace=0.03)
    gs.update(hspace=0.)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.errorbar(t[0], rv[0], yerr = np.sqrt(rvErr[0]**2. + jitterSqrd[0]), fmt = 'o', color = colors[0],  markersize = 10, label = "Spectra 1")
    for ii in range(1, len(t)):
        ax0.errorbar(t[ii], rv[ii] + gammaOffset[ii-1], yerr=np.sqrt(rvErr[ii]**2. + jitterSqrd[ii]), fmt='o', color = colors[ii],  markersize = 10, label = "Spectra " + str(ii+1))


    t_plot = np.arange(15000, 60000)
    model = RV_model(t_plot, p)
    ax0.plot(t_plot, model, color = 'k')


    rv_models = []
    for ii in range(0, len(t)):
        rv_models.append(RV_model(t[ii], p))
    
    ax1.plot([15000, 60000], [0., 0.], color = 'k')

    ax1.errorbar(t[0], rv[0] - rv_models[0], yerr = np.sqrt(rvErr[0]**2. + jitterSqrd[0]), fmt = 'o', markersize = 10,  color = colors[0])
    for ii in range(1, len(t)):
        ax1.errorbar(t[ii], rv[ii] + gammaOffset[ii-1] - rv_models[ii], yerr = np.sqrt(rvErr[ii]**2. + jitterSqrd[ii]), fmt = 'o',  markersize = 10, color = colors[ii])

   
    ax1.set_xlabel("Time (units?)", fontsize = 18)
    ax0.set_ylabel("Radial Velocity (km/s)", fontsize = 18)
    ax1.set_ylabel("Residuals (km/s)", fontsize = 18)
    yticks = ax0.yaxis.get_major_ticks()
    xticks = ax0.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)
    yticks[0].label1.set_visible(False)
    ax0.legend(numpoints = 1, loc = 2, fontsize = 18)
    plt.xlim(15000,60000)

    plt.savefig(foldername + 'unfolded_' + RVfigname)
    plt.show()

def plot_foldedRV(p, t, rv, rvErr):
    '''
    Plot the RV data against RV model folded

    '''
    # Define all parameters except gammaOffsets and jitters
    (period, ttran, ecosomega, esinomega, K, gamma) = p[0:6]

    # Define a list of gamma offset parameters
    gammaOffset = p[6:6+numSpec-1]

    # Define a list of jitter squared parameters
    jitterSqrd = p[6+numSpec-1:len(p)]

    colors = [
    '#800000', '#9A5324', '#808000', '#469990', '#000075', '#e6194B', '#f58231', 
    '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4', '#f032e6'
    ]

    plt.figure(figsize=(15,10))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], wspace=0.03)
    gs.update(hspace=0.)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    phase_rv_1 = ((t[0] - p[1]) % p[0]) / p[0]
    ax0.errorbar(phase_rv_1, rv[0], yerr = np.sqrt(rvErr[0]**2. + jitterSqrd[0]), fmt = 'o', color = colors[0],  markersize = 10, label = "Spectra 1")

    phase_rv = [phase_rv_1]
    for ii in range(1, len(t)):
        phase_rv_i = ((t[ii]-p[1]) % p[0])/p[0]
        phase_rv.append(phase_rv_i)
        ax0.errorbar(phase_rv_i, rv[ii] + gammaOffset[ii-1], yerr=np.sqrt(rvErr[ii]**2. + jitterSqrd[ii]), fmt='o', color = colors[ii],  markersize = 10, label = "Spectra " + str(ii+1))


    
    
    tMod = np.arange(p[0], p[0] + p[1])
    model = RV_model(tMod, p)
    phase = ((tMod-p[1]) % p[0]) / p[0]
    lsort = np.argsort(phase)
    ax0.plot(phase[lsort], model[lsort], color = 'k')


    rv_models = []
    for ii in range(0, len(t)):
        rv_models.append(RV_model(t[ii], p))

    
    ax1.plot([0., 1.], [0., 0.], color = 'k')
    ax1.errorbar(phase_rv[0], rv[0] - rv_models[0], yerr = np.sqrt(rvErr[0]**2. + jitterSqrd[0]), fmt = 'o', markersize = 10,  color = colors[0])

    for ii in range(1, len(t)):
        ax1.errorbar(phase_rv[ii], rv[ii] + gammaOffset[ii-1] - rv_models[ii], yerr = np.sqrt(rvErr[ii]**2. + jitterSqrd[ii]), fmt = 'o',  markersize = 10, color = colors[ii])

    ax1.set_xlabel("Phase", fontsize = 18)
    ax0.set_ylabel("Radial Velocity (km/s)", fontsize = 18)
    ax1.set_ylabel("Residuals (km/s)", fontsize = 18)
    yticks = ax0.yaxis.get_major_ticks()
    xticks = ax0.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)
    yticks[0].label1.set_visible(False)
    ax0.legend(numpoints = 1, loc = 2, fontsize = 18)


    plt.savefig(foldername + RVfigname)
    plt.show()





def get_RMS_residuals_2obs(p, t_RV1, RV1, RVerr1, t_RV2, RV2, RVerr2):
    '''
    p: input parameters
    the rest are observations
    '''
    predicted_RV1 = RV_model_2obs(t_RV1, p)
    predicted_RV2 = RV_model_2obs(t_RV2, p)

    n = len(t_RV1) + len(t_RV2)

    rms = np.sqrt( (np.sum((RV1 - predicted_RV1)**2) + np.sum((RV2+gammaOffset[ii-1] - predicted_RV2) **2)) / n)

    print RV1 - predicted_RV1
    print RV2 + gammaOffset[ii-1] - predicted_RV2
    mean1 = np.mean(RV1 - predicted_RV1)
    mean2 = np.mean(RV2 + gammaOffset[ii-1] - predicted_RV2)
    mean = np.mean(np.array([mean1, mean2]))

    return rms, mean


t, rv, rvErr = readObservations('./HD102509/HD102509.orb', True)

print plot_RV(meds, t, rv, rvErr)

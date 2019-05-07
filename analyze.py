"""
Analyze the results of an MCMC run.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from funcs import *
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import csv

# current parameters for the model and their order

#labels HIRES or TRES
labels = ['$P$ (days)', '$t_{tran}$ (days)', '$e cos\omega$', '$e sin\omega$',
          '$K_1$ (km/s)']

#WDS04342
numSpec = 4

#HD102509
#numSpec = 7


for i in range(0, numSpec):
    labels.append('$\gamma_{' + str(i+1) + '} (km/s)$')

labels.append('$\sigma^2_{j} (km/s)^2$')

for i in range(0, numSpec):
    labels.append('$M_{error, ' + str(i+1) + '} $')


# the file with the MCMC chain results
infile = './WDS04342/chain_100000_errorMult.txt'


# after the burn in, only use every thin amount for speed
nthin = 1

# output the median and 1-sigma error results to a TeX file
# use None if not desired
texout = './WDS04342/chain_100000_errorMult.tex'

foldername = './WDS04342/'
RVfigname_meds = 'RVfit_meds_100000_errorMult.jpg'
RVfigname_best = 'RVfit_best_100000_errorMult.jpg'
cornerFigname = 'corner_100000_errorMult.jpg'
chainFigname = 'chainPlot_100000_errorMult.jpg'


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


def plot_RV(p, t, rv,rvErr):
    '''
    Plot the RV data against RV model

    '''
    # Define all parameters except gamma and jitters
    (period, ttran, ecosomega, esinomega, K) = p[0:5]

    # Define a list of gamma parameters
    gammas = p[5:5+numSpec]

    # Define jitter squared parameter
    jitterSqrd = p[5+numSpec]

    #Define list of error multipliers
    errorMult = p[5+numSpec+1:len(p)]


    gammaOffsets = []
    for ii in range(0, len(gammas)):
        gammaOffsets.append(gammas[0] - gammas[ii])


    colors = [
    '#800000', '#9A5324', '#808000', '#469990', '#000075', '#e6194B', '#f58231', 
    '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4', '#f032e6'
    ]

    plt.figure(figsize=(15,10))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], wspace=0.03)
    gs.update(hspace=0.)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    #ax0.errorbar(t[0], rv[0], yerr = np.sqrt(rvErr[0]**2. + jitterSqrd[0]), fmt = 'o', color = colors[0],  markersize = 10, label = "Spectra 1")
    for ii in range(0, len(t)):
        ax0.errorbar(t[ii], rv[ii] + gammaOffsets[ii], yerr=(errorMult[ii] * np.sqrt(rvErr[ii]**2. + jitterSqrd)), fmt='o', color = colors[ii],  markersize = 10, label = "Spectra " + str(ii+1))


    t_plot = np.arange(10000, 60000)
    model = RV_model(t_plot, period, ttran, ecosomega, esinomega, K, gammas[0])
    ax0.plot(t_plot, model, color = 'k')


    rv_models = []
    for ii in range(0, len(t)):
        rv_models.append(RV_model(t[ii], period, ttran, ecosomega, esinomega, K, gammas[0]))
    
    ax1.plot([10000, 60000], [0., 0.], color = 'k')

    #ax1.errorbar(t[0], rv[0] - rv_models[0], yerr = np.sqrt(rvErr[0]**2. + jitterSqrd[0]), fmt = 'o', markersize = 10,  color = colors[0])
    for ii in range(0, len(t)):
        ax1.errorbar(t[ii], rv[ii] - rv_models[ii] + gammaOffsets[ii], yerr=(errorMult[ii] * np.sqrt(rvErr[ii]**2. + jitterSqrd)), fmt = 'o',  markersize = 10, color = colors[ii])

   
    ax1.set_xlabel("Time (units?)", fontsize = 18)
    ax0.set_ylabel("Radial Velocity (km/s)", fontsize = 18)
    ax1.set_ylabel("Residuals (km/s)", fontsize = 18)
    yticks = ax0.yaxis.get_major_ticks()
    xticks = ax0.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)
    yticks[0].label1.set_visible(False)
    ax0.legend(numpoints = 1, loc = 2, fontsize = 18)
    plt.xlim(10000,60000)

    plt.savefig(foldername + 'unfolded_' + RVfigname)
    plt.show()


def plot_foldedRV(p, t, rv, rvErr, filename):
    '''
    Plot the RV data against RV model folded

    '''
    # Define all parameters except gamma and jitters
    (period, ttran, ecosomega, esinomega, K) = p[0:5]

    # Define a list of gamma parameters
    gammas = p[5:5+numSpec]

    # Define jitter squared parameter
    jitterSqrd = p[5+numSpec]

    #Define list of error multipliers
    errorMult = p[5+numSpec+1:len(p)]

    colors = [
    '#800000', '#9A5324', '#808000', '#469990', '#000075', '#e6194B', '#f58231', 
    '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4', '#f032e6'
    ]

    plt.figure(figsize=(15,10))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], wspace=0.03)
    gs.update(hspace=0.)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    gammaOffsets = []
    for ii in range(0, len(gammas)):
        gammaOffsets.append(gammas[0] - gammas[ii])

    phase_rv = []
    for ii in range(0, len(t)):
        phase_rv_i = ((t[ii]-p[1]) % p[0])/p[0]
        phase_rv.append(phase_rv_i)
        ax0.errorbar(phase_rv_i, rv[ii] + gammaOffsets[ii], yerr=(errorMult[ii] * np.sqrt(rvErr[ii]**2. + jitterSqrd)), fmt='o', color = colors[ii],  markersize = 10, label = "Spectra " + str(ii+1))


    
    
    tMod = np.arange(p[0], p[0] + p[1])
    model = RV_model(tMod, period, ttran, ecosomega, esinomega, K, gammas[0])
    phase = ((tMod-p[1]) % p[0]) / p[0]
    lsort = np.argsort(phase)
    ax0.plot(phase[lsort], model[lsort], color = 'k')


    rv_models = []
    for ii in range(0, len(t)):
        rv_models.append(RV_model(t[ii], period, ttran, ecosomega, esinomega, K, gammas[0]))

    
    ax1.plot([0., 1.], [0., 0.], color = 'k')
    #ax1.errorbar(phase_rv[0], rv[0] - rv_models[0], yerr = np.sqrt(rvErr[0]**2. + jitterSqrd[0]), fmt = 'o', markersize = 10,  color = colors[0])

    for ii in range(0, len(t)):
        ax1.errorbar(phase_rv[ii], rv[ii] - rv_models[ii] + gammaOffsets[ii], yerr=(errorMult[ii] * np.sqrt(rvErr[ii]**2. + jitterSqrd)), fmt = 'o',  markersize = 10, color = colors[ii])

    ax1.set_xlabel("Phase", fontsize = 18)
    ax0.set_ylabel("Radial Velocity (km/s)", fontsize = 18)
    ax1.set_ylabel("Residuals (km/s)", fontsize = 18)
    yticks = ax0.yaxis.get_major_ticks()
    xticks = ax0.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)
    yticks[0].label1.set_visible(False)
    ax0.legend(numpoints = 1, loc = 2, fontsize = 18)


    plt.savefig(filename)
    plt.show()





def get_RMS_residuals(p, t, rv, rvErr):
    '''
    p: input parameters
    the rest are observations
    '''
    # Define all parameters except gamma and jitters
    (period, ttran, ecosomega, esinomega, K) = p[0:5]

    # Define a list of gamma parameters
    gammas = p[5:5+numSpec]

    # Define jitter squared parameter
    jitterSqrd = p[5+numSpec]

    #Define list of error multipliers
    errorMult = p[5+numSpec+1:len(p)]


    rv_models = []
    for ii in range(0, len(t)):
        rv_models.append(RV_model(t[ii], period, ttran, ecosomega, esinomega, K, gammas[ii]))

    n = 0
    for ii in range(0, len(t)):
        n += len(t[ii])

    rms = 0
    for ii in range(0, len(t)):
        print ii, np.sum((rv[ii] - rv_models[ii])**2) / n
        rms += ( np.sum((rv[ii] - rv_models[ii])**2) / n)

    rms = np.sqrt(rms)


    return rms





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

    #save corner plot
    plt.savefig(foldername + cornerFigname)
    plt.show()


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


#t, rv, rvErr = readObservations('./HD102509/HD102509.orb', True)
t, rv, rvErr = readObservations('./WDS04342/L79.txt', True)
numObs = 0
for ii in range(0, len(t)):
    numObs += len(t[ii])


print plot_foldedRV(meds, t, rv, rvErr, foldername + RVfigname_meds)
print plot_foldedRV(best, t, rv, rvErr, foldername + RVfigname_best)

rms_meds = get_RMS_residuals(meds, t, rv, rvErr)
print 'rms meds', rms_meds

rms_best = get_RMS_residuals(best, t, rv, rvErr)
print 'rms best', rms_best

redChisQ_meds = (loglikelihood(
    meds, t, rv, rvErr, chisQ=True) /
    (numObs - len(meds)))

print 'Reduced chi-square medians: ',  redChisQ_meds


redChisQ_best = (loglikelihood(
    best, t, rv, rvErr, chisQ=True) /
    (numObs - len(best)))

print 'Reduced chi-square best: ',  redChisQ_best


# put the MCMC results into a TeX table
if texout is not None:
    best_out = best.copy()
    best_out = list(best_out)

    # calculate eccentricity and add it to the list of parameters
    e = (np.sqrt(x[:, 2]**2. + x[:, 3]**2.)).reshape((len(x[:, 0]), 1))
    e_best = np.sqrt(best[2]**2. + best[3]**2.)
    best_out.append(e_best)
    x = np.concatenate((x, e), axis=1)
    labels.append('$e$')

    # add omega to the list
    omega = np.arctan2(x[:, 3], x[:, 2]).reshape((len(x[:, 0]), 1))*180./np.pi
    omega_best = np.arctan2(best[3], best[2])*180./np.pi
    best_out.append(omega_best)
    x = np.concatenate((x, omega), axis=1)
    labels.append('$\omega$ (deg)')

    # what are the median and 1-sigma limits of each parameter we care about
    stds = [15.87, 50., 84.13]
    neg1, med, plus1 = np.percentile(x, stds, axis=0)



    # get ready to write them out
    ofile = open(texout, 'w')
    ofile.write('\\documentclass{article}\n\\usepackage{graphicx}\n\\usepackage[margin=1in]{geometry}\n\n\\begin{document}\n\n')
    ofile.write('\\begin{table}\n\\centering\n')
    ofile.write('\\caption{Median Reduced $\\chi^2$: ' + str(np.round(redChisQ_meds, decimals = 2)) + ' -- Maximum-Likelihood Reduced $\\chi^2$: ' + str(np.round(redChisQ_best, decimals = 2)) + '}\n')
    ofile.write('\\begin{tabular}{| c | c | c |}\n\\hline\n')

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
    ofile.write('Parameter & Median and $1 \sigma$ Values & Maximum-Likelihood \\\\\n\\hline\n')
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
        
        best_val = round(best_out[ii], sigfigs[ii])
        ostr += '}$ & $' + str(best_val)

        ostr += '$ \\\\\n\\hline\n'
        ofile.write(ostr)

    ofile.write('\\end{tabular}\n\\end{table}\n\n')


    ofile.write('\\clearpage\n\n')
    ofile.write('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=0.9\\textwidth]{' + str(RVfigname_meds) + '}\n\\caption{RV fit to median MCMC parameters. RMS residual velocity of ' + str(np.round(rms_meds, decimals = 2)) + ' $\\rm{km \\: s^{-1}}$.}\n\n')
    ofile.write('\\includegraphics[width=0.9\\textwidth]{' + str(RVfigname_best) + '}\n\\caption{RV fit to maximum-likelihood MCMC parameters. RMS residual velocity of ' + str(np.round(rms_best, decimals = 2)) + ' $\\rm{km \\: s^{-1}}$.}\n\\end{figure}\n\n\n')
    ofile.write('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\textwidth]{' + str(cornerFigname) + '}\n\\caption{Contour plots showing the $1 \\sigma$, $2 \\sigma$, and $3 \\sigma$ constraints on pairs of parameters for the MCMC model.}\n\\end{figure}\n\n\n')
    ofile.write('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\textwidth]{' + str(chainFigname) + '}\n\\caption{MCMC chains for all 50 walkers. Green line is burnout: ' + str(burnin) + ' steps.}\n\\end{figure}\n\n')

    ofile.write('\\end{document}')
    ofile.close()





#median parameters for HD102509 100,000 step run w gammas instead of gamma_os
p_med = [ 7.16902791e+01,  4.30725716e+04, -2.27864352e-04,  4.68399935e-04,
          3.00898058e+01,  1.86445675e+00,  1.77603777e+00,  1.28836824e+00,
          7.16327739e-01,  9.74341232e-01,  6.93817102e-01,  4.04686472e-01,
          3.96837830e-01,  2.78218700e-01,  4.31310021e-01,  1.22508258e-01,
          2.91819070e-02,  7.35546530e-02,  9.58132551e-02]

p_best = [7.16900862e+01, 4.30725734e+04, 6.53477837e-06, 5.06544786e-06,
 3.00994492e+01, 1.45330737e+00, 1.76800811e+00, 1.33881120e+00,
 6.05038833e-01, 9.09778590e-01, 7.29530164e-01, 3.78518888e-01,
 3.80209986e-01, 2.85164883e-02, 5.49199523e-01, 7.96691754e-02,
 2.45596315e-02, 5.14855994e-02, 7.00575041e-02]




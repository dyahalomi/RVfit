import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from funcs import *
import csv




def runMCMC(p, t, rv, rvErr, outfile, niter=10000, nwalkers=50):

	"""
	Run the MCMC Orbital fit to Spectroscopic RV Observations

	Input
	-----
	p : ndarray
		Orbital parameters. See RV model in funcs.py for order
	t, rv, rvErr : MxNdarray
		times, RV, and RV errors of the data.
		arranged as a list of lists
		len(array) = number of observing devices
	outfile : string
		name of output file where MCMC chain is stored
	niter : int, optional
        number of MCMC iterations to run. Default = 10,000
    nwalkers : int, optional
        number of MCMC walkers in modeling. Default = 50
		
	Returns
	------
	String stating "MCMC complete"

	(Outputs MCMC chain into file labeled whatever input into variable: outfile)


	"""

	ndim = len(p)
	nspec = len(t)

	#start walkers in a ball near the optimal solution
	startlocs = [p + initrange(p, nspec) * np.random.randn(ndim) for i in np.arange(nwalkers)]

	#run emcee MCMC code
	#run both data sets
	sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args = [t, rv, rvErr])

	#clear output file
	ofile = open(outfile, 'w')
	ofile.close()

	#run the MCMC...record parameters for every walker at every step
	for result in sampler.sample(startlocs, iterations = niter, storechain = False):
		pos = result[0]
		iternum = sampler.iterations
		ofile = open(outfile, 'a')

		#write iteration number, walker number, and log likelihood
		#and value of parameters for the step
		for walker in np.arange(pos.shape[0]):
			ofile.write('{0} {1} {2} {3}\n'.format(iternum, walker, str(result[1][walker]), " ".join([str(x) for x in pos[walker]])))

		ofile.close()


		#keep track of step number
		mod = iternum % 100
		if mod == 0:
			print iternum
			print pos[0]


	return "MCMC complete"




'''
#set first guess of parameters for modeling
#p = [period, ttran, ecosomega, esinomega, K,...]
p = [3677, 48667, 0.84, -0.26, 4.4, 42]

# p = [...gamma_1, gamma_2, gamma_3...]
for i in range(1, len(t)):	
	p.append(1)

# p = [...jitterSqrd1, jitterSqrd2, jitterSqrd3,...]
for i in range(0, len(t)):
	p.append(0.0)



#median parameters for HD102509 10,000 step run
p = [ 7.16902685e+01,  4.30725785e+04, -0.0003,  0.0004,
  	  30.094,  1.86, 1.86,  1.86,
  	  1.86,  1.86,  1.86,  1.86,
  	  2.11738201e-01,  5.27868681e-01,  4.84473006e-01,  1.05587322e-01,
  	  5.18178767e-02,  8.50094542e-02,  9.69902369e-02]


#median parameters for HD102509 100,000 step run w gammas instead of gamma_os
p = [ 7.16902791e+01,  4.30725716e+04, -2.27864352e-04,  4.68399935e-04,
  	  3.00898058e+01,  1.86445675e+00,  1.77603777e+00,  1.28836824e+00,
  	  7.16327739e-01,  9.74341232e-01,  6.93817102e-01,  4.04686472e-01,
  	  3.96837830e-01,  2.78218700e-01,  4.31310021e-01,  1.22508258e-01,
  	  2.91819070e-02,  7.35546530e-02,  9.58132551e-02]



#median parameters for HD102509 100,000 step run w gammas instead of gamma_os -- taking out 1 set of spectra
p = [ 7.16902791e+01,  4.30725716e+04, -2.27864352e-04,  4.68399935e-04,
  	  3.00898058e+01,  1.77603777e+00,  1.28836824e+00,
  	  7.16327739e-01,  9.74341232e-01,  6.93817102e-01,  4.04686472e-01,
  	  2.78218700e-01,  4.31310021e-01,  1.22508258e-01,
  	  2.91819070e-02,  7.35546530e-02,  9.58132551e-02]

'''


#median parameters from L79 first run
p = [ 3.67959833e+03,  4.88297491e+04,  7.42153203e-01, -2.25073992e-01,
  4.57461506e+00,  4.14003163e+01,  4.14003163e+01,  4.14003163e+01,
  4.14003163e+01,  1.27090779e-01,  2.20538442e-01,  5.46693365e-01,
  5.19543822e-01]




#run MCMC
t, rv, rvErr = readObservations('./WDS04342/L79.txt', True)
print runMCMC(p, t, rv, rvErr, './WDS04342/chain_100000_gammas.txt', niter = 100000)






import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from funcs_FixedEcc import *
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


#median parameters from L79 first run
p = [ 3.67959833e+03,  4.88297491e+04,  7.42153203e-01, -2.25073992e-01,
  4.57461506e+00,  4.14003163e+01,  4.14003163e+01,  4.14003163e+01,
  4.14003163e+01,  1.27090779e-01,  2.20538442e-01,  5.46693365e-01,
  5.19543822e-01]


#median parameters for HD102509 100,000 step run w errorMult
p = [ 7.16900784e+01,  4.30725878e+04, -2.46513438e-04,  5.95445237e-04,
  	  3.00757323e+01,  1.88366697e+00,  1.78760200e+00,  1.27424424e+00,
  	  7.19202146e-01,  9.77856151e-01,  6.87239969e-01,  4.00793310e-01,
  	  4.44447603e-02,  2.87753672e-01,  9.67377623e-01,  2.04950820e+00,
  	  1.02475238e+00,  6.80393269e-01,  6.77493148e-01,  9.51853040e-01]
'''


#median start parameters for HD102509 100,000 step run w errorMult an fixed at zero eccentricity
p = [ 7.16900784e+01,  4.30725878e+04,
  	  3.00757323e+01,  1.88366697e+00,  1.78760200e+00,  1.27424424e+00,
  	  7.19202146e-01,  9.77856151e-01,  6.87239969e-01,  4.00793310e-01,
  	  4.44447603e-02,  2.87753672e-01,  9.67377623e-01,  2.04950820e+00,
  	  1.02475238e+00,  6.80393269e-01,  6.77493148e-01,  9.51853040e-01]

'''
#median parameters for HD102509 100,000 step run w errorMult and fixed at zero eccentricity
[7.16900737e+01 4.30725788e+04 3.00751732e+01 1.88152279e+00
 1.79440860e+00 1.27456723e+00 7.15247162e-01 9.81296682e-01
 6.79691402e-01 3.94885406e-01 4.26277584e-02 2.88102260e-01
 9.66705501e-01 1.95490426e+00 1.03524189e+00 6.81420302e-01
 6.70360320e-01 9.14737510e-01]
'''


#run MCMC
#t, rv, rvErr = readObservations('./WDS04342/L79.txt', True)
#print runMCMC(p, t, rv, rvErr, './WDS04342/chain_100000_gammas.txt', niter = 100000)


t, rv, rvErr = readObservations('./HD102509/HD102509.orb', True)
print runMCMC(p, t, rv, rvErr, './HD102509/chain_100000_fixedEcc.txt', niter = 100000)






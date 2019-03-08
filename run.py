import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from funcsTemp import *
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

	#start walkers in a ball near the optimal solution
	startlocs = [p + initrange(p, 7) * np.random.randn(ndim) for i in np.arange(nwalkers)]

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


t, rv, rvErr = readObservations('./HD102509/HD102509.orb', True)

'''
#set first guess of parameters for modeling
#p = [period, ttran, ecosomega, esinomega, K, gamma,...]
p = [3677, 48667, 0.84, -0.26, 4.4, 42]

# p = [...gamma_offset1, gamma_offset2, gamma_offset3...]
for i in range(1, len(t)):	
	p.append(1)

# p = [...jitterSqrd1, jitterSqrd2, jitterSqrd3,...]
for i in range(0, len(t)):
	p.append(0.0)


#median parameters from L79 first run
p = [ 3.67959833e+03,  4.88297491e+04,  7.42153203e-01, -2.25073992e-01,
  4.57461506e+00,  4.14003163e+01,  2.21742301e-01,  5.60983043e-01,
  3.70576015e-01,  1.27090779e-01,  2.20538442e-01,  5.46693365e-01,
  5.19543822e-01]

'''

#median parameters for HD102509 10,000 step run
p = [ 7.16902685e+01,  4.30725785e+04, -0.0003,  0.0004,
  	  30.094,  1.86, 1.86,  1.86,
  	  1.86,  1.86,  1.86,  1.86,
  	  2.11738201e-01,  5.27868681e-01,  4.84473006e-01,  1.05587322e-01,
  	  5.18178767e-02,  8.50094542e-02,  9.69902369e-02]


#run MCMC
print runMCMC(p, t, rv, rvErr, './HD102509/chain_100000_gammas.txt', niter = 100000)






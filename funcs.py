import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import csv


def initrange(p, numSpec):
	"""
	Return initial error estimates in each parameter.
	Used to start the MCMC chains in a small ball near an estimated solution.

	Input
	-----
	p : ndarray
		Model parameters. See light_curve_model for the order.

	numSpec: int
		Number of observatories

	Returns
	-------
	errs : ndarray
		The standard deviation to use in each parameter
		for MCMC walker initialization.
	"""

	
	errorEst = [5.8e-03,   1.4e-01,   2.5e-03,   2.5e-03,   3.8e-02]

	for i in range(0, numSpec):
		errorEst.append(4.8e-02)

	for i in range(0, numSpec):
		errorEst.append(1e-5)

	errorEst = np.array(errorEst)

	return errorEst



def kepler(M, e):
	"""
	Simple Kepler solver.
	Iterative solution via Newton's method. Could likely be sped up,
	but this works for now; it's not the major roadblock in the code.

	Input
	-----
	M : ndarray
	e : float or ndarray of same size as M

	Returns
	-------
	E : ndarray
	"""

	M = np.array(M)
	E = M * 1.
	err = M * 0. + 1.

	while err.max() > 1e-8:
		#solve using Newton's method
		guess = E - (E - e * np.sin(E) - M) / (1. - e * np.cos(E))
		err = np.abs(guess - E)
		E = guess

	return E



def RV_model(t, period, ttran, ecosomega, esinomega, K, gamma):
	"""
	Given the orbital parameters compute the RV at times t, without gamma

	Input
	-----
	t : ndarray
		Times to return the model RV.
	period : float [days]
	ttran : float [days]
	ecosomega : float
	esinomega : float 
	K : float [km/s]
	gamma : float [km/s]

	Returns
	-------
	RV_model : ndarray
		RV corresponding to the times in t [km/s].

	"""




	e = np.sqrt(ecosomega**2. + esinomega**2.)
	omega = np.arctan2(esinomega, ecosomega)

	#mean motion: n = 2pi/period
	n = 2. * np.pi / period

	# Sudarsky 2005 Eq. 9 to convert between center of transit
	# and pericenter passage (tau)
	edif = 1. - e**2.
	fcen = np.pi/2. - omega
	tau = (ttran + np.sqrt(edif) * period / (2 * np.pi) * 
		  (e * np.sin(fcen) / (1. + e * np.cos(fcen)) - 2. / np.sqrt(edif) * 
		  np.arctan(np.sqrt(edif) * np.tan(fcen / 2.) / (1. + e))))


	#Define mean anomaly: M
	M = (n * (t - tau)) % (2. * np.pi)



	#Determine the Energy: E
	E = kepler(M, e)

	#Solve for fanom (measure of location on orbit)
	tanf2 = np.sqrt((1. + e) / (1. - e)) * np.tan(E / 2.)
	fanom = (np.arctan(tanf2) * 2.) % (2. * np.pi)

	#Calculate RV at given location on orbit
	RV = K * (e * np.cos(omega) + np.cos(fanom + omega)) + gamma

	return RV



def loglikelihood(p, t, RV, RVerr, chisQ=False):
	"""
	Compute the log likelihood of a RV signal with these orbital
	parameters given the data. 
	
	Input
	-----
	p : ndarray
		Orbital parameters. See RV model for order
	t, RV, RVerr : MxNdarray
		times, RV, and RV errors of the data.
		arranged as a list of lists
		len(array) = number of observing devices

	chisQ : boolean, optional
        If True, we are trying to minimize the chi-square rather than
        maximize the likelihood. Default False.


		
	Returns
	------
	likeli : float
		Log likelihood that the model fits the data.
	"""

	# Define all parameters except gamma and jitters
	(period, ttran, ecosomega, esinomega, K) = p[0:5]

	# Determine the number of spectra observers
	numSpec = len(t)

	# Define a list of gamma parameters
	gammas = p[5:5+numSpec]

	# Define a list of jitter squared parameters
	jitterSqrd = p[5+numSpec:len(p)]

	# Check observations are consistent
	if ( len(t) != len(RV) or len(t) != len(RVerr) or len(RV) != len(RVerr) ):
		print "Error! Mismatched number of spectra!"

	# Compute RV model light curve for the first all spectra without gamma added
	models = []
	for ii in range(0, numSpec):
		models.append(RV_model(t[ii], period, ttran, ecosomega, esinomega, K, gammas[ii]))


	# compute loglikelihood for model
	# Eastman et al., 2013 equation 
	# Christiansen et al., 2017 sec. 3.2 eq. 1
	totchisq = 0
	loglikelihood = 0

	for ii in range(0, numSpec):
		totchisq += np.sum((RV[ii]-models[ii])**2. / ( (RVerr[ii]**2. + jitterSqrd[ii]) ))	
		loglikelihood += -np.sum( 
			(RV[ii]-models[ii])**2. / ( 2. * (RVerr[ii]**2. + jitterSqrd[ii]) ) +
			np.log(np.sqrt(2. * np.pi * (RVerr[ii]**2. + jitterSqrd[ii])))
			)



	# If we want to minimize chisQ, return it now
	if chisQ:
		return totchisq
	
	# Else return log likelihood
	return loglikelihood



def logprior(p, numSpec):
	"""
	Priors on the input parameters.

	Input
	-----
	p : ndarray
		Orbital parameters. RV_model for the order.

	numSpec : int
		Number of individual sets of spectra observations
		
	Returns
	-------
	prior : float
		Log likelihood of this set of input parameters based on the
		priors.
	"""

	# Define all parameters except gammas and jitters
	(period, ttran, ecosomega, esinomega, K) = p[0:5]

	# Define a list of gamma offset parameters
	gammas = p[5:5+numSpec]

	# Define a list of jitter squared parameters
	jitterSqrd = p[5+numSpec:len(p)]

	e = np.sqrt(ecosomega**2. + esinomega**2.)
	omega = np.arctan2(esinomega, ecosomega)

	#If any parameters not physically possible, return negative infinity
	if (period < 0. or e < 0. or e >= 1.):
		return -np.inf

	# If jitter squared terms not physically possible or too large (> 1 km/s), return negative infinity
	for ii in range(0, len(jitterSqrd)):
		if (jitterSqrd[ii]**.5 > 1 or jitterSqrd[ii] < 0):
			return -np.inf

	totchisq = 0.


	# otherwise return a uniform prior (except modify the eccentricity to
	# ensure the prior is uniform in e)
	return -totchisq / 2. - np.log(e)





def logprob(p, t, RV, RVerr):
	"""
	Get the log probability of the data given the priors and the model.
	See loglikeli for the input parameters.
	
	Returns
	-------
	prob : float
		Log likelihood of the model given the data and priors, up to a
		constant.
	"""
	numSpec = len(t)
	lp = logprior(p, numSpec)
	llike = loglikelihood(p, t, RV, RVerr)


	if not np.isfinite(lp):
		return -np.inf

	if not np.isfinite(llike):
		return -np.inf

	return lp + llike 



def readObservations(filename, sepSpectra=False):
	"""
	Take data as output from TRES and turn into parsable t, RV, and RVerr np.arrays

	Input
	-----
	filename : string
		Name of file containing data

	sepSpectra : boolean
		default = False
		if True will seperate spectra by built in tags

	Returns
	------
	[t, RV, RVerr] : list of ndarrays
		times, RV, and RV errors of the data.
		

	"""
	# Read data from the observing output file
	# Each sub list contains [time in JD, RV, IGNORE, RVerr]
	observations = []
	with open(filename, "rb") as f:
		reader = csv.reader(f)
		for row in reader:
			observations.append(row)

	# Remove first and last discriptionary rows
	observations = observations[1:len(observations)-1]

	# Seperate out individual rows
	tStart = []
	tEnd = []
	rvStart = []
	rvEnd = []
	specStart = []
	specEnd = []
	rvErrStart = []
	rvErrEnd = []

	pointer = 'tStart'
	for observation in observations:
		for index in range(0, len(observation[0])):
			element = observation[0][index]
			
			if pointer == 'tStart':
				if element != ' ':
					tStart.append(index)
					pointer = 'tEnd' 	
							
			elif pointer == 'tEnd':
				if element == ' ':
					tEnd.append(index-1)
					pointer = 'rvStart'

			elif pointer == 'rvStart':
				if element != ' ':
					rvStart.append(index)
					pointer = 'rvEnd'

			elif pointer == 'rvEnd':	
				if element == ' ':
					rvEnd.append(index-1)
					pointer = 'specStart'

			elif pointer == 'specStart':
				if element != ' ':
					specStart.append(index)
					pointer ='specEnd'
				
			elif pointer == 'specEnd':
				if element == ' ':
					specEnd.append(index-1)
					pointer = 'rvErrStart'

			elif pointer == 'rvErrStart':
				if element != ' ':
					rvErrStart.append(index)
					rvErrEnd.append(len(observation[0])-1)
					pointer = 'tStart'
					break


	t = []
	rv = []
	rvErr = []
	spectra = []
	for obsNum in range(0, len(observations)):
		t.append(float(observations[obsNum][0][tStart[obsNum]:tEnd[obsNum] + 1]) )
		rv.append(float(observations[obsNum][0][rvStart[obsNum]:rvEnd[obsNum] + 1]) )
		rvErr.append(float(observations[obsNum][0][rvErrStart[obsNum]:rvErrEnd[obsNum] + 1]) ) 

		if sepSpectra:
			#wds04342
			#spectra.append(int(observations[obsNum][0][specStart[obsNum]:specEnd[obsNum] + 1]))

			#hd102509
			spectra.append(int(observations[obsNum][0][specStart[obsNum] + 4:specEnd[obsNum] + 1]))

	tSep = []
	rvSep = []
	rvErrSep = []

	tFinal = []
	rvFinal = []
	rvErrFinal = []
	if sepSpectra:

		currentSpec = 0
		for spec in range(0, len(spectra)):
			if spectra[spec] == currentSpec:
				tSep.append(t[spec])
				rvSep.append(rv[spec])
				rvErrSep.append(rvErr[spec])


			else:
				tSep = np.array(tSep)
				rvSep = np.array(rvSep)
				rvErrSep = np.array(rvErrSep)

				tFinal.append(tSep)
				rvFinal.append(rvSep)
				rvErrFinal.append(rvErrSep)

				tSep = []
				tSep.append(t[spec])
				rvSep = []
				rvSep.append(rv[spec])
				rvErrSep = []
				rvErrSep.append(rvErr[spec])

				currentSpec = spectra[spec]

			if spec == len(spectra)-1:
				tSep = np.array(tSep)
				rvSep = np.array(rvSep)
				rvErrSep = np.array(rvErrSep)

				tFinal.append(tSep)
				rvFinal.append(rvSep)
				rvErrFinal.append(rvErrSep)

				tSep = []
				tSep.append(t[spec])
				rvSep = []
				rvSep.append(rv[spec])
				rvErrSep = []
				rvErrSep.append(rvErr[spec])

				currentSpec = spectra[spec]

	return [tFinal, rvFinal, rvErrFinal]


"""
A slight modification to Scipy's implementation of differential evolution. To speed up predictions, the entire parameters array is passed to `self.func`, where a neural network model can batch its computations.

Taken from
https://github.com/scipy/scipy/blob/70e61dee181de23fdd8d893eaa9491100e2218d7/scipy/optimize/_differentialevolution.py

----------

differential_evolution: The differential evolution global optimization algorithm
Added by Andrew Nelson 2014
"""
from __future__ import division, print_function, absolute_import
from audioop import reverse
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize.optimize import _status_message
from scipy._lib._util import check_random_state
from scipy._lib.six import xrange, string_types
import matplotlib.pyplot as plt
import warnings


__all__ = ['differential_evolution']

_MACHEPS = np.finfo(np.float64).eps


def differential_evolution(func, bounds, args=(), strategy='rand1bin',
                           maxiter=1000, popsize=15, tol=0.01,
                           mutation=0.5, recombination=0.7, seed=None,
                           callback=None, disp=False, polish=True,
                           init='latinhypercube', atol=0, sharing_radius=0.2):


    solver = DifferentialEvolutionSolver(func, bounds, args=args,
                                         strategy=strategy, maxiter=maxiter,
                                         popsize=popsize, tol=tol,
                                         mutation=mutation,
                                         recombination=recombination,
                                         seed=seed, polish=polish,
                                         callback=callback,
                                         disp=disp, init=init, atol=atol, sharing_radius=sharing_radius)
    return solver.solve()


class DifferentialEvolutionSolver(object):



    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'currenttobest1bin': '_currenttobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1',
                 'niche1bin': '_currenttoniche'}

    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'currenttobest1exp': '_currenttobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2',
                    'niche1exp': '_currenttoniche'}

    __init_error_msg = ("The population initialization method must be one of "
                        "'latinhypercube' or 'random', or an array of shape "
                        "(M, N) where N is the number of parameters and M>5")

    def __init__(self, func, bounds, args=(),
                 strategy='rand1bin', maxiter=1000, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=np.inf, callback=None, disp=False, polish=True,
                 init='latinhypercube', atol=0, sharing_radius=0.2):

        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy
        # print (self.strategy)
        self.callback = callback
        self.polish = polish
        # relative and absolute tolerances for convergence
        self.tol, self.atol = tol, atol
        self.sharing_radius = sharing_radius
        # print (self.sharing_radius)

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.

        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            raise ValueError('The mutation constant must be a float in '
                             'U[0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in U[0, 2).')

        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        self.func = func
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.limits = np.array(bounds, dtype='float').T
        if (np.size(self.limits, 0) != 2 or not
                np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence containing '
                             'real valued (min, max) pairs for each value'
                             ' in x')

        if maxiter is None:  # the default used to be None
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:  # the default used to be None
            maxfun = np.inf
        self.maxfun = maxfun

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        # print (self.limits)
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.parameter_count = np.size(self.limits, 1)

        # print (self.__scale_arg1, self.__scale_arg2)
        # print (self.parameter_count)

        self.random_number_generator = check_random_state(seed)

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.
        # the minimum is 5 because 'best2bin' requires a population that's at
        # least 5 long
        self.num_population_members = max(5, popsize * self.parameter_count)

        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        self._nfev = 0
        if isinstance(init, string_types):
            if init == 'latinhypercube':
                self.init_population_lhs()
            elif init == 'random':
                self.init_population_random()
            else:
                raise ValueError(self.__init_error_msg)
        else:
            self.init_population_array(init)

        self.disp = disp

    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the following
        # size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (segsize * rng.random_sample(self.population_shape)

        # Offset each segment to cover the entire parameter range [0, 1)
                   + np.linspace(0., 1., self.num_population_members,
                                 endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        self.population = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies
        self.population_energies = (np.ones(self.num_population_members) *
                                    np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_random(self):
        """
        Initialises the population at random.  This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population_shape)

        # reset population energies
        self.population_energies = (np.ones(self.num_population_members) *
                                    np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_array(self, init):
        """
        Initialises the population with a user specified population.
        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (M, len(x)), where len(x) is the number of parameters.
            The population is clipped to the lower and upper `bounds`.
        """
        # make sure you're using a float array
        popn = np.asfarray(init)

        if (np.size(popn, 0) < 5 or
                popn.shape[1] != self.parameter_count or
                len(popn.shape) != 2):
            raise ValueError("The population supplied needs to have shape"
                             " (M, len(x)), where M > 4.")

        # scale values and clip to bounds, assigning to population
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)

        self.num_population_members = np.size(self.population, 0)

        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        # reset population energies
        self.population_energies = (np.ones(self.num_population_members) *
                                    np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        """
        return self._scale_parameters(self.population)

    @property
    def convergence(self):
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        return (np.std(self.population_energies) /
                np.abs(np.mean(self.population_energies) + _MACHEPS))

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """
        nit, warning_flag = 0, False
        status_message = _status_message['success']
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()
        # do the optimisation.
        for nit in xrange(1, self.maxiter + 1):
            next(self) #without early stop search for diverse solution
        status_message = _status_message['maxiter']
        # print (np.max(self.population_energies))
        DE_result = OptimizeResult(
            x= self._scale_parameters(self.population),
            fun=self.population_energies,
            nfev=self._nfev,
            nit=nit,
            message=status_message
            )
        return DE_result

    def _calculate_population_energies(self):
        """
        Calculate the energies of all the population members at the same time.
        Puts the best member in first place. Useful if the population has just
        been initialised.
        """

        ##############
        ## CHANGES: self.func operates on the entire parameters array
        ##############
        itersize = max(0, min(len(self.population), self.maxfun - self._nfev + 1))
        # print (itersize)
        candidates = self.population[:itersize]
        parameters = np.array([self._scale_parameters(c) for c in candidates]) # TODO: vectorize
        # print (parameters)
        energies = self.func(parameters, *self.args)
        # print (energies)
        self.population_energies = energies
        self._nfev += itersize
        ##############
        ##############
        # minval = np.argmin(self.population_energies)
        # # put the lowest energy into the best solution position.
        # lowest_energy = self.population_energies[minval]
        # self.population_energies[minval] = self.population_energies[0]
        # self.population_energies[0] = lowest_energy
        # self.population[[0, minval], :] = self.population[[minval, 0], :]

    def __iter__(self):
        return self

    def __next__(self):

        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        if self.dither is not None:
            self.scale = (self.random_number_generator.rand()
                          * (self.dither[1] - self.dither[0]) + self.dither[0])
        
        itersize = max(0, min(self.num_population_members, self.maxfun - self._nfev + 1)) 


        trials = np.array([self._mutate(c) for c in range(itersize)]) # TODO: vectorize
        for trial in trials: self._ensure_constraint(trial)
        parameters = np.array([self._scale_parameters(trial) for trial in trials])
        energies = self.func(parameters, *self.args)
        self._nfev += itersize

        energies_all = np.concatenate((self.population_energies, energies), 0) # all confidence score 
        population_all = np.concatenate((self.population, trials), 0) 

        #compute all sharing fitness 
        pos = self._extract_coordinate(population_all)

        pair_dist = self._pairwise_euclidean(pos)
        shdist = self._sharing_distance(pair_dist, self.sharing_radius, 1.0)
        shfit = energies_all / shdist 

        # Optional 1. following DE compared with parents only
        # shfit_parents, shfit_trials = shfit[:self.num_population_members], shfit[self.num_population_members:]
        # for candidate, (shfit_trial, shfit_parent, trial) in enumerate(zip(shfit_trials, shfit_parents, trials)):
        #     if shfit_trial > shfit_parent:
        #         self.population[candidate] = trial 
        #         self.population_energies[candidate] = energies[candidate]
        
        # optional 2. Rank and select better performance population 

        elitism_index = np.argmax(energies_all)
        select_index = np.argsort(-shfit)[:self.num_population_members].reshape(-1)
        if elitism_index not in select_index:
            select_index[-1] = elitism_index

        self.population = np.take(population_all, select_index, axis=0)
        self.population_energies = np.take(energies_all, select_index, axis=0)

        return self.population, self.population_energies

    def next(self):
        """
        Evolve the population by a single generation
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # next() is required for compatibility with Python2.7.
        return self.__next__()
    
    
    def _extract_coordinate(self, parameters):
        '''
        parameters: (popsize, pixels*(x, y, r, g, b)) Nx(5*n)
        positions:  (popsize, pixels, (x, y))  Nxnx2
        '''
        num_pixels = int(parameters.shape[1]/ 5)
        coord_pos = np.zeros(shape=(parameters.shape[0], num_pixels, 2))
        for i in range(num_pixels):
            coord_pos[:, i, :] = parameters[:, [i*5, i*5+1]]
        return coord_pos

    def _pairwise_euclidean(self, pos):
        '''
        compute the minimum euclidean distance between two set of point
        input: (popsize, pixels, (x, y))  Nxnx2
        '''
        # pos = position.clone() 
        # pos = pos.astype(int)
        # print (pos)
        if pos.shape[1] == 1:
            pos = np.reshape(pos, (pos.shape[0], 2))
            #one pixel attack just need to compute euclidean distance 
            dist = (pos[:, None, :] - pos[None,:,:])**2
            dist = (np.sum(dist, axis=-1, keepdims=False))**0.5
        else:
            dist = np.zeros(shape=(pos.shape[0], pos.shape[0]))
            for i in range(pos.shape[1]):
                pixel = pos[:,i,:]
                pixel = np.tile(pixel[:,None,:], (1, pos.shape[1], 1))
                pixel_dist = (pixel[:, None, :, :] - pos[None,:,:,:])**2
                pixel_dist = np.min(((np.sum(pixel_dist, axis=-1, keepdims=False))**0.5),axis=-1, keepdims=False)
                dist += pixel_dist
            dist = dist/pos.shape[1]
        return dist 
    
    def _sharing_distance(self, dist, radius, alpha):
        dist = np.maximum(0.0,1 - (dist/radius)**alpha)
        shdist= np.sum(dist, axis=-1)
        return shdist 
    
    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """
        make sure the parameters lie between the limits
        """
        for index in np.where((trial < 0) | (trial > 1))[0]:
            trial[index] = self.random_number_generator.rand()

    def _mutate(self, candidate, dist=None):
        """
        create a trial vector based on a mutation strategy
        """
        trial = np.copy(self.population[candidate])

        rng = self.random_number_generator

        fill_point = rng.randint(0, self.parameter_count)

        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidate,
                                        self._select_samples(candidate, 5))
        else:
            # bprime = self.mutation_func(self._select_niche_samples(candidate, 5, distance=dist))
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        if self.strategy in self._binomial:
            crossovers = rng.rand(self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.strategy in self._exponential:
            i = 0
            while (i < self.parameter_count and
                   rng.rand() < self.cross_over_probability):

                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1

            return trial

    def _best1(self, samples):
       
        """
        best one mutation with random select two individual
        best1bin, best1exp
        """
        r0, r1 = samples[:2]
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _currenttoniche(self, candidate, samples):
        r0, r1 = samples[:2]
        return (self.population[candidate] *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        """
        random select three individual
        rand1bin, rand1exp
        """
        r0, r1, r2 = samples[:3]
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, samples):
        """
        random choose one mutation with best first then use it to mutation with 
        random two individuals
        randtobest1bin, randtobest1exp
        """
        r0, r1, r2 = samples[:3]
        bprime = np.copy(self.population[r0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r1] -
                                self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        """
        current best and random two
        currenttobest1bin, currenttobest1exp
        """
        r0, r1 = samples[:2]
        bprime = (self.population[candidate] + self.scale * 
                  (self.population[0] - self.population[candidate] +
                   self.population[r0] - self.population[r1]))
        return bprime

    def _best2(self, samples):
        """
        best random 4
        best2bin, best2exp
        """
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.scale *
                  (self.population[r0] + self.population[r1] -
                   self.population[r2] - self.population[r3]))

        return bprime

    def _rand2(self, samples):
        """
        random 5
        rand2bin, rand2exp
        """
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.scale *
                  (self.population[r1] + self.population[r2] -
                   self.population[r3] - self.population[r4]))

        return bprime
    # def _assorative(self, candidate, samples):

    def _select_samples(self, candidate, number_samples):
    
        """
        obtain random integers from range(self.num_population_members),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs

    def _select_niche_samples(self, candidate, number_samples, distance):

        candidate_dist = distance
        idxs = list(np.where(candidate_dist<self.sharing_radius)[0])
        # print (idxs)
        # idxs.remove(candidate)
        if len(idxs) < number_samples:
            idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs
    def _select_samples2(self, candidate,  number_samples, fitness):
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)

        return idxs

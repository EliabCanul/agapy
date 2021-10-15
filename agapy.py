# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
This is a python version of AGA (asexual genetic algorithm), a method to find
local and global minima of analytic/numeric functions.
Originally proposed by Cantó et al 2009: https://arxiv.org/pdf/0905.3712.pdf

Author: Eliab F. Canul
"""

import numpy as np
from multiprocessing import Pool
import warnings
from dataclasses import dataclass , field


# TODO:  set a __call__ function 

@dataclass
class Agapy:
	"""A class to perform the asexual genetic algorithm to find function
	minima
	"""

	func  : object   # The merith function to minimize
	bounds : tuple   # An array of boundaries with shape (ndim, [lower upper])
	kwargs : dict = field(default_factory=dict) # kwargs for the merith function
	generations : int = 50  	# Maximum number of generations
	individuals : int = 50 		# Number of idividuals 
	fparents : float = 0.1		# Fraction for parents respect to individuals
	cores : int = 1  			# Number of cores
	# dispersion / convergency / tolerance
	init_pop : tuple = None     # A given intial population, otherwise random
	save_evolution : bool = False # A flag to store the evolution 
	
	
	def create_population(self, bounds, n):
		"""Create a population from random uniform dstribution

		Parameters
		----------
		bounds : arr
			An array delimiting the boundaries of shape (ndim, limits),
			where limits is of the form [lower, upper] 
		n : int
			The number of points 

		Returns
		-------
		arr
			An array of shape (n, ndim)
		"""
		
		np.random.seed()

		boundsT = bounds.T

		init_pop = np.random.uniform(low=boundsT[0], high=boundsT[1], 
									 size=(n,self.ndim) )

		return init_pop


	def evaluate(self,  x):
		"""Evaluate the given function. The class attribute kwargs is used here

		Parameters
		----------
		x : arr
			The batch of data points to evaluate, with elements of length ndim

		Returns
		-------
		arr
			An array of the function values for every datapoint
		"""				

		# TODO: Make parallel the function evaluation
		"""
		with warnings.catch_warnings(record=True) as w: 
			warnings.simplefilter("always")

			pool = Pool(processes=self.cores)

			results = [ pool.apply_async(func, (ind,)) for ind in population ]
			
			output = [p.get() for p in results]	
			print(output)	

		pool.terminate()
		"""

		output =  self.func(x.T, **self.kwargs)

		return output


	def sort(self, f, x):
		"""Sort arrays according to the function values from low to high

		Parameters
		----------
		f : arr
			An array of the function values 
		X : arr
			The array of the data points

		Returns
		-------
		tuple
			A tuple with the function values and data points sorted according
			to values from low to high
		"""

		sort_index = np.argsort(f)

		return (f[sort_index], x[sort_index])
	

	""" Old way:
		def reproduction(self, parents, side):
		
		# Make the reproduction using one of the algorithm:
		# TODO: gaussian: make new population around the parents
			using gaussian ball. Parents loca
		# linear: reduce the boundaries around parents based
			on the closing parameter: p
		
		#ndim = len(self.bounds)
		#nchild = int(self.individuals/self.nparents)

		children = np.zeros((self.nparents,self.nchild,self.ndim))
		frontiers = []
		# Se calculan fronteras individuales para cada padre
		for ip, parent in enumerate(parents):
				# New bounds
				nb = [
					(max(self.bounds[i][0], parent[i]-side[i]), 
					min(self.bounds[i][1], parent[i]+side[i]))
					for i in range(self.ndim)]
				frontiers.append(nb)

				ch = self.create_population(nb, self.nchild)

				# Clone parent
				ch[0] = parent
				#Update the new generation
				children[ip,:] = ch

		return frontiers, children.reshape(-1, children.shape[-1])"""


	def linear_reduction(self, parents, side):
		"""A function to linearly reduce the boundaries around every parent

		Parameters
		----------
		parents : arr
			The parents coordinates to center the new boundaries
		side : arr
			An array of shape (ndim,) containing the length of the new bounds
		
		Returns
		-------
		arr
			An array of shape (parent, dimension, low, upp) with the new
			boundaries centered over the parents coordinates
		"""		

		nb = np.zeros((self.nparents, self.ndim, 2))
		
		for d in range(self.ndim):
			l_bounds = np.maximum(parents.T[d]-side[d],self.bounds[d,0])
			r_bounds = np.minimum(parents.T[d]+side[d],self.bounds[d,1])
			
			nb[:,d,0] = l_bounds
			nb[:,d,1] = r_bounds
			
		return nb


	def reproduction(self, parents, bounds):
		"""A function to reproduce the individuals around the parents coordina-
		tes. The number of new children are such that complete the total number
		of individuals

		Parameters
		----------
		parents : arr
			An array with the parents coordinates, shape (nparents, ndim)
		bounds : arr
			The boundaries centered aroun every single parent where the new
			children will be born, shape (nparents, limits, ndim)

		Returns
		-------
		arr
			The population of individuals where the first elements are a copy
			of the parents coordinates and the remaining are the new children
		"""		

		pop = np.zeros((self.individuals, self.ndim))
		
		# Make a copy of the parents
		pop[:self.nparents] = parents

		for i, n_b in enumerate(bounds):
			idx_init = self.nparents + i*(self.nchild-1)
			idx_fin = idx_init+(self.nchild-1)
			
			hijos = self.create_population(n_b, self.nchild-1 )
			pop[idx_init:idx_fin] = hijos	

		return pop


	def run(self):
		"""The calling function to perform the minima search

		Returns
		-------
		tuple
			if save_evolution:
				Returns a tuple with the results at each generation:
				(sides, values, coordinates), where values and coordinates
				are sorted according to values from minimum to maximum
			else:
				Returns a tuple with the function value and coordinates of the
				best solution.
		"""		

		# Verification
		assert 0 < self.fparents <=0.5, "fparents should be 0.5 at most"

		text = "fparents must be an integer fraction of individuals"
		assert (self.fparents * self.individuals).is_integer(), text
		

		# Initial population
		if self.init_pop != None:
			population = self.init_pop
		else:
			population = self.create_population(self.bounds, 
											self.individuals)
		
		# Initial half side of the boundaries
		side0 = (self.bounds[:,1] - self.bounds[:,0])/2 

		# The slope to linear decrement of the boundaries 
		m = (side0-.1)/self.generations

		# Prepare to store
		if self.save_evolution:
			fun_val = np.zeros((self.generations,self.individuals))
			evolution = np.zeros((self.generations,self.individuals,self.ndim))
			boxes = np.zeros((self.generations,self.ndim))


		for g in range(0, self.generations):
			
			# Evaluate
			values = self.evaluate(population)

			# Sort 
			fun, population = self.sort(values, population)

			# Check convergency 
			# TODO write convergence criteria
			
			# Update the boundaries length
			side = side0-m*g

			# Save
			if self.save_evolution:
				fun_val[g,:] = fun

				evolution[g,:,:] = population

				# TODO: Does is it necessary?
				boxes[g,:] = side


			# Select the parents
			parents = population[:self.nparents]

			# Reduce the boundaries
			new_bounds = self.linear_reduction(parents, side)

			# Reproduce the population
			population = self.reproduction(parents, new_bounds)


		if self.save_evolution:
			# fun and population orted from best to worse
			return boxes, fun_val, evolution 
		else:
			return (fun[0], parents[0]) 


	@property
	def ndim(self):
		return len(self.bounds)


	@property
	def nparents(self):
		return int(self.fparents * self.individuals)


	@property
	def nchild(self):
		return int(self.individuals/self.nparents)
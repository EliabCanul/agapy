# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
This is a python version of de AGA (asexual genetic algorithm),
originally proposed by Cantó et al 2009: https://arxiv.org/pdf/0905.3712.pdf

Author: Eliab F. Canul
"""

import numpy as np
from multiprocessing import Pool
import warnings
from dataclasses import dataclass


# TODO:  revisar __call__, necesito que la clase regrese los resultados

@dataclass
class agapy:

	"""
	AGAPY
	"""

	bounds : tuple = None
	generations : int = 50  	# Maximum number of Generations
	individuals : int = 50 		# Individuals per generation
	fparents : float = 0.1		# Population fraction for parents
	cores : int = 1  			# Number of cores
	# dispersion / convergency / tolerance
	p : float = 0.6 				# velocity of boundaries closing
	init_pop : tuple = None
	
	np.random.seed()
	
	def create_population(self, bounds, individuals):
		"""
		Create initial population
		"""
		ndim = len(bounds)
		bounds = np.array(bounds).T
		init_pop = np.random.uniform(low=bounds[0], high=bounds[1], 
									 size=(individuals,ndim) )

		return init_pop


	def evaluate(self, func, population, args):
		"""
		Evaluate the current population
		"""

		# TODO: Make parallel this code
		# Evaluate in parallel
		"""
		with warnings.catch_warnings(record=True) as w: 
			warnings.simplefilter("always")

			pool = Pool(processes=self.cores)

			results = [ pool.apply_async(func, (ind,)) for ind in population ]
			
			output = [p.get() for p in results]	
			print(output)	

		pool.terminate()
		"""
		fun = np.array([[func(p, *args)] for p in population])

		output = np.concatenate( (fun,population), axis=1)

		return output


	def sort(self, evaluation):
		"""Sort from low fun to high fun
		"""
		evaluation = evaluation[evaluation[:,0].argsort()]

		fun, population = evaluation[:,0], evaluation[:,1:]

		return fun, population
	

	def reproduction(self, parents, side):
		"""
		Make the reproduction using one of the algorithm:
		TODO: gaussian: make new population around the parents
			using gaussian ball. Parents loca
		linear: reduce the boundaries around parents based
			on the closing parameter: p
		"""
		ndim = len(self.bounds)
		nchild = int(self.individuals/self.nparents)

		children = np.zeros((self.nparents,nchild,ndim))
		frontiers = []
		for ip, parent in enumerate(parents):
				# New bounds
				nb = [
					(max(self.bounds[i][0], parent[i]-side[i]), 
					min(self.bounds[i][1], parent[i]+side[i]))
					for i in range(len(self.bounds))]
				frontiers.append(nb)

				ch = self.create_population(nb, nchild)

				# Clone parent
				ch[0] = parent
				#Update the new generation
				children[ip,:] = ch

		return frontiers, children.reshape(-1, children.shape[-1])


	def run(self, func, args=()):


		assert 0 < self.fparents <=0.5, "fparents should be 0.5 at most"

		self.nparents = self.fparents * self.individuals
		assert self.nparents.is_integer(), "fparents must be an integer fraction of individuals"
		self.nparents = int(self.nparents)

		# Initial population
		if self.init_pop != None:
			population = self.init_pop
		else:
			population = self.create_population(self.bounds, self.individuals)
		
		# 
		b_arr = np.array(self.bounds)
		side0 = (b_arr[:,1] - b_arr[:,0])/2 # self.fclosing

		evolution = []
		boxes = []
		fun_val = []
		for g in range(1, self.generations+1):
			
			#print(f"Generation: {g}")

			# Evaluate
			evaluation = self.evaluate(func, population, args)

			# Sort 
			fun, population = self.sort(evaluation)

			fun_val.append(fun)

			# Check convergency 
			#print()

			# Reproduce and update the population
			side = side0*self.p**g
			parents = population[:self.nparents]
			
			front, population = self.reproduction(parents,side)
			
			#side = side/self.fclosing		

			boxes.append(front)

			evolution.append(population)



		return boxes, fun_val, evolution # Last generation sorted from best to worse


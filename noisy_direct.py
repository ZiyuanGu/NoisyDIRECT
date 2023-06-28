# -*- coding: utf-8 -*-
"""
@author: Ziyuan Gu
"""

import numpy as np
from scipy.stats import norm, invgamma
from scipy.spatial import ConvexHull

"""
NoisyDirect is a simulation-based robust optimization algorithm that extends the original DIRECT algorithm for noisy objective fucntions.

References:
    Gu, Z., Li, Y., Saberi, M., Rashidi, T.H., Liu, Z., 2023. Macroscopic parking dynamics and equitable pricing: Integrating trip-based modeling with simulation-based robust optimization. Transp. Res. Part B 173, 354-381.
    Jones, D.R., Perttunen, C.D., Stuckman, B.E., 1993. Lipschitzian optimization without the Lipschitz constant. J. Optim. Theory Appl. 79(1), 157-181.
    Deng, G., Ferris, M.C., 2007. Extension of the DIRECT optimization algorithm for noisy functions, 2007 Winter Simulation Conference. IEEE, Washington, DC, pp. 497-504.
    Deng, G., Ferris, M.C., 2006. Adaptation of the UOBYQA Algorithm for Noisy Functions, Proceedings of the 2006 Winter Simulation Conference. IEEE, Monterey, CA, pp. 312-319.
    
Parameters:
    eps: the epsilon parameters used by the original DIRECT algorithm for finding the potentially optimal hyperrectangles
    min_eval_per_point: the minimum number of objective function evaluations for each decision vector
    max_eval_per_point: the maximum number of objective function evaluations for each decision vector
    add_eval_per_point: the additional number of objective function evaluations for each decision vector every time the significance level is violated (i.e., uncertainty still exists)
    max_total_func_eval: the maximum number of objective function evaluations (one of the three termination criteria)
    max_total_point_eval: the maximum number of decision vectors to be evaluated (one of the three termination criteria)
    max_iter: the maximum number of iterations (one of the three termination criteria)
    sig_level: the significance level (i.e. confidence about the uncertainty)
    noisy_direct: set to True to enable variable-number sample-path optimization, otherwise fixed-number sample-path optimization
    num_MC: the number of Monte Carlo experiments for identifying the potentially optimal hyperrectangles if noisy_direct is set to True
    robust_obj: set to True to enable robust optimization where the objective function becomes "minimize mu + M * max(std - std_thld, 0)", otherwise "minimize mu"
    M: the penalty parameter used by robust optimization which theoretically should be set to sufficiently large to satisfy the variance constraint
    var_thld: the variance threshold used by robust optimization
    
Usage:
    for deterministic objective functions, noisy_direct = False and min_eval_per_point = max_eval_per_point = 1
    for stochastic objective functions and fixed-number sample-path optimization, noisy_direct = False and min_eval_per_point = max_eval_per_point = any value other than 1
    for stochastic objective functions and variable-number sample-path optimization, noisy_direct = True
"""

class NoisyDirect:
    def __init__(self, eps=0.0001, min_eval_per_point=10, max_eval_per_point=100, add_eval_per_point=10, max_total_func_eval=10000, max_total_point_eval=1000, 
                 max_iter=100, sig_level=0.1, noisy_direct=True, num_MC=100, robust_obj=True, M=10, var_thld=10 ** 2):
        self.eps = eps
        self.min_eval_per_point = min_eval_per_point
        self.max_eval_per_point = max_eval_per_point
        self.add_eval_per_point = add_eval_per_point
        self.max_total_func_eval = max_total_func_eval
        self.max_total_point_eval = max_total_point_eval
        self.max_iter = max_iter
        self.sig_level = sig_level
        self.noisy_direct = noisy_direct
        self.num_MC = num_MC
        self.robust_obj = robust_obj
        self.M = M
        self.var_thld = var_thld
        self.dim = None # problem dimension or the size of the decision vector
        self.sim_data = {} # dictionary of simulation results
        self.iter_ = 0 # iteration counter
        self.total_func_eval = 0 # total number of function evaluations counter
        self.total_point_eval = 0 # total number of evaluated decion vectors counter
        self.x_min = None # numpy array of optimal decision vectors for each iteration
        self.lower_bounds = None # numpy array of lower bounds on the decision vector
        self.upper_bounds = None # numpy array of upper bounds on the decision vector
        self.total_func_eval_series = [] # total number of function evaluations per iteration
        self.total_point_eval_series = [] # total number of evaluated decion vectors per iteration
    
    
    """check the correctness of the input bounds"""
    def _check_bounds(self, lower_bounds, upper_bounds):
        # make lower bounds numpy array
        if isinstance(lower_bounds, list):
            lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        elif isinstance(lower_bounds, np.ndarray):
            lower_bounds.astype(np.float64)
            assert isinstance(lower_bounds.size, int)
        else:
            raise TypeError("Lower bounds should be either a list or a numpy array!")
        # make upper bounds numpy array
        if isinstance(upper_bounds, list):
            upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
        elif isinstance(upper_bounds, np.ndarray):
            upper_bounds.astype(np.float64)
            assert isinstance(upper_bounds.size, int)
        else:
            raise TypeError("Upper bounds should be either a list or a numpy array!")
        # check consistency between both bounds and record the problem dimension
        assert lower_bounds.size == upper_bounds.size
        self.dim = lower_bounds.size
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
    
    
    """convert the unit decision vector back to the true one"""
    def _back_transform(self, point):
        return point * (self.upper_bounds - self.lower_bounds) + self.lower_bounds
    
    
    """perform some number of function evaluations and record the results as a numpy array"""
    def _func_eval(self, point, num_eval, func):
        res = []
        point = point if isinstance(point, np.ndarray) else np.asarray(point) # make decision vectors a tuple for use in the dictionary
        true_point = self._back_transform(point)
        for _ in range(num_eval):
            res.append(func(true_point)) # every function evaluation should return a value (i.e. a stochastic realization of the expectation)
        return np.asarray(res)
    
    
    """evaluate the objective function"""
    def _obj_eval(self, point, enable_MC=False, random_seed=None):
        tuple_point = tuple(point) if isinstance(point, np.ndarray) else point # make decision vectors a tuple for use in the dictionary
        mu = np.mean(self.sim_data[tuple_point]['res']) # mean of function evaluations
        var = np.var(self.sim_data[tuple_point]['res'], ddof=1) # variance of functon evaluations
        num_eval = self.sim_data[tuple_point]['res'].size # number of functon evaluations
        # if Monte Carlo experiments are needed, sample the mean and the variance according to the posterior normal and inverse gamma distributions, respectively
        if enable_MC:
            mu = norm.rvs(loc=mu, scale=var / num_eval, random_state=random_seed) # random_state is for reproducibility
            var = invgamma.rvs(a=(num_eval - 1) / 2, loc=0, scale=var * (num_eval - 1) / 2, random_state=random_seed) # random_state is for reproducibility
        # if robust optimization is needed, the objective function value is mu + M * max(std - std_thld, 0)
        if self.robust_obj:
            obj = mu + self.M * max(np.sqrt(var) - np.sqrt(self.var_thld), 0)
        else:
            obj = mu
        return obj
    
    
    """initialize the unit hypercube"""
    def _init_c(self, func):
        c_init = np.full_like(self.lower_bounds, 1 / 2) # initialize the center/decision vector
        assert tuple(c_init) not in self.sim_data
        self.sim_data[tuple(c_init)] = {'side_len': None, 'res': None}
        self.sim_data[tuple(c_init)]['side_len'] = np.ones_like(self.lower_bounds) # initialize the corresponding side lengths
        # perform function evaluations for this decision vector and record the results
        self.sim_data[tuple(c_init)]['res'] = self._func_eval(c_init, self.min_eval_per_point, func)
        self.total_point_eval += 1
        self.total_func_eval += self.min_eval_per_point
        self.x_min = c_init.reshape(1, -1)
        return c_init
    
    
    """to satisfy the significance level of variance being smaller or larger than the variance threshold, only applicable when noisy_direct and robust_obj are both True"""
    def _satisfy_var(self, point, func):
        assert tuple(point) in self.sim_data
        var = np.var(self.sim_data[tuple(point)]['res'], ddof=1)
        num_eval = self.sim_data[tuple(point)]['res'].size
        # calculate the probability of variance being smaller than the variance threshold using the posterior inverse gamma distribution
        p_var = invgamma.cdf(x=self.var_thld, a=(num_eval - 1) / 2, loc=0, scale=var * (num_eval - 1) / 2)
        # if this probability is not small or large enough, perform extra function evaluations until the probability is satisfied
        if self.sig_level < p_var < 1 - self.sig_level:
            if num_eval + self.add_eval_per_point <= self.max_eval_per_point: # check if the maximum number of function evaluations will be reached
                self.sim_data[tuple(point)]['res'] = np.append(self.sim_data[tuple(point)]['res'], 
                                                               self._func_eval(point, self.add_eval_per_point, func))
                self.total_func_eval += self.add_eval_per_point
                self._satisfy_var(point, func) # recursive function call
    
    
    """to satisfy the significance level of mean assuming variance is now fixed, only applicable when noisy_direct is True"""
    def _satisfy_mu(self, point_1, point_2, func):
        assert tuple(point_1) in self.sim_data
        assert tuple(point_2) in self.sim_data
        # check if robust optimization is needed
        if self.robust_obj:
            self._satisfy_var(point_1, func)
            self._satisfy_var(point_2, func)
        # calculate the mean objective function value, the variance of function evaluations, and the number of function evaluations for both decision vectors to be compared
        # obj_1 = self._obj_eval(point_1)
        obj_1 = np.mean(self.sim_data[tuple(point_1)]['res'])
        var_1 = np.var(self.sim_data[tuple(point_1)]['res'], ddof=1)
        num_eval_1 = self.sim_data[tuple(point_1)]['res'].size
        # obj_2 = self._obj_eval(point_2)
        obj_2 = np.mean(self.sim_data[tuple(point_2)]['res'])
        var_2 = np.var(self.sim_data[tuple(point_2)]['res'], ddof=1)
        num_eval_2 = self.sim_data[tuple(point_2)]['res'].size
        # calculate the probability of obj_1 being no greater than obj_2
        # pcs = norm.cdf(x=(obj_2 - obj_1) / np.sqrt(var_1 / num_eval_1 + var_2 / num_eval_2))
        const = self.M * max(np.sqrt(var_2) - np.sqrt(self.var_thld), 0) - self.M * max(np.sqrt(var_1) - np.sqrt(self.var_thld), 0)
        pcs = norm.cdf(x=(obj_2 - obj_1 + const) / np.sqrt(var_1 / num_eval_1 + var_2 / num_eval_2))
        # if this probability is not small or large enough, perform extra function evaluations until the probability is satisfied
        if self.sig_level < pcs < 1 - self.sig_level:
            # determine which decision vector to be further evaluated and check if the maximum number of function evaluations will be reached 
            if var_1 / num_eval_1 ** 2 >= var_2 / num_eval_2 ** 2: 
                if num_eval_1 + self.add_eval_per_point <= self.max_eval_per_point:
                    self.sim_data[tuple(point_1)]['res'] = np.append(self.sim_data[tuple(point_1)]['res'], 
                                                                     self._func_eval(point_1, self.add_eval_per_point, func))
                else:
                    if num_eval_2 + self.add_eval_per_point <= self.max_eval_per_point:
                        self.sim_data[tuple(point_2)]['res'] = np.append(self.sim_data[tuple(point_2)]['res'], 
                                                                         self._func_eval(point_2, self.add_eval_per_point, func))
                    else:
                        return
            else:
                if num_eval_2 + self.add_eval_per_point <= self.max_eval_per_point:
                    self.sim_data[tuple(point_2)]['res'] = np.append(self.sim_data[tuple(point_2)]['res'], 
                                                                     self._func_eval(point_2, self.add_eval_per_point, func))
                else:
                    if num_eval_1 + self.add_eval_per_point <= self.max_eval_per_point:
                        self.sim_data[tuple(point_1)]['res'] = np.append(self.sim_data[tuple(point_1)]['res'], 
                                                                         self._func_eval(point_1, self.add_eval_per_point, func))
                    else:
                        return
            
            self.total_func_eval += self.add_eval_per_point
            self._satisfy_mu(point_1, point_2, func) # recursive function call
            
    
    """sort the minimum/maximum along each dimension of hyperrectangles"""
    def _sort_c_neighbors(self, c, func):
        max_side_len = np.max(self.sim_data[tuple(c)]['side_len']) # find the maximum side length
        delta = 1 / 3 * max_side_len # distance to the new neighboring centers
        dims = np.where(self.sim_data[tuple(c)]['side_len'] == max_side_len)[0] # find the dimensions with the maximum side length
        c_neighbors = {dim: {'min': None, 'max': None} for dim in dims} # record the minimum/maximum along each dimension as a dictionary
        # iterate through the dimensions with the maximum side length
        for dim in dims: 
            e = np.zeros_like(c)
            e[dim] = 1
            # evaluate one neighboring center and record the results
            c_plus = c + delta * e
            assert tuple(c_plus) not in self.sim_data
            self.sim_data[tuple(c_plus)] = {'side_len': None, 'res': None}
            self.sim_data[tuple(c_plus)]['res'] = self._func_eval(c_plus, self.min_eval_per_point, func)
            self.total_point_eval += 1
            self.total_func_eval += self.min_eval_per_point
            # evaluate the other neighboring center and record the results
            c_minus = c - delta * e
            assert tuple(c_minus) not in self.sim_data
            self.sim_data[tuple(c_minus)] = {'side_len': None, 'res': None}
            self.sim_data[tuple(c_minus)]['res'] = self._func_eval(c_minus, self.min_eval_per_point, func)
            self.total_point_eval += 1
            self.total_func_eval += self.min_eval_per_point
            # if noisy_direct is True, need to satisfy the significance level of mean when compared
            if self.noisy_direct:
                self._satisfy_mu(c_plus, c_minus, func)
            # record the minimum/maximum
            if self._obj_eval(c_plus) <= self._obj_eval(c_minus):
                c_neighbors[dim]['min'] = c_plus
                c_neighbors[dim]['max'] = c_minus
            else:
                c_neighbors[dim]['min'] = c_minus
                c_neighbors[dim]['max'] = c_plus
        return c_neighbors
    
    
    """sort the dimensions with the maximum side length for dividing hyperrectangles"""
    def _sort_dims(self, c_neighbors, func, current_x_min):
        # retrieve the dimensions and the minimum for each dimension
        dims, mins = [], []
        for dim in c_neighbors:
            dims.append(dim)
            mins.append(c_neighbors[dim]['min'])
        # sort the dimensions from the lowest minimum to the highest minimum in a for loop
        for i in range(len(dims)):
            min_idx = i
            for j in range(i + 1, len(dims)):
                # if noisy_direct is True, need to satisfy the significance level of mean when compared
                if self.noisy_direct:
                    self._satisfy_mu(mins[min_idx], mins[j], func)
                if self._obj_eval(mins[j]) < self._obj_eval(mins[min_idx]):
                    min_idx = j
            dims[i], dims[min_idx] = dims[min_idx], dims[i]
            mins[i], mins[min_idx] = mins[min_idx], mins[i]
        # if noisy_direct is True, need to satisfy the significance level of mean when compared
        if self.noisy_direct:
            self._satisfy_mu(mins[0], current_x_min, func)
        # update the current minimum
        if self._obj_eval(mins[0]) <= self._obj_eval(current_x_min):
            current_x_min = mins[0]
        return dims, current_x_min
    
    
    """divide potentially optimal hyperrectangles"""
    def _divide_c(self, c, c_neighbors, rev_sorted_dims):
        # divide new centers first in reverse order because centers along the dimension of the highest minimum is divided along all dimensions
        for idx, dim in enumerate(rev_sorted_dims):
            self.sim_data[tuple(c_neighbors[dim]['min'])]['side_len'] = self.sim_data[tuple(c)]['side_len'] * [1 / 3 if i in rev_sorted_dims[idx:] else 1 for i in range(self.dim)]
            self.sim_data[tuple(c_neighbors[dim]['max'])]['side_len'] = np.copy(self.sim_data[tuple(c_neighbors[dim]['min'])]['side_len'])
        # divide the current center last
        self.sim_data[tuple(c)]['side_len'][rev_sorted_dims] = 1 / 3 * self.sim_data[tuple(c)]['side_len'][rev_sorted_dims]
    
    
    """calculate the distance between a hyperrectangle center and its vertices"""
    def _calc_v2c(self, point):
        tuple_point = tuple(point) if isinstance(point, np.ndarray) else point
        return 1 / 2 * np.sqrt(np.sum(np.square(self.sim_data[tuple_point]['side_len'])))
    
    
    """identify potentially optimal hyperrectangles (one Mone Carlo experiment if noisy_direct is True)"""
    def _identify_c_once(self, mode='default', MC_num=None):
        pot_c, pot_c_pair, pot_c_extra = {}, {}, {}
        # loop through existing centers and record the center with the minimum objective function value for each center-to-vertex distance
        for tuple_point in self.sim_data:
            obj = self._obj_eval(tuple_point) if mode != 'MC' else self._obj_eval(tuple_point, enable_MC=True, random_seed=MC_num)
            v2c = self._calc_v2c(tuple_point)
            if v2c not in pot_c:
                pot_c[v2c] = {'c': tuple_point, 'obj': obj}
                pot_c_pair[v2c] = obj
            else:
                if obj < pot_c[v2c]['obj']:
                    pot_c[v2c] = {'c': tuple_point, 'obj': obj}
                    pot_c_pair[v2c] = obj
                elif obj == pot_c[v2c]['obj']: # record centers with the same minimum
                    pot_c_extra[tuple_point] = {'v2c': v2c, 'obj': obj}
        pot_c_pair = np.asarray([pair for pair in pot_c_pair.items()]) # convert dictionary to numpy array 
        pot_c_pair = pot_c_pair[np.argsort(pot_c_pair[:, 0])] # sort the array according to the center-to-vertex distance
        # find the convex hull only if the number of centers is greater than 2, otherwise all centers are along the convex hull
        if pot_c_pair.shape[0] > 2:
            hull = ConvexHull(pot_c_pair) 
            pot_c_pair = pot_c_pair[np.sort(hull.vertices), :] # find centers along the convex hull
        # loop through centers along the convex hull to check epsilon improvement on the objective function value
        current_f_min = self._obj_eval(self.x_min[-1]) # current minimum objective function value to compare
        flag = False
        for row_idx in range(pot_c_pair.shape[0] - 1):
            improve = (pot_c_pair[row_idx + 1, 1] - pot_c_pair[row_idx, 1]) / (pot_c_pair[row_idx + 1, 0] - pot_c_pair[row_idx, 0]) * pot_c_pair[row_idx, 0] # improvement is slope * abscissa
            if pot_c_pair[row_idx, 1] - improve <= current_f_min - self.eps * abs(current_f_min):
                opt_c = [pot_c[pot_c_pair[row, 0]]['c'] for row in range(row_idx, pot_c_pair.shape[0])]
                pot_c_pair = pot_c_pair[row_idx:]
                flag = True
                break
        # the last center is always potentially optimal
        if not flag:
            opt_c = [pot_c[pot_c_pair[-1, 0]]['c']]
            pot_c_pair = pot_c_pair[-1:]
        # check if centers with the same minimum are potentially optimal
        if pot_c_extra:
            for tuple_point, dict_ in pot_c_extra.items():
                if dict_['v2c'] in pot_c_pair[:, 0]:
                    if dict_['obj'] == pot_c_pair[pot_c_pair[:, 0] == dict_['v2c'], 1][0]:
                        opt_c.append(tuple_point)
        return opt_c
    
    
    """identify potentially optimal hyperrectangles"""
    def _identify_c(self, func):
        opt_c_default = self._identify_c_once()
        # if noisy_direct is True, perform Monte Carlo experiments to satisfy the overlap
        if not self.noisy_direct:
            return opt_c_default
        else:
            overlap = []
            c_to_eval = set()
            # loop through all Monte Carlo experiments and record overlap and difference
            for iter_ in range(self.num_MC):
                opt_c_MC = self._identify_c_once(mode='MC', MC_num=iter_)
                overlap.append(len(set(opt_c_default).intersection(opt_c_MC)) / len(opt_c_default))
                c_to_eval.union(set(opt_c_default).symmetric_difference(opt_c_MC))
            # if the average overlap is not large enough, perform extra function evaluations for all non-overlap centers
            if sum(overlap) / len(overlap) < 1 - self.sig_level:
                flag = False
                # loop through all non-overlap centers and check if they have reached the maximum number of function evaluations
                for c in list(c_to_eval):
                    if self.sim_data[c]['res'].size + self.add_eval_per_point <= self.max_eval_per_point:
                        self.sim_data[c]['res'] = np.append(self.sim_data[c]['res'], 
                                                            self._func_eval(c, self.add_eval_per_point, func))
                        self.total_func_eval += self.add_eval_per_point
                        flag = True
                if flag:
                    opt_c_default = self._identify_c(func) # recursive function call
            return opt_c_default
    
    
    """the main function to be called for minimization"""
    def minimize(self, func, lower_bounds, upper_bounds):
        self._check_bounds(lower_bounds, upper_bounds) # check input bounds
        # main loop with three termination criteria
        while (self.iter_ < self.max_iter) and (self.total_func_eval < self.max_total_func_eval) and (self.total_point_eval < self.max_total_point_eval):
            # initialize the center of the unit hypercube for iteration 1, for subsequent iterations find the potentially optimal hyperrectangles
            if self.iter_ == 0:
                opt_c = self._init_c(func).reshape(1, -1)
            else:
                opt_c = np.asarray(self._identify_c(func))
            # divide each potentially optimal hyperrectangle
            current_x_min = self.x_min[-1]
            for c in opt_c:
                c_neighbors = self._sort_c_neighbors(c, func)
                sorted_dims, current_x_min = self._sort_dims(c_neighbors, func, current_x_min)
                self._divide_c(c, c_neighbors, sorted_dims[::-1])
            self.x_min = np.concatenate((self.x_min, current_x_min.reshape(1, -1)), axis=0) # record the history of minima
            self.total_func_eval_series.append(self.total_func_eval)
            self.total_point_eval_series.append(self.total_point_eval)
            # display results for each iteration
            print("\niteration {}".format(self.iter_ + 1))
            print("   " + "current optimal x: {}".format(self._back_transform(self.x_min[-1])))
            print("   " + "current optimal y: {}".format(self._obj_eval(self.x_min[-1])))
            print("   " + "current func eval: {}".format(self.total_func_eval))
            self.iter_ += 1

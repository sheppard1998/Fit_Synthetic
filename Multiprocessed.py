#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:21:07 2019
Tests the credibility interval from orbits.py against true values to try and
get a coverage fraction similar to the percentage described by the interval. 
Prints these success rates and the statistics of the intervals over a certain 
amount of iterations.

Including multiprocessing.
@author: ssheppa1
"""

# Make division and print() behave like Python 3 even if 
# using Python 2.x:
from __future__ import print_function, division

# Import the key packages we need:
import numpy as np

import time as Time

import orbits

import multiprocessing as mp

import random

def print_min_max_avg(lower_arr, upper_arr, num_iters, var, unit=""):
    '''
    Create maxima, minima, and mean for upper bound, lower bound, and range of intervals then prints them
    Parameters: Arrays of the lower and upper bounds.
                The number of iterations the program ran as an int.
                The unit of the variable as a string.
                The name of the variable as a string.
    Returns: nothing, just calculates and prints
    '''
    low_min = np.min(lower_arr)
    low_max = np.max(lower_arr)
    low_avg = np.mean(lower_arr)
    high_min = np.min(upper_arr)
    high_max = np.max(upper_arr)
    high_avg = np.mean(upper_arr)
    range_min = np.min(upper_arr - lower_arr)
    range_max = np.max(upper_arr - lower_arr)
    range_avg = np.mean(upper_arr - lower_arr)
    print("For the %s:" %(var))
    print("--------------------------------------------------")
    print("The minimum lower bound of the interval is %0.3f %s after %d runs" %(low_min, unit, num_iters))
    print()
    print("The maximum lower bound of the interval is %0.3f %s after %d runs" %(low_max, unit, num_iters))
    print()
    print("The average lower bound of the interval is %0.3f %s after %d runs" %(low_avg, unit, num_iters))
    print()
    print("The minimum upper bound of the interval is %0.3f %s after %d runs" %(high_min, unit, num_iters))
    print()
    print("The maximum upper bound of the interval is %0.3f %s after %d runs" %(high_max, unit, num_iters))
    print()
    print("The average upper bound of the interval is %0.3f %s after %d runs" %(high_avg, unit, num_iters))
    print()
    print("The minimum range of the interval is %0.3f %s after %d runs" %(range_min, unit, num_iters))
    print()
    print("The maximum range of the interval is %0.3f %s after %d runs" %(range_max, unit, num_iters))
    print()
    print("The average range of the interval is %0.3f %s after %d runs" %(range_avg, unit, num_iters))
    print("--------------------------------------------------")
    
def computation(i, correct, uppers, lowers, lit_vals, seed):
    '''
    Parameters: i - Index of where in the shared lists to append the result
                correct - Shared list to increment and keep track of number of
                          correct results
                uppers - Shared list to append the intervals upper bounds
                lowers - Shared list to append the intervals lower bounds
                lit_vals - true values for orbital parameters as well as
                           crucial calculated theoretical positions and
                           uncertainties
                seed - random number between 0 and 100000 times the number of
                       iterations to ensure the machine does not generate the
                       same set of random numbers in each process
    '''
    np.random.seed(seed)
    lit_P = lit_vals[0]
    lit_T = lit_vals[1]
    lit_e = lit_vals[2]
    lit_a = lit_vals[3]
    lit_i = lit_vals[4]
    lit_w = lit_vals[5]
    lit_Omega = lit_vals[6]
    
    f_orb_Syn = lit_vals[7]
    
    x_errs = lit_vals[11]
    y_errs = lit_vals[12]
    times_obs = lit_vals[8]
        
    #Eq. 11 L14
    ra_obs_Syn = lit_vals[9] + np.random.normal(0, y_errs)
    dec_obs_Syn = lit_vals[10] + np.random.normal(0, x_errs)
    
    # Careful with our x-y coordinate system - not the same as RA-Dec!
    x_obs = dec_obs_Syn
    y_obs = ra_obs_Syn
    
    # Now that we have the data, find the best fit
    # orbit by searching over a range of parameters:
    
    # Get the start date of the data - we'll use
    # this to set what times of periastron we test:
    data_start = np.min(times_obs)
    
    # Set trial orbital elements over a range.
    # Careful specifying this number - the output grid is of size n**3
    # This takes about 5 seconds with n = 100; time should scale 
    # from there roughly as (5 seconds) * (n/100)**3
    n = 100
    
    # Call a routine to define a grid of search parameters. 
    # Default is to search all eccentricities and periastrons.  Periods
    # to search are passed as log(years), so 1 to 3 is periods of 10 to 
    # 1000 years, for example.  For this system, we have a better constraint 
    # on the period since we have most of the orbit. 
    
    e_max = 0.99
    
    logP_min = np.log10(lit_P*f_orb_Syn)
    logP_max = np.log10(1000)
    P_array, e_array, T_array = orbits.grid_P_e_T(n, logP_min, logP_max, T_start=data_start, e_max=e_max)
    
    
    # This is the routine that really does the optimization, returning parameters for 
    # *all* the orbits it tries, and their chi-squares: 
    A_array, B_array, F_array, G_array, sigma_list, chi_squared = orbits.Thiele_Innes_optimal(times_obs, P_array, e_array, \
                                                                                      T_array, x_obs, y_obs, \
                                                                                      x_errs, y_errs, debug=False)
    
    # Now optimize the grid a bit - only keep values within the bounds that give 
    # delta chi squared less than 10 from the best fit found so far: 
    best_chi_squared = np.min(chi_squared)
    delta_chi_squared = 21.85
    
    good_inds = np.where((chi_squared - best_chi_squared) < delta_chi_squared)
    
    e_min = np.min(e_array[good_inds])
    e_max = np.max(e_array[good_inds])
    
    logP_min = np.log10(np.min(P_array[good_inds]))
    logP_max = np.log10(np.max(P_array[good_inds]))
    
    tau_min = np.min((T_array[good_inds] - data_start)/P_array[good_inds])
    tau_max = np.max((T_array[good_inds] - data_start)/P_array[good_inds])
    
    # Now regrid with these bounds, and run grid search again: 
    P_array, e_array, T_array = orbits.grid_P_e_T(n, logP_min, logP_max, e_min, e_max, \
                                                      tau_min, tau_max, T_start=data_start)
    A_array, B_array, F_array, G_array, sigma_list, chi_squared = orbits.Thiele_Innes_optimal(times_obs, P_array, e_array, \
                                                                                      T_array, x_obs, y_obs, \
                                                                                      x_errs, y_errs, debug=False)
    
    # Then take these and get the other orbital parameters, too: 
    w_array, a_array, i_array, Omega_array = orbits.Campbell_from_Thiele_Innes(A_array, B_array, F_array, G_array)
        
    # Now get a more refined version of the mass posterior: 
    # Resample the above grid by an extra factor of N, following 
    # method in Lucy 2014B:
    
    N = 50
    
    w_N, a_N, i_N, T_N, e_N, P_N, Omega_N, new_likelihood, script_ABFG = orbits.correct_orbit_likelihood(\
                                                                                            P_array, e_array, \
                                                                                            T_array, A_array, \
                                                                                            B_array, F_array, \
                                                                                            G_array, sigma_list,\
                                                                                            chi_squared, N)
        
        
    # Get the credible interval for the semimajor axis: 
    a_mean, a_low, a_high = orbits.credible_interval(a_N, new_likelihood)
    
    # Get the credible interval for the inclination: 
    i_mean, i_low, i_high = orbits.credible_interval(i_N, new_likelihood)
    
    # Get the credible interval for the time of periastron passage: 
    T_mean, T_low, T_high = orbits.credible_interval(T_N, new_likelihood)
    
    # Get the credible interval for the semimajor axis: 
    e_mean, e_low, e_high = orbits.credible_interval(e_N, new_likelihood)
    
    # Get the credible interval for the period: 
    P_mean, P_low, P_high = orbits.credible_interval(P_N, new_likelihood)
    
    # Get the credible interval for the position angle of ascending node: 
    Omega_mean, Omega_low, Omega_high = orbits.credible_interval(Omega_N, new_likelihood)
    
    #Taking care of Omega offset that occurs in conversion to campbell elements
    if (np.deg2rad(lit_Omega) < 0):
        Omega_mean -= np.pi
        Omega_low -= np.pi
        Omega_high -= np.pi
    elif(np.deg2rad(lit_Omega) > np.pi):
        Omega_mean += np.pi
        Omega_low += np.pi
        Omega_high += np.pi
        
    # Get the credible interval for the longitude of periastron: 
    w_mean, w_low, w_high = orbits.credible_interval(w_N, new_likelihood)
    
    #Taking care of w offset that occurs in conversion to campbell elements
    if (np.deg2rad(lit_Omega) < 0):
        w_mean -= np.pi
        w_low -= np.pi
        w_high -= np.pi
    elif(np.deg2rad(lit_Omega) > np.pi):
        w_mean += np.pi
        w_low += np.pi
        w_high += np.pi
    
    #Returning to degrees
    w_mean = np.rad2deg(w_mean)
    w_low = np.rad2deg(w_low)
    w_high = np.rad2deg(w_high)
    
    Omega_mean = np.rad2deg(Omega_mean)
    Omega_low = np.rad2deg(Omega_low)
    Omega_high = np.rad2deg(Omega_high)
    
    i_mean = np.rad2deg(i_mean)
    i_low = np.rad2deg(i_low)
    i_high = np.rad2deg(i_high)
    
    #Comparison with true values
    #Remembering standard output of P, T, e, a, i, w, Omega
    if (lit_P > P_low and lit_P < P_high):
        correct[0] += 1
    if (lit_T > T_low and lit_T < T_high):
        correct[1] += 1
    if (lit_e > e_low and lit_e < e_high):
        correct[2] += 1
    if (lit_a > a_low and lit_a < a_high):
        correct[3] += 1
    if (lit_i > i_low and lit_i < i_high):
        correct[4] += 1
    if (lit_w > w_low and lit_w < w_high):
        correct[5] += 1 
    if (lit_Omega > Omega_low and lit_Omega < Omega_high):
        correct[6] += 1

    #Remembering standard output of P, T, e, a, i, w, Omega
    uppers[i][0] += P_high
    lowers[i][0] += P_low
    uppers[i][1] += T_high
    lowers[i][1] += T_low
    uppers[i][2] += e_high
    lowers[i][2] += e_low
    uppers[i][3] += a_high
    lowers[i][3] += a_low
    uppers[i][4] += i_high
    lowers[i][4] += i_low
    uppers[i][5] += w_high
    lowers[i][5] += w_low
    uppers[i][6] += Omega_high
    lowers[i][6] += Omega_low

if __name__ == '__main__':

    
    num_iters = int(input("Number of iterations: "))
    a_uppers = np.zeros(num_iters)
    a_lowers = np.zeros(num_iters)
    w_uppers = np.zeros(num_iters)
    w_lowers = np.zeros(num_iters)
    i_uppers = np.zeros(num_iters)
    i_lowers = np.zeros(num_iters)
    T_uppers = np.zeros(num_iters)
    T_lowers = np.zeros(num_iters)
    P_uppers = np.zeros(num_iters)
    P_lowers = np.zeros(num_iters)
    e_uppers = np.zeros(num_iters)
    e_lowers = np.zeros(num_iters)
    Omega_uppers = np.zeros(num_iters)
    Omega_lowers = np.zeros(num_iters)
    
    lit_P = 100
    tau_Syn = 0.4
    lit_e = 0.5
    lit_a = 1
    lit_i = 60
    lit_w = 250
    lit_Omega = 120
    lit_T = tau_Syn*lit_P
    
    f_orb_Syn = 0.6
    num_obs_Syn = 15
    times_obs_Syn = f_orb_Syn*lit_P*np.arange(num_obs_Syn)/(num_obs_Syn-1)
    
    A_Syn, B_Syn, F_Syn, G_Syn = orbits.Thiele_Innes_from_Campbell(np.radians(lit_w), lit_a, np.radians(lit_i), np.radians(lit_Omega))
    
    ra_theo_Syn, dec_theo_Syn = orbits.keplerian_xy_Thiele_Innes(times_obs_Syn, A_Syn, B_Syn, F_Syn, G_Syn, lit_T, lit_e, lit_P)
    
    ra_errs = 0.05*lit_a*np.ones(num_obs_Syn)
    dec_errs = 0.05*lit_a*np.ones(num_obs_Syn)
    
    lit_vals = [lit_P, lit_T, lit_e, lit_a, lit_i, lit_w, lit_Omega, f_orb_Syn, times_obs_Syn, \
                ra_theo_Syn, dec_theo_Syn, ra_errs, dec_errs]

    manager = mp.Manager()
    upper_list = manager.list([])
    lower_list = manager.list([])
    correct_list = manager.list([0,0,0,0,0,0,0])
    for i in range(num_iters):
        upper_list.append(manager.list([0,0,0,0,0,0,0]))
        lower_list.append(manager.list([0,0,0,0,0,0,0]))
    processes = []
    for i in range(num_iters):
        p = mp.Process(target=computation, args=(i, correct_list, upper_list, lower_list, lit_vals, random.randrange(100000*num_iters)))
        processes.append(p)
        if (i == 0):
            overall_start_time = Time.time()
        p.start()
    
    for i in range(num_iters):
        processes[i].join()
        if (i == num_iters - 1):
            end_time = Time.time()
            
    #upper_list = list(upper_list)
    #lower_list = list(lower_list)
    
    for i in range(num_iters):
        #upper_list[i] = list(upper_list[i])
        #lower_list[i] = list(lower_list[i])
        P_uppers[i] = upper_list[i][0]
        T_uppers[i] = upper_list[i][1]
        e_uppers[i] = upper_list[i][2]
        a_uppers[i] = upper_list[i][3]
        i_uppers[i] = upper_list[i][4]
        w_uppers[i] = upper_list[i][5]
        Omega_uppers[i] = upper_list[i][6]
        P_lowers[i] = lower_list[i][0]
        T_lowers[i] = lower_list[i][1]
        e_lowers[i] = lower_list[i][2]
        a_lowers[i] = lower_list[i][3]
        i_lowers[i] = lower_list[i][4]
        w_lowers[i] = lower_list[i][5]
        Omega_lowers[i] = lower_list[i][6]
    
    correct_P = correct_list[0]
    correct_T = correct_list[1]
    correct_e = correct_list[2]
    correct_a = correct_list[3]
    correct_i = correct_list[4]
    correct_w = correct_list[5]
    correct_Omega = correct_list[6]
    
    cov_frac_P = correct_P/num_iters
    print("Coverage fraction for period (P) stands at %0.3f over %d runs" %(cov_frac_P, num_iters))
    print()
    cov_frac_T = correct_T/num_iters
    print("Coverage fraction for time of periastron passage (T) stands at %0.3f over %d runs" %(cov_frac_T, num_iters))
    print()
    cov_frac_e = correct_e/num_iters
    print("Coverage fraction for eccentricity (e) stands at %0.3f over %d runs" %(cov_frac_e, num_iters))
    print()
    cov_frac_a = correct_a/num_iters
    print("Coverage fraction for semi major axis (a) stands at %0.3f over %d runs" %(cov_frac_a, num_iters))
    print()
    cov_frac_i = correct_i/num_iters
    print("Coverage fraction for inclination (i) stands at %0.3f over %d runs" %(cov_frac_i, num_iters))
    print()
    cov_frac_w = correct_w/num_iters
    print("Coverage fraction for longitude of periastron (w) stands at %0.3f over %d runs" %(cov_frac_w, num_iters))
    print()
    cov_frac_Omega = correct_Omega/num_iters
    print("Coverage fraction for position angle of ascending node (Omega) stands at %0.3f over %d runs" %(cov_frac_Omega, num_iters))
    print()
    avg_runtime = (end_time - overall_start_time)/num_iters
    print("The average runtime of one iteration stands at %0.3f seconds after %d runs" %(avg_runtime, num_iters))
    print("--------------------------------------------------")
    print()
    
    P_range = np.vstack((P_lowers, P_uppers)).T
    T_range = np.vstack((T_lowers, T_uppers)).T
    e_range = np.vstack((e_lowers, e_uppers)).T
    a_range = np.vstack((a_lowers, a_uppers)).T
    i_range = np.vstack((i_lowers, i_uppers)).T
    w_range = np.vstack((w_lowers, w_uppers)).T
    Omega_range = np.vstack((Omega_lowers, Omega_uppers)).T
    
    np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/P_Intervals_Synthetic_68.3.txt", P_range, fmt="%s")
    np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/T_Intervals_Synthetic_68.3.txt", T_range, fmt="%s")
    np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/e_Intervals_Synthetic_68.3.txt", e_range, fmt="%s")
    np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/a_Intervals_Synthetic_68.3.txt", a_range, fmt="%s")
    np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/i_Intervals_Synthetic_68.3.txt", i_range, fmt="%s")
    np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/w_Intervals_Synthetic_68.3.txt", w_range, fmt="%s")
    np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/Omega_Intervals_Synthetic_68.3.txt", Omega_range, fmt="%s")
    
    P_unit = "years"
    P_name = "period"
    print_min_max_avg(P_lowers, P_uppers, num_iters, P_name, P_unit)
    
    T_unit = "years"
    T_name = "time of periastron passage"
    print_min_max_avg(T_lowers, T_uppers, num_iters, T_name, T_unit)
    
    e_name = "eccentricity"
    print_min_max_avg(e_lowers, e_uppers, num_iters, e_name)

    a_unit = "arcseconds"
    a_name = "semi-major axis"
    print_min_max_avg(a_lowers, a_uppers, num_iters, a_name, a_unit)
    
    i_unit = "degrees"
    i_name = "inclination"
    print_min_max_avg(i_lowers, i_uppers, num_iters, i_name, i_unit)
    
    w_unit = "degrees"
    w_name = "longitude of periastron"
    print_min_max_avg(w_lowers, w_uppers, num_iters, w_name, w_unit)
    
    Omega_unit = "degrees"
    Omega_name = "position angle of ascending node"
    print_min_max_avg(Omega_lowers, Omega_uppers, num_iters, Omega_name, Omega_unit)
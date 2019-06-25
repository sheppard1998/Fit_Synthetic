#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:52:53 2019
Tests the credibility interval from orbits.py against true values to try and
get a coverage fraction similar to the percentage described by the interval. 
Prints these success rates and the statistics of the intervals over a certain 
amount of iterations.

Including threading.
@author: ssheppa1
"""

# Make division and print() behave like Python 3 even if 
# using Python 2.x:
from __future__ import print_function, division

# Import the key packages we need:
import numpy as np

import time as Time

import orbits

import threading

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
    
def computation(correct, runtime, lowers, uppers):
    #Eq.8 L14
    P_Syn = 100
    tau_Syn = 0.4
    e_Syn = 0.5
    a_Syn = 1
    i_Syn = np.radians(60)
    w_Syn = np.radians(250)
    Omega_Syn = np.radians(120)
    T_Syn = tau_Syn * P_Syn 
    A_Syn, B_Syn, F_Syn, G_Syn = orbits.Thiele_Innes_from_Campbell(w_Syn, a_Syn, i_Syn, Omega_Syn)
    
    f_orb_Syn = 0.6
    num_obs_Syn = 15
    times_obs_Syn = np.zeros(num_obs_Syn)
    times_obs_Syn = f_orb_Syn*P_Syn*np.arange(num_obs_Syn)/(num_obs_Syn-1)
    
    ra_theo_Syn, dec_theo_Syn = orbits.keplerian_xy_Thiele_Innes(times_obs_Syn, A_Syn, B_Syn, F_Syn, G_Syn, T_Syn, e_Syn, P_Syn)
    err_size = 0.05*a_Syn
    ra_errs_Syn = err_size*np.ones(num_obs_Syn)
    dec_errs_Syn = err_size*np.ones(num_obs_Syn)
    
    x_errs = dec_errs_Syn
    y_errs = ra_errs_Syn
    times_obs = times_obs_Syn
    
    # Measured values:
    #c.f. Fantino & Casotto pg. 11
    lit_a = a_Syn
    lit_i = np.rad2deg(i_Syn)
    lit_T = T_Syn
    lit_e = e_Syn
    lit_P = P_Syn
    lit_Omega = np.rad2deg(Omega_Syn)
    lit_w = np.rad2deg(w_Syn)
        
    #Eq. 11 L14
    ra_obs_Syn = ra_theo_Syn + np.random.normal(0, ra_errs_Syn)
    dec_obs_Syn = dec_theo_Syn + np.random.normal(0, dec_errs_Syn)
    
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
    
    logP_min = np.log10(70)
    logP_max = np.log10(130)
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
    if (Omega_Syn < 0):
        Omega_mean -= np.pi
        Omega_low -= np.pi
        Omega_high -= np.pi
    elif(Omega_Syn > np.pi):
        Omega_mean += np.pi
        Omega_low += np.pi
        Omega_high += np.pi
        
    # Get the credible interval for the longitude of periastron: 
    w_mean, w_low, w_high = orbits.credible_interval(w_N, new_likelihood)
    
    #Taking care of w offset that occurs in conversion to campbell elements
    if (Omega_Syn < 0):
        w_mean -= np.pi
        w_low -= np.pi
        w_high -= np.pi
    elif(Omega_Syn > np.pi):
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
    uppers[0] = P_high
    lowers[0] = P_low
    uppers[1] = T_high
    lowers[1] = T_low
    uppers[2] = e_high
    lowers[2] = e_low
    uppers[3] = a_high
    lowers[3] = a_low
    uppers[4] = i_high
    lowers[4] = i_low
    uppers[5] = w_high
    lowers[5] = w_low
    uppers[6] = Omega_high
    lowers[6] = Omega_low


def main():
    
    num_iters = int(input("Number of iterations: "))
    correct_a = 0
    correct_e = 0
    correct_i = 0
    correct_P = 0
    correct_T = 0
    correct_Omega = 0
    correct_w = 0
    runtimes = np.zeros(num_iters)
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
    
    #Make use of standard output of P, T, e, a, i, w, Omega
    result_correct = [0,0,0,0,0,0,0]
    runtime = [0]
    result_lowers = [0,0,0,0,0,0,0]
    result_uppers = [0,0,0,0,0,0,0]
    
    
    threads = []
    for i in range(num_iters):
        t = threading.Thread(target=computation, args=(result_correct, runtime, result_lowers, result_uppers))
        threads.append(t)
        if (i == 0):
            overall_start_time = Time.time()
        t.start()
    
    for i in range(num_iters):
        threads[i].join()
        if (i == num_iters - 1):
            end_time = Time.time()
        correct_P = result_correct[0]
        correct_T = result_correct[1]
        correct_e = result_correct[2]
        correct_a = result_correct[3]
        correct_i = result_correct[4]
        correct_w = result_correct[5]
        correct_Omega = result_correct[6]
        runtimes[i] = runtime[0]
        P_uppers[i] = result_uppers[0]
        T_uppers[i] = result_uppers[1]
        e_uppers[i] = result_uppers[2]
        a_uppers[i] = result_uppers[3]
        i_uppers[i] = result_uppers[4]
        w_uppers[i] = result_uppers[5]
        Omega_uppers[i] = result_uppers[6]
        P_lowers[i] = result_lowers[0]
        T_lowers[i] = result_lowers[1]
        e_lowers[i] = result_lowers[2]
        a_lowers[i] = result_lowers[3]
        i_lowers[i] = result_lowers[4]
        w_lowers[i] = result_lowers[5]
        Omega_lowers[i] = result_lowers[6]
    
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
    print(runtimes)
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
    
main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:32:13 2019
Tests the credibility interval from orbits.py against true values to try and
get a coverage fraction similar to the percentage described by the interval. 
Prints these success rates and the statistics of the intervals over a certain 
amount of iterations.
@author: ssheppa1
"""

# Make division and print() behave like Python 3 even if 
# using Python 2.x:
from __future__ import print_function, division

# Import the key packages we need:
import numpy as np
import matplotlib.pyplot as plt
# Make default font a bit bigger:
import matplotlib
matplotlib.rcParams['font.size'] = 18   # Font size in points

from astropy.table import Table
import time as Time

import orbits

import random

# Plots show in the notebook, not pop-up:
#%matplotlib inline

def print_min_max_avg(lower_arr, upper_arr, num_iters, var, int_prcnt, unit=""):
    '''
    Create maxima, minima, and mean for upper bound, lower bound, and range of intervals then prints them
    Parameters: Arrays of the lower and upper bounds.
                The number of iterations the program ran as an int.
                The unit of the variable as a string.
                The name of the variable as a string.
    Returns: nothing, just calculates and prints
    '''
    for i in range(len(int_prcnt)):
        low_min = np.min(lower_arr[i])
        low_max = np.max(lower_arr[i])
        low_avg = np.mean(lower_arr[i])
        high_min = np.min(upper_arr[i])
        high_max = np.max(upper_arr[i])
        high_avg = np.mean(upper_arr[i])
        range_min = np.min(upper_arr[i] - lower_arr[i])
        range_max = np.max(upper_arr[i] - lower_arr[i])
        range_avg = np.mean(upper_arr[i] - lower_arr[i])
        print("For the %0.1f interval for %s:" %(int_prcnt[i], var))
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
    
    
def main():
    
    num_iters = int(input("Number of iterations: "))
    print_every = int(input("Checkpoint amount of iterations: "))
    print()
    valid = False
    while (valid == False):
        num_int = input("Number of intervals to be calculated (max 3): ")
        if (float(num_int) % int(num_int) == 0.0):
            if (int(num_int) == 1):
                sig_int = np.array([1])
                valid = True
            elif (int(num_int) == 2):
                sig_int = np.array([1, 2])
                valid = True
            elif (int(num_int) == 3):
                sig_int = np.array([1, 2, 3])
                valid = True
            else:
                print("Invalid input")
                print()
                valid = False
        else:
            print("Invalid input")
            print()
            valid = False
    if (len(sig_int) == 1):
        int_prcnt = np.array([68.3])
    elif(len(sig_int) == 2):
        int_prcnt = np.array([68.3, 95.5])
    else:
        int_prcnt = np.array([68.3, 95.5, 99.7])
    iter_correct_a = np.zeros(len(sig_int))
    iter_correct_e = np.zeros(len(sig_int))
    iter_correct_i = np.zeros(len(sig_int))
    iter_correct_P = np.zeros(len(sig_int))
    iter_correct_T = np.zeros(len(sig_int))
    iter_correct_Omega = np.zeros(len(sig_int))
    iter_correct_w = np.zeros(len(sig_int))
    iters_comp = 0
    runtimes = np.zeros(num_iters)
    a_uppers = np.zeros((len(sig_int), num_iters))
    a_lowers = np.zeros((len(sig_int), num_iters))
    w_uppers = np.zeros((len(sig_int), num_iters))
    w_lowers = np.zeros((len(sig_int), num_iters))
    i_uppers = np.zeros((len(sig_int), num_iters))
    i_lowers = np.zeros((len(sig_int), num_iters))
    T_uppers = np.zeros((len(sig_int), num_iters))
    T_lowers = np.zeros((len(sig_int), num_iters))
    P_uppers = np.zeros((len(sig_int), num_iters))
    P_lowers = np.zeros((len(sig_int), num_iters))
    e_uppers = np.zeros((len(sig_int), num_iters))
    e_lowers = np.zeros((len(sig_int), num_iters))
    Omega_uppers = np.zeros((len(sig_int), num_iters))
    Omega_lowers = np.zeros((len(sig_int), num_iters))
    
    for k in range(num_iters):
        if(iters_comp % print_every == 0 and iters_comp != 0):
            cov_frac_P = iter_correct_P/iters_comp
            cov_frac_T = iter_correct_T/iters_comp
            cov_frac_e = iter_correct_e/iters_comp
            cov_frac_a = iter_correct_a/iters_comp
            cov_frac_i = iter_correct_i/iters_comp
            cov_frac_w = iter_correct_w/iters_comp
            cov_frac_Omega = iter_correct_Omega/iters_comp
            avg_runtime = np.mean(runtimes[:k])
            print("--------------------------------------------------")
            for j in range(len(sig_int)):
                print("For the %0.1f interval:" %(int_prcnt[j]))
                print("--------------------------------------------------")
                print("Coverage fraction for period (P) stands at %0.3f over %d runs" %(cov_frac_P[j], iters_comp))
                print()
                print("Coverage fraction for time of periastron passage (T) stands at %0.3f over %d runs" %(cov_frac_T[j], iters_comp))
                print()
                print("Coverage fraction for eccentricity (e) stands at %0.3f over %d runs" %(cov_frac_e[j], iters_comp))
                print()
                print("Coverage fraction for semi major axis (a) stands at %0.3f over %d runs" %(cov_frac_a[j], iters_comp))
                print()
                print("Coverage fraction for inclination (i) stands at %0.3f over %d runs" %(cov_frac_i[j], iters_comp))
                print()
                print("Coverage fraction for longitude of periastron (w) stands at %0.3f over %d runs" %(cov_frac_w[j], iters_comp))
                print()
                print("Coverage fraction for position angle of ascending node (Omega) stands at %0.3f over %d runs" %(cov_frac_Omega[j], iters_comp))
                print()
                print("The average runtime of one iteration stands at %0.3f seconds after %d runs" %(avg_runtime, iters_comp))
                print("--------------------------------------------------")
            print()
                
            
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
        
        f_orb_Syn = 0.4
        num_obs_Syn = 15
        times_obs_Syn = np.zeros(15)
        times_obs_Syn = f_orb_Syn*P_Syn*np.arange(num_obs_Syn)/(num_obs_Syn-1)
    
        ra_theo_Syn, dec_theo_Syn = orbits.keplerian_xy_Thiele_Innes(times_obs_Syn, A_Syn, B_Syn, F_Syn, G_Syn, T_Syn, e_Syn, P_Syn)
        ra_errs_Syn = 0.075*ra_theo_Syn
        dec_errs_Syn = 0.075*dec_theo_Syn
    
        #Eq. 11 L14
        ra_obs_Syn = ra_theo_Syn + np.random.normal(0, abs(ra_errs_Syn))
        dec_obs_Syn = dec_theo_Syn + np.random.normal(0, abs(dec_errs_Syn))
    
        overall_start_time = Time.time()
        
        # Name of the system we're fitting: 
        star_name = "Synthetic"
    
    
        # Careful with our x-y coordinate system - not the same as RA-Dec!
        x_obs = dec_obs_Syn
        y_obs = ra_obs_Syn
        x_errs = dec_errs_Syn
        y_errs = ra_errs_Syn
        times_obs = times_obs_Syn
        
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
        delta_chi_squared = 10
    
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
    
        # Not using a mass prior so just set to 1:
        mass_prior = 1
    
        # Calculate reduced chi-squared: 
        reduced_chi_squared = chi_squared/(times_obs.size - 3)
    
        # Likelihood of a given model is exp(-chi_squared/2); calculate while
        # also taking the prior into account:
        likelihood = np.exp(-0.5*reduced_chi_squared) * mass_prior
        
        # Now get a more refined version of the mass posterior: 
        # Resample the above grid by an extra factor of N, following 
        # method in Lucy 2014B:
    
        N = 20
    
        w_N, a_N, i_N, T_N, e_N, P_N, Omega_N, new_likelihood, script_ABFG = orbits.correct_orbit_likelihood(\
                                                                                            P_array, e_array, \
                                                                                            T_array, A_array, \
                                                                                            B_array, F_array, \
                                                                                            G_array, sigma_list,\
                                                                                            chi_squared, N)
        
        # Get the credible interval for the semimajor axis: 
        a_mean, a_low, a_high = orbits.credible_interval(a_N, new_likelihood, sig_int)
        
        # Get the credible interval for the inclination: 
        i_mean, i_low, i_high = orbits.credible_interval(i_N, new_likelihood, sig_int)
        
        # Get the credible interval for the time of periastron passage: 
        T_mean, T_low, T_high = orbits.credible_interval(T_N, new_likelihood, sig_int)
        
        # Get the credible interval for the semimajor axis: 
        e_mean, e_low, e_high = orbits.credible_interval(e_N, new_likelihood, sig_int)
        
        # Get the credible interval for the period: 
        P_mean, P_low, P_high = orbits.credible_interval(P_N, new_likelihood, sig_int)
        
        # Get the credible interval for the position angle of ascending node: 
        Omega_mean, Omega_low, Omega_high = orbits.credible_interval(Omega_N, new_likelihood, sig_int)
        
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
        w_mean, w_low, w_high = orbits.credible_interval(w_N, new_likelihood, sig_int)
        
        #Taking care of w offset that occurs in conversion to campbell elements
        if (Omega_Syn < 0):
            w_mean -= np.pi
            w_low -= np.pi
            w_high -= np.pi
        elif(Omega_Syn > np.pi):
            w_mean += np.pi
            w_low += np.pi
            w_high += np.pi
        
        # Measured values:
        #c.f. Fantino & Casotto pg. 11
        lit_a = a_Syn
        lit_i = np.rad2deg(i_Syn)
        lit_T = T_Syn
        lit_e = e_Syn
        lit_P = P_Syn
        lit_Omega = np.rad2deg(Omega_Syn)
        lit_w = np.rad2deg(w_Syn)
        
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
        for i in range(len(sig_int)):
            if (lit_a > a_low[i] and lit_a < a_high[i]):
                iter_correct_a[i] += 1    
            if (lit_i > i_low[i] and lit_i < i_high[i]):
                iter_correct_i[i] += 1    
            if (lit_T > T_low[i] and lit_T < T_high[i]):
                iter_correct_T[i] += 1    
            if (lit_e > e_low[i] and lit_e < e_high[i]):
                iter_correct_e[i] += 1    
            if (lit_P > P_low[i] and lit_P < P_high[i]):
                iter_correct_P[i] += 1    
            if (lit_Omega > Omega_low[i] and lit_Omega < Omega_high[i]):
                iter_correct_Omega[i] += 1    
            if (lit_w > w_low[i] and lit_w < w_high[i]):
                iter_correct_w[i] += 1
        
        end_time = Time.time()
        runtimes[k] = end_time - overall_start_time
        for j in range(len(sig_int)):
            a_uppers[j][k] = a_high[j]
            a_lowers[j][k] = a_low[j]
            w_uppers[j][k] = w_high[j]
            w_lowers[j][k] = w_low[j]
            i_uppers[j][k] = i_high[j]
            i_lowers[j][k] = i_low[j]
            T_uppers[j][k] = T_high[j]
            T_lowers[j][k] = T_low[j]
            P_uppers[j][k] = P_high[j]
            P_lowers[j][k] = P_low[j]
            e_uppers[j][k] = e_high[j]
            e_lowers[j][k] = e_low[j]
            Omega_uppers[j][k] = Omega_high[j]
            Omega_lowers[j][k] = Omega_low[j]
        
        iters_comp += 1
    
    cov_frac_P = iter_correct_P/iters_comp
    cov_frac_T = iter_correct_T/iters_comp
    cov_frac_e = iter_correct_e/iters_comp
    cov_frac_a = iter_correct_a/iters_comp
    cov_frac_i = iter_correct_i/iters_comp
    cov_frac_w = iter_correct_w/iters_comp
    cov_frac_Omega = iter_correct_Omega/iters_comp
    avg_runtime = np.mean(runtimes)
    print("--------------------------------------------------")
    for j in range(len(sig_int)):
        print("For the %0.1f interval:" %(int_prcnt[j]))
        print("--------------------------------------------------")
        print("Coverage fraction for period (P) stands at %0.3f over %d runs" %(cov_frac_P[j], iters_comp))
        print()
        print("Coverage fraction for time of periastron passage (T) stands at %0.3f over %d runs" %(cov_frac_T[j], iters_comp))
        print()
        print("Coverage fraction for eccentricity (e) stands at %0.3f over %d runs" %(cov_frac_e[j], iters_comp))
        print()
        print("Coverage fraction for semi major axis (a) stands at %0.3f over %d runs" %(cov_frac_a[j], iters_comp))
        print()
        print("Coverage fraction for inclination (i) stands at %0.3f over %d runs" %(cov_frac_i[j], iters_comp))
        print()
        print("Coverage fraction for longitude of periastron (w) stands at %0.3f over %d runs" %(cov_frac_w[j], iters_comp))
        print()
        print("Coverage fraction for position angle of ascending node (Omega) stands at %0.3f over %d runs" %(cov_frac_Omega[j], iters_comp))
        print()
        print("The average runtime of one iteration stands at %0.3f seconds after %d runs" %(avg_runtime, iters_comp))
        print("--------------------------------------------------")
    print()
    
    ###########################################################################
    ##MUST IMPLEMENT MULTIPLE FUNCTIONALITY DOWNWARDS FROM HERE
    ###########################################################################
    P_range = np.zeros((len(sig_int),2))
    T_range = np.zeros((len(sig_int),2))
    e_range = np.zeros((len(sig_int),2))
    a_range = np.zeros((len(sig_int),2))
    i_range = np.zeros((len(sig_int),2))
    w_range = np.zeros((len(sig_int),2))
    Omega_range = np.zeros((len(sig_int),2))
    for i in range(len(sig_int)):        
        if (int_prcnt[i] == 68.3):
            P_range = np.vstack((P_lowers[i], P_uppers[i])).T
            T_range = np.vstack((T_lowers[i], T_uppers[i])).T
            e_range = np.vstack((e_lowers[i], e_uppers[i])).T
            a_range = np.vstack((a_lowers[i], a_uppers[i])).T
            i_range = np.vstack((i_lowers[i], i_uppers[i])).T
            w_range = np.vstack((w_lowers[i], w_uppers[i])).T
            Omega_range = np.vstack((Omega_lowers[i], Omega_uppers[i])).T
            
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/P_Intervals_Synthetic_68.3.txt", P_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/T_Intervals_Synthetic_68.3.txt", T_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/e_Intervals_Synthetic_68.3.txt", e_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/a_Intervals_Synthetic_68.3.txt", a_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/i_Intervals_Synthetic_68.3.txt", i_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/w_Intervals_Synthetic_68.3.txt", w_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/Omega_Intervals_Synthetic_68.3.txt", Omega_range, fmt="%s")
        
        elif (int_prcnt[i] == 95.5):
            P_range = np.vstack((P_lowers[i], P_uppers[i])).T
            T_range = np.vstack((T_lowers[i], T_uppers[i])).T
            e_range = np.vstack((e_lowers[i], e_uppers[i])).T
            a_range = np.vstack((a_lowers[i], a_uppers[i])).T
            i_range = np.vstack((i_lowers[i], i_uppers[i])).T
            w_range = np.vstack((w_lowers[i], w_uppers[i])).T
            Omega_range = np.vstack((Omega_lowers[i], Omega_uppers[i])).T
            
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/P_Intervals_Synthetic_95.5.txt", P_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/T_Intervals_Synthetic_95.5.txt", T_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/e_Intervals_Synthetic_95.5.txt", e_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/a_Intervals_Synthetic_95.5.txt", a_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/i_Intervals_Synthetic_95.5.txt", i_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/w_Intervals_Synthetic_95.5.txt", w_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/Omega_Intervals_Synthetic_95.5.txt", Omega_range, fmt="%s")
        
        elif (int_prcnt[i] == 99.7):
            P_range = np.vstack((P_lowers[i], P_uppers[i])).T
            T_range = np.vstack((T_lowers[i], T_uppers[i])).T
            e_range = np.vstack((e_lowers[i], e_uppers[i])).T
            a_range = np.vstack((a_lowers[i], a_uppers[i])).T
            i_range = np.vstack((i_lowers[i], i_uppers[i])).T
            w_range = np.vstack((w_lowers[i], w_uppers[i])).T
            Omega_range = np.vstack((Omega_lowers[i], Omega_uppers[i])).T
            
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/P_Intervals_Synthetic_99.7.txt", P_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/T_Intervals_Synthetic_99.7.txt", T_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/e_Intervals_Synthetic_99.7.txt", e_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/a_Intervals_Synthetic_99.7.txt", a_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/i_Intervals_Synthetic_99.7.txt", i_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/w_Intervals_Synthetic_99.7.txt", w_range, fmt="%s")
            np.savetxt("/Users/ssheppa1/Documents/Notebooks/Fit_Synthetic/Intervals/Omega_Intervals_Synthetic_99.7.txt", Omega_range, fmt="%s")
            
    P_unit = "years"
    P_name = "period"
    print_min_max_avg(P_lowers, P_uppers, num_iters, P_name, int_prcnt, P_unit)
    
    T_unit = "years"
    T_name = "time of periastron passage"
    print_min_max_avg(T_lowers, T_uppers, num_iters, T_name, int_prcnt, T_unit)
    
    e_name = "eccentricity"
    print_min_max_avg(e_lowers, e_uppers, num_iters, e_name, int_prcnt)
    
    a_unit = "arcseconds"
    a_name = "semi-major axis"
    print_min_max_avg(a_lowers, a_uppers, num_iters, a_name, int_prcnt, a_unit)
    
    i_unit = "degrees"
    i_name = "inclination"
    print_min_max_avg(i_lowers, i_uppers, num_iters, i_name, int_prcnt, i_unit)
    
    w_unit = "degrees"
    w_name = "longitude of periastron"
    print_min_max_avg(w_lowers, w_uppers, num_iters, w_name, int_prcnt, w_unit)
    
    Omega_unit = "degrees"
    Omega_name = "position angle of ascending node"
    print_min_max_avg(Omega_lowers, Omega_uppers, num_iters, Omega_name, int_prcnt, Omega_unit)
    
main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:50:28 2019
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
    
    
def main():
    
    num_iters = int(input("Number of iterations: "))
    print_every = int(input("Checkpoint amount of iterations: "))
    iter_correct_a = 0
    iter_correct_e = 0
    iter_correct_i = 0
    iter_correct_P = 0
    #iter_correct_T = 0
    iter_correct_Omega = 0
    iter_correct_w = 0
    iters_comp = 0
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
    
    for k in range(num_iters):
        if(iters_comp % print_every == 0 and iters_comp != 0):
            print("Most recent credibility interval guesses: ")
            print("--------------------------------------------------")
            print("Period (P): %0.3f to %0.3f years" %(P_low, P_high))
            print()
            print("Time of periastron passage (T): %0.3f to %0.3f years" %(T_low, T_high))
            print()
            print("Eccentricity (e): %0.3f to %0.3f" %(e_low, e_high))
            print()
            print("Semi major axis (a): %0.3f to %0.3f arcseconds" %(a_low, a_high))
            print()
            print("Inclination (i): %0.3f to %0.3f degrees" %(i_low, i_high))
            print()
            print("Longitude of periastron (w): %0.3f to %0.3f degrees" %(w_low, w_high))
            print()
            print("Position angle of ascending node (Omega): %0.3f to %0.3f degrees" %(Omega_low, Omega_high))
            print("--------------------------------------------------")
            print()
            cov_frac_P = iter_correct_P/iters_comp
            print("Coverage fraction for period (P) stands at %0.3f over %d runs" %(cov_frac_P, iters_comp))
            print()
            #cov_frac_T = iter_correct_T/iters_comp
            #print("Coverage fraction for time of periastron passage (T) stands at %0.3f over %d runs" %(cov_frac_T, iters_comp))
            #print()
            cov_frac_e = iter_correct_e/iters_comp
            print("Coverage fraction for eccentricity (e) stands at %0.3f over %d runs" %(cov_frac_e, iters_comp))
            print()
            cov_frac_a = iter_correct_a/iters_comp
            print("Coverage fraction for semi major axis (a) stands at %0.3f over %d runs" %(cov_frac_a, iters_comp))
            print()
            cov_frac_i = iter_correct_i/iters_comp
            print("Coverage fraction for inclination (i) stands at %0.3f over %d runs" %(cov_frac_i, iters_comp))
            print()
            cov_frac_w = iter_correct_w/iters_comp
            print("Coverage fraction for longitude of periastron (w) stands at %0.3f over %d runs" %(cov_frac_w, iters_comp))
            print()
            cov_frac_Omega = iter_correct_Omega/iters_comp
            print("Coverage fraction for position angle of ascending node (Omega) stands at %0.3f over %d runs" %(cov_frac_Omega, iters_comp))
            print()
            avg_runtime = np.mean(runtimes[:k])
            print("The average runtime of one iteration stands at %0.3f seconds after %d runs" %(avg_runtime, iters_comp))
            print("--------------------------------------------------")
            print()
        
        overall_start_time = Time.time()
        
        # Distance to the system in pc:
        d_pc = 26.1  # cf. Dupuy et al.  
        # Name of the system we're fitting: 
        star_name = "SDSS J105213.51+442255.7"
        
        # Read in orbital data, which we have saved in a file. 
        # Positions are in milliarcsecond units.  Different conversions
        # might be needed if your data are in different units.
        datafile = "SDSS_J1052.txt"
        t = Table.read(datafile, format='ascii.commented_header')
        
        # Convert the position angle and separation to RA and Dec separation: 
        ra_obs, dec_obs, ra_errs, dec_errs = orbits.rho_PA_to_RA_Dec( t['rho'],t['PA'], \
                                                                      t['rho_err'], t['PA_err'])
        
        # Careful with our x-y coordinate system - not the same as RA-Dec!
        x_obs = dec_obs + np.random.normal(0, abs(dec_errs))
        y_obs = ra_obs + np.random.normal(0, abs(ra_errs))
        x_errs = dec_errs
        y_errs = ra_errs
        
        # Code below assumes dates in years.  Convert if necessary. 
        times_obs = t['Date']
        
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
    
        logP_min = np.log10(5)
        logP_max = np.log10(10)
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

        # Get the credible interval for the longitude of periastron: 
        w_mean, w_low, w_high = orbits.credible_interval(w_N, new_likelihood)
  
        # Measured values:
        #c.f. Dupuy et al.
        lit_a = 70.59
        lit_i = 62
        #No lit_T as Dupuy et al. doesn't specify one to test against 
        lit_e = 0.1387
        lit_P = 8.614
        lit_Omega = 126.7
        lit_w = 186.5
        
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
        if (lit_a > a_low and lit_a < a_high):
            iter_correct_a += 1    
        if (lit_i > i_low and lit_i < i_high):
            iter_correct_i += 1    
        #if (lit_T > T_low and lit_T < T_high):
            #iter_correct_T += 1    
        if (lit_e > e_low and lit_e < e_high):
            iter_correct_e += 1    
        if (lit_P > P_low and lit_P < P_high):
            iter_correct_P += 1    
        if (lit_Omega > Omega_low and lit_Omega < Omega_high):
            iter_correct_Omega += 1    
        if (lit_w > w_low and lit_w < w_high):
            iter_correct_w += 1
        
        end_time = Time.time()
        runtimes[k] = end_time - overall_start_time
        a_uppers[k] = a_high
        a_lowers[k] = a_low
        w_uppers[k] = w_high
        w_lowers[k] = w_low
        i_uppers[k] = i_high
        i_lowers[k] = i_low
        T_uppers[k] = T_high
        T_lowers[k] = T_low
        P_uppers[k] = P_high
        P_lowers[k] = P_low
        e_uppers[k] = e_high
        e_lowers[k] = e_low
        Omega_uppers[k] = Omega_high
        Omega_lowers[k] = Omega_low
        
        iters_comp += 1
    
    cov_frac_P = iter_correct_P/num_iters
    print("Coverage fraction for period (P) stands at %0.3f over %d runs" %(cov_frac_P, num_iters))
    print()
    #cov_frac_T = iter_correct_T/num_iters
    #print("Coverage fraction for time of periastron passage (T) stands at %0.3f over %d runs" %(cov_frac_T, num_iters))
    #print()
    cov_frac_e = iter_correct_e/num_iters
    print("Coverage fraction for eccentricity (e) stands at %0.3f over %d runs" %(cov_frac_e, num_iters))
    print()
    cov_frac_a = iter_correct_a/num_iters
    print("Coverage fraction for semi major axis (a) stands at %0.3f over %d runs" %(cov_frac_a, num_iters))
    print()
    cov_frac_i = iter_correct_i/num_iters
    print("Coverage fraction for inclination (i) stands at %0.3f over %d runs" %(cov_frac_i, num_iters))
    print()
    cov_frac_w = iter_correct_w/num_iters
    print("Coverage fraction for longitude of periastron (w) stands at %0.3f over %d runs" %(cov_frac_w, num_iters))
    print()
    cov_frac_Omega = iter_correct_Omega/num_iters
    print("Coverage fraction for position angle of ascending node (Omega) stands at %0.3f over %d runs" %(cov_frac_Omega, num_iters))
    print()
    avg_runtime = np.mean(runtimes[:k])
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
    
    np.savetxt("P_Intervals_J1052.txt", P_range, fmt="%s")
    np.savetxt("T_Intervals_J1052.txt", T_range, fmt="%s")
    np.savetxt("e_Intervals_J1052.txt", e_range, fmt="%s")
    np.savetxt("a_Intervals_J1052.txt", a_range, fmt="%s")
    np.savetxt("i_Intervals_J1052.txt", i_range, fmt="%s")
    np.savetxt("w_Intervals_J1052.txt", w_range, fmt="%s")
    np.savetxt("Omega_Intervals_J1052.txt", Omega_range, fmt="%s")
    
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
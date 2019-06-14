# Make division and print() behave like Python 3 even if 
# using Python 2.x:
from __future__ import print_function, division

# Import the key packages we need:
import numpy as np

def Thiele_Innes_from_Campbell(w, a, i, Omega):
    '''Given the four Campbell orbital parameters as input:
    w(little omega): argument of periastron, radians
    a: semimajor axis
    i: orbital inclination, radians
    Omega: longitude of ascending node, radians
    
    as input, calculate and return the four Thiele-Innes parameters
    A, B, F, and G. These will be returned with the same units as a.
    Inputs can be either scalars or numpy arrays.'''
    
    from numpy import cos,sin
    
    # We use the various trig calls multiple times, so just
    # calculate them once for efficiency;
    cos_w = cos(w)
    sin_w = sin(w)
    cos_Omega = cos(Omega)
    sin_Omega = sin(Omega)
    cos_i = cos(i)
    
    A = a * (cos_w*cos_Omega - sin_w*sin_Omega*cos_i)
    B = a * (cos_w*sin_Omega + sin_w*cos_Omega*cos_i)
    F = a * (-sin_w*cos_Omega - cos_w*sin_Omega*cos_i)
    G = a * (-sin_w*sin_Omega + cos_w*cos_Omega*cos_i)
    
    return A, B, F, G



def getE_parallel(M, e):
    """
    Solve Kepler's Equation for the "eccentric anomaly", E.
    This function calculates an approximate solution for Kepler's Equation:
        M = E - e * sin(E),
    where M is the "mean anomaly" and E is the "eccentric anomaly".
    The implementation follows the prescription given by
    Markley (Markley 1995, CeMDA, 63, 101).
    Because it works without iteration, it can take inputs that
    are numpy arrays, and calculate multiple values at once.
    This approximation should be good to a precision of
    better than 1E-15 (though you should test that). 
    
    Parameters
    ----------
    M : numpy array or float
        Mean anomaly.
    e : numpy array or float, same dimension as M
        Eccentricity
    
    Returns
    -------
    Eccentric anomaly: numpy array or float, same dimension as M and e
        The solution(s) of Kepler's Equation for the input value(s)
        
        
    This is a translation of the Kepler's equation solver from the PyAstronomy
    package, to allow it to be a standalone subtroutine rather than a
    class with methods, and also to allow it to work in parallel, i.e.
    to find E for inputs that are full numpy arrays, without using a for
    loop.

    ELNJ 2016-11-04, translated from code in keplerOrbit.py from
    ------------------------------------------------------------------------------------------
    Copyright (c) 2011, PyA group

    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify, merge,
    publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
    to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies
    or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    PyAstronomy 0.10.0
    ------------------------------------------------------------------------------------------
    """

    pi2 = np.pi**2

    # Since we do some list-oriented operations that fail on scalars, 
    # make sure we're dealing with arrays instead: 
    if (hasattr(M, "__len__")):
        is_array = True
    else:
        is_array = False
    
    # For the mean anomaly, use values between
    # -pi and pi.
    M = M - (np.floor(M/(2.*np.pi)) * 2.*np.pi)
    # Flag values greater than pi, which we'll flip
    # around to the -pi to 0 range: 
    flip = (M > np.pi)
    if is_array:
        M[flip] = 2.*np.pi - M[flip]
        # Flag values that have M of 0 on input; we'll return zero for these on output
        # but we temporarily change their values so they aren't singular in the computations: 
        zero_vals = (M == 0)
        M[zero_vals] = 1E-15
    else:
        if flip:
            M = 2.*np.pi - M
        if M==0:
            return 0
    # Now calculate the quantities that go into the solution.  These are labeled
    # with their equation numbers from the source paper cited above. 
        
    # Eq. 20
    alpha = ( 3.*pi2 + 1.6*np.pi*(np.pi-np.abs(M))/(1.+e) )/(pi2 - 6.)
    # Eq. 5
    d = 3.*(1. - e) + alpha*e

    # Eq. 10
    r = 3.*alpha*d * (d-1.+e)*M + M**3

    # Eq. 9
    q = 2.*alpha*d*(1.-e) - M**2

    # Eq. 14
    w = (np.abs(r) + np.sqrt(q**3 + r**2))**(2./3.)

    # Eq. 15
    E1 = ( 2.*r*w/(w**2 + w*q + q**2) + M ) / d

    #  Eq. 21, 25, 26, 27, and 28 (f, f', f'', f''', and f'''')
    f0 = E1 - e*np.sin(E1) - M
    f1 = 1. - e*np.cos(E1)
    f2 = e*np.sin(E1)
    f3 = 1.-f1
    f4 = -f2

    # Eq. 22 
    d3 = -f0/(f1 - 0.5*f0*f2/f1)

    # Eq. 23
    d4 = -f0/( f1 + 0.5*d3*f2 + (d3**2)*f3/6. )

    # Eq. 24
    d5 = -f0/( f1 + 0.5*d4*f2 + d4**2*f3/6. + d4**3*f4/24.)

    # Eq. 29
    E5 = E1 + d5
    # Flip back those we flipped initially:
    if is_array:
        E5[flip] = 2.*np.pi - E5[flip]
        E5[zero_vals] = 0
        M[zero_vals] = 0
    elif flip:
        # Not an array, just one value to flip:
        E5 = 2.*np.pi - E5
        # Don't need to worry about zero M values here since if M 
        # was a single value of zero, we already exited above. 
    
    # Return the final value: 
    return E5


def Campbell_from_Thiele_Innes(A, B, F, G):
    '''Given the Thiele-Innes elements A, B, F, G as input,
    return the values of the Campbell elements w, a, i, and Omega, 
    with the angles in radians and semimajor axis a in the same
    units as the input parameters.
    
    Inputs can be either all numpy arrays of the same dimensions, or
    all scalars.'''
    
    import numpy as np

    # Whether to print some additional debug information
    # and do some assertion testing:
    debug = True
    
    w_plus_Omega = np.arctan2((B-F),(A+G))
    w_minus_Omega = np.arctan2((-B-F),(A-G))
#    w_plus_Omega = np.arctan((B-F)/(A+G))
#    w_minus_Omega = np.arctan((-B-F)/(A-G))

    if debug:
        # Check that the above arctan2 function returns values that
        # meet the proper conditions about quadrant of these
        # variables:
        sign_B_minus_F = (B-F > 0)
        sign_minus_B_minus_F = (-B-F > 0)
        sign_sin_w_plus_Omega = (np.sin(w_plus_Omega) > 0)
        sign_sin_w_minus_Omega = (np.sin(w_minus_Omega) > 0)

        assert np.array_equal(sign_B_minus_F, sign_sin_w_plus_Omega), \
          "Not all B_minus_F same sign as all sin(w + Omega)"
        assert np.array_equal(sign_minus_B_minus_F, sign_sin_w_minus_Omega), \
          "Not all -B-F same sign as all sin(w - Omega)"
        
    w = 0.5 * (w_plus_Omega + w_minus_Omega)
    Omega = 0.5 * (w_plus_Omega - w_minus_Omega)

    # Because of the range of the arctan2 function (from -pi to pi),
    # after taking the sums and differences we could end up with
    # values of these from -2pi to 2pi.  Wrap the negative ones into
    # the positive range:
    w_neg = w < 0
    w[w_neg] += 2. * np.pi

    Omega_neg = Omega < 0
    Omega[Omega_neg] += 2. * np.pi
    
    
    #!!!  Need to update comments below. 
    
    # Return only values in the range 0, pi for Omega, and
    # w (little omega).  Note that this is slightly different 
    # from what is given in Lucy 2014 (Eqs. A.11, A.12 and following
    # text) where he restricts these to the range of 0 to pi.
    # We adopt 0 to 2 pi here since that is the full range allowed in
    # the definition of these parameters, even if visual orbits alone
    # don't allow us to determine which side of pi is the right
    # value.  Choosing this convention allows more straightforward
    # comparison with other orbital fitting methods (e.g. MCMC).

    
    # To do this we need to handle it differently for arrays
    # vs. scalars: 
    
    is_array = hasattr(A, "__len__")
    
    if is_array:
        # Per discussions in L14 Eqs. A.11 and A.12, if Omega is out
        # of the range 0 to pi, we shift it, and change values of w
        # accordingly so that the sums and differences of Omega and w
        # are preserved:
        
        neg = Omega < 0
        Omega[neg] +=  np.pi
        w[neg] +=  np.pi

        # Now deal with the too-large values: 
        big = Omega >= np.pi
        Omega[big] -=  np.pi
        w[big] -=  np.pi

        # Now having done this, we might have pushed
        # w out of the 0 to 2 pi range in some cases, so we need to
        # address that possibility:

        neg_w = w < 0
        w[neg_w] += 2. * np.pi

        big_w = w >= 2. * np.pi
        w[big_w] -= 2. * np.pi

        

    else:
        # Only have scalars, no indexing:
        if Omega < 0:
            Omega += np.pi
            w += np.pi
        elif Omega >= np.pi:
            Omega -= np.pi
            w -= np.pi

        # Wrap w if out of range:
        if w < 0:
            w += 2. * np.pi
        elif w >= 2. * np.pi:
            w -= 2. * np.pi
    
    if debug:
        assert np.min(Omega) >= 0, \
          "Omega has negative values, min is %10.4e" % np.min(Omega)
        assert np.max(Omega) < np.pi, "Omega has values greater than pi"
        assert np.min(w) >= 0, "w has negative values."
        assert np.max(w) < 2. * np.pi, "w has values greater than 2 pi"
        
    q1 = (A+G)/np.cos(w+Omega)
    q2 = (A-G)/np.cos(w-Omega)
    
    i = 2. * np.arctan(np.sqrt(q2/q1))
    a = 0.5 * (q1 + q2)
    
    return w, a, i, Omega

# Some simple conversion functions to just make the code
# cleaner (and avoid conversion errors):

def arcsec2au(angles, d_pc):
    ''' Given input vector of angular distances in arcsec, 
    return their value in AU, given the distance in pc.'''
    
    return angles*d_pc

def au2arcsec(au_vals, d_pc):
    ''' Given input vector of distances in AU, 
    return their value on the sky in arcsec, given the distance in pc.'''
    
    return au_vals/d_pc

def au2mas(au_vals, d_pc):
    ''' Given input vector of distances in AU, 
    return their value on the sky in milliarcseconds, given the distance in pc.'''
    
    return 1000*au_vals/d_pc
    
def mas2au(angles, d_pc):
    ''' Given input vector of angular distances in milliarcseconds, 
    return their value in AU, given the distance in pc.'''
    
    return angles*d_pc/1000.

def rho_PA_to_RA_Dec(rho, PA, rho_errs, PA_errs):
    '''Given input vectors of separation rho and 
    position angle PA, return vectors of separations 
    RA and Dec that give positions on the sky in 
    RA and Dec space, in the same units as rho, typically
    arcsec or mas.  PA is assumed to be in degrees. 
    '''
    
    # Shift by 90 to get angle up from RA axis
    # rather than down from Dec axis, and 
    # convert to radians while we're at it: 
    t = np.deg2rad(90. - PA)
    RA = rho * np.cos(t)
    Dec = rho * np.sin(t)
    
    t_errs = np.deg2rad(PA_errs)
    RA_errs  = np.sqrt((np.cos(t)*rho_errs)**2 + (np.sin(t) * rho * t_errs)**2)
    Dec_errs = np.sqrt((np.sin(t)*rho_errs)**2 + (np.cos(t) * rho * t_errs)**2)
    
    
    return RA, Dec, RA_errs, Dec_errs


def keplerian_xy_Campbell(times, w, a, i, T, e, P, Omega):
    '''For the input array of times (assumed in same units as period P, typically
    years), and orbital elements (all scalars):
    w: argument of periastron (radians)
    a: semimajor axis (AU)
    i: inclination (radians)
    T: time of periastron passage (same units as P, optional);
    e: eccentricity
    P: orbital period
    Omega: PA of the ascending node (radians)
    
    return two separate arrays RA and Dec, which are
    the apparent N and E positions on the sky (e.g. RA and Dec)
    at the specified times.  Dimensions of x and y are the same
    as that of array times.  RA values increase toward the east, 
    so if plotting these in an x-y plot, use plt.gca().invert_xaxis()
    to have RA increase toward the left. '''

    A, B, F, G = Thiele_Innes_from_Campbell(w, a, i, Omega)
    
    return keplerian_xy_Thiele_Innes(times, A, B, F, G, T, e, P)


def keplerian_xy_Thiele_Innes(times, A, B, F, G, T, e, P):
    '''For the input array of times (assumed in same units as period P, typically
    years), the four Thiele-Innes elements A, B, F, and G; and orbital elements (all scalars):
    T: time of periastron passage (same units as P);
    e: eccentricity
    P: orbital period
    
    return two separate arrays RA and Dec, which are
    the apparent N and E positions on the sky (e.g. RA and Dec)
    at the specified times.  Dimensions of x and y are the same
    as that of array times.  RA values increase toward the east, 
    so if plotting these in an x-y plot, use plt.gca().invert_xaxis()
    to have RA increase toward the left.
    '''

    # Calculate the "mean anomaly" M:
    M = (2*np.pi/P) * (times - T)
    # Then the eccentric anomaly:
    E = getE_parallel(M,e)
    X = np.cos(E) - e
    Y = np.sin(E) * np.sqrt(1 - e**2)
    x = A*X + F*Y
    y = B*X + G*Y
    
    # This x,y coordinate system isn't RA/Dec.  Rather, positive
    # x axis is north, so swap to return in RA, Dec order: 
    return y, x

# Now create a function to find the least-squares solution for orbital elements
# given set values of P, e, and T.  Follows the prescription of Lucy et al. 2014,
# A&A 563 A126

def Thiele_Innes_optimal(times_in, P_in, e_in, T_in, x_in, y_in, \
                         x_errs=None, y_errs=None, debug=False):
    '''Given an input array of times, and values of period P, 
    eccentricity e, and time of periastron T, and a set of 
    x, y positions on the sky at those times, find the least-squares
    solution that optimizes the fit of the orbit to the data.
    
    Here x and y are Dec and RA, respectively, following the typical 
    convention.  Optionally, errors on x and y can be specified, in the 
    same units as x and y, and the positions will be weighted accordingly in the 
    least-squares calculation, by 1/err**2 .

    Each of the arrays x_in, y_in, and times_in is assumed to be
    1-dimensional.  P_in, e_in, and T_in can be 1D or multi-D. 

    Returns arrays A, B, F, G, and chi-squared, as well as a list
    containing the arrays of variances and covariances of A, B, F, and
    G.  The first four (A, B, F, G) are
    the Thiele-Innes best-fit elements at each of the input sets of P,
    e, and T; and the latter is an array of the chi-squared values for
    each of those fits.  The best fit overall can be found from
    finding the array element with the minimum chi-squared, and
    extracting the corresponding elements of A, B, F, and G (along
    with P, e, and T).

    This follows the method of L. Lucy, "Mass estimates for visual
    binaries with incomplete orbits", A&A 563, A126 (2014).  Equation
    numbers below refer to equations from that paper.
    http://adsabs.harvard.edu/abs/2014A%26A...563A.126L '''
    
    if x_errs is not None:
        x_weights = 1. / x_errs**2
    else:
        x_weights = 1.
        
    if y_errs is not None:
        y_weights = 1. / y_errs**2
    else:
        y_weights = 1.
        
        
    # Now we will always have multiple times we are fitting, but 
    # the elements P, e, and T may be scalars, or they may 
    # be arrays, in which case we will find the optimal solution
    # at each array entry.  If P, e, and T are arrays, we need 
    # to expand both those arrays and the 'times' array to be
    # multi-dimensional. The resulting arrays have the various 
    # time steps along dimension 0, and variation in input 
    # orbital parameters along the remaining dimensions.  For example,
    # if the input e array is 1-dimensional, it becomes 2D; if it was
    # 2D, it becomes 3D, etc.   These dimension changes are internal
    # to this routine; eventually we sum the arrays along that time
    # dimension and the output arrays have the same dimensions as the
    # input. 
    
    n_times = times_in.size

    if np.isscalar(P_in):
        is_array = False
        n_orb_elements = 1
        
        # Just dealing with scalars - no need
        # to expand anything:
        P = P_in
        e = e_in
        T = T_in
        times = times_in
        x = x_in
        y = y_in
    else:
        is_array = True
        n_orb_elements = P_in.size
        P_shape = P_in.shape
        P_ndim = P_in.ndim
        if debug:
            print("Working with %d element arrays." % n_orb_elements)
        
        assert (P_in.shape == e_in.shape) and (P_in.shape == T_in.shape), \
            'Orbital element arrays P, e, and T must all have same dimensions.'
        
        # Reshape the orbital element arrays to expand along
        # a time dimension:
        new_shape =  (n_times,) + P_shape
        P = np.broadcast_to(P_in, new_shape)
        e = np.broadcast_to(e_in, new_shape)
        T = np.broadcast_to(T_in, new_shape)
        # Reshape the time array to expand along
        # the orbital element dimension; here we have to 
        # reshape it first (essentially a transpose) since 
        # we're expanding in the other direction.  To broadcast it
        # to expand to the shape of the arrays above, it needs to
        # extra dimensions of size 1 that trail.  The 'ndmin'
        # argument to np.array adds size-1 dimensions, but they are
        # leading by default, so we have to transpose.
        
        times = np.broadcast_to(np.array(times_in, ndmin=P_ndim+1).transpose(), new_shape)
        x = np.broadcast_to(np.array(x_in, ndmin=P_ndim+1).transpose(), new_shape)
        y = np.broadcast_to(np.array(y_in, ndmin=P_ndim+1).transpose(), new_shape)
        
        # Weights may need to be reshaped if they aren't just a single value:
        if not np.isscalar(x_weights):
            assert x_weights.size == n_times, 'Must have same number of x weights as data points if not scalar.'
            x_weights = np.broadcast_to(np.array(x_weights, ndmin=P_ndim+1).transpose(), new_shape)
        if not np.isscalar(y_weights):
            assert y_weights.size == n_times, 'Must have same number of y weights as data points if not scalar.'
            y_weights = np.broadcast_to(np.array(y_weights, ndmin=P_ndim+1).transpose(), new_shape)
        
    
    if debug:
        print("Shape of P array is", P.shape)
        print("Min and max are %0.2f and %0.2f." % (np.min(P), np.max(P)))
        plt.hist(np.ravel(P))
        plt.xlabel("Period (yrs)")
        plt.show()
        print("Size of times array is %d." % times.size)
        print("Min and max are %0.2f and %0.2f." % (np.min(times), np.max(times)))

    # Calculate the "mean anomaly" M:
    M = (2*np.pi/P) * (times - T)
    #M = M - np.floor(M/(2.*np.pi))*2*np.pi
    # Then the eccentric anomaly:
    E = getE_parallel(M,e)
    # Eqs. A.3
    X = np.cos(E) - e
    Y = np.sin(E) * np.sqrt(1 - e**2)

    if debug:
        print("Size of E array is %d." % E.size)
        print("Min and max are %0.2f and %0.2f." % (np.min(E), np.max(E)))

    # Now do the calculations that lead to the optimal A, B, F, and G
    # values, section A.2 of Lucy 2014A. 

    # In calculating the chi-squared and the uncertainties on
    # parameters, there is a leading "sigma" variable on the
    # chi-squared sum in the equations in Lucy 2014, Eq. A.5.  But for
    # our usual case of using the meaqsurement uncertainties from the
    # data for the weights w_n = (1/sigma_n)**2, this leading sigma
    # term is 1.  Still, carry it through here and in Eqs. A.9 and
    # A.10 for the variance and covariance of A, B, F, and G for
    # completeness in case it's relevant for another use case.

    sigma = 1
    
    # For all of our sums over data points, we are 
    # always summing along axis 0 - this is the only 
    # axis present if orbital elements are scalars, 
    # and it's the data-point (time or spatial) axis
    # for 2D arrays (Eqs. A.6):
    a = np.sum(x_weights*X**2, axis=0)
    b = np.sum(x_weights*Y**2, axis=0)  
    c = np.sum(x_weights*X*Y, axis=0)
    Delta = a*b - c**2
    r_11 = np.sum(x_weights*x*X, axis=0)
    r_12 = np.sum(x_weights*x*Y, axis=0)

    # Eq. A.7:
    A = ( b*r_11 - c*r_12)/Delta
    F = (-c*r_11 + a*r_12)/Delta

    # Calculate the variance and covariance of A and F:
    sigma_A = np.sqrt((b/Delta)* sigma**2)
    sigma_F = np.sqrt((a/Delta)* sigma**2)
    cov_AF = (-c/Delta) * sigma**2
    
    if debug:
        print("Size of A array is %d." % A.size)
        print("Min and max are %0.2f and %0.2f." % (np.min(A), np.max(A)))

    # Above quantities A and F depend on the x values 
    # and their weights, while the quantities B and G 
    # depend on the y values and their weights, so 
    # recalculate a, b, c and Delta using the y_weights:

    # Eq. A.6 again, for y:
    a = np.sum(y_weights*X**2, axis=0)
    b = np.sum(y_weights*Y**2, axis=0)    
    c = np.sum(y_weights*X*Y, axis=0)    
    Delta = a*b - c**2
    r_21 = np.sum(y_weights*y*X, axis=0)
    r_22 = np.sum(y_weights*y*Y, axis=0)

    # Eq. A.7 again: 
    B = ( b*r_21 - c*r_22)/Delta
    G = (-c*r_21 + a*r_22)/Delta
    
    # Calculate the variance and covariance of B and G.
    sigma_B = np.sqrt((b/Delta)* sigma**2)
    sigma_G = np.sqrt((a/Delta)* sigma**2)
    cov_BG = (-c/Delta) * sigma**2
    

    # Calculate and return the chi-squared value for 
    # this orbit (Eq. A.2):
    x_fit = A*X + F*Y
    y_fit = B*X + G*Y
    
    if debug:
        print("Size of A array is %d." % A.size)
        print("Size of X array is %d." % X.size)
        print("Size of x_fit array is %d." % x_fit.size)


    
    chi_squared = (1/sigma**2) * (np.sum(x_weights*(x_fit - x)**2, axis=0) + \
                                  np.sum(y_weights*(y_fit - y)**2, axis=0))

    # Put all the uncertainty terms into a list so it's a little
    # easier to pass back: 
    sigma_list = [sigma_A, sigma_B, sigma_F, sigma_G, cov_AF, cov_BG]
                                  
    return A, B, F, G, sigma_list, chi_squared


def grid_P_e_T(n, logP_min, logP_max, e_min=0, e_max=0.99, tau_min=0, tau_max=0.99, T_start=0):
    '''Return arrays of P, e, and T values, with dimension n**3, that grid 
    the possible combinations of parameters for sampling the orbit space. The bounds of 
    logP must be specified (units of log(years)).  Eccentricity is assumed to span the 
    space 0 to 0.99 unless otherwise specified, and T spans a whole orbital period (in n steps)
    at each different period value.  The variable tau
    is the orbital phase of the periastron passage, so has dimensionless units 0 to 1. If you 
    want T to have values that are more like the data values, 
    then pass in a T start value, e.g. the minimum date in the data you are fitting.
    '''
    P_array = np.zeros(n**3)
    e_array = np.zeros(n**3)
    T_array = np.zeros(n**3)

    j = 0
    for logP in np.linspace(logP_min,logP_max,n):
        # For each period, get a range of 
        # times of periastron that spans
        # the course of the period.
        # Do this by gridding tau evenly, 
        # where tau is the fraction of the period
        # at which periastron occurs, starting
        # from the first data point. So we then 
        # calculate the T's from there separately 
        # for each P.
        P = 10**logP
        for tau in np.linspace(tau_min,tau_max,n):
            T = T_start + tau*P
            for e in np.linspace(e_min,e_max,n):
                P_array[j] = P
                e_array[j] = e
                T_array[j] = T
                j += 1

    return P_array, e_array, T_array


def best_fit_orbit(times_obs, x_obs, y_obs, x_errs, y_errs, logP_min=1, logP_max=4, \
                   e_min=0, e_max=0.99, n=100, refine_grid=False, verbose=True):
    '''Given the input x, y, and time positions, and uncertainties on x and y, find 
    the best-fit orbit in the range of logP (years) and eccentricity values given. 
    If refine_grid=True, do a second fitting pass where the search boundaries are 
    refined based on the first fit to only include regions of parameter space that 
    are within a given value of the chi-squared of the initial best fit from the first 
    pass. 
    
    Returns w, a, i, T, e, P, Omega, and chi-squared values of the best fit.  Angles are 
    given in degrees, a and P in units of input.'''
    
    import time as Time
    
    start_time = Time.time()
    P_array, e_array, T_array = grid_P_e_T(n, logP_min, logP_max, e_min, \
                                           e_max, T_start=np.min(times_obs))

    A_array, B_array, F_array, G_array, sigma_list, chi_squared = Thiele_Innes_optimal(times_obs, P_array, e_array, \
                                                                  T_array, x_obs, y_obs, \
                                                                  x_errs, y_errs, debug=False)

    w_array, a_array, i_array, Omega_array = Campbell_from_Thiele_Innes(A_array, B_array, F_array, G_array)
    end_time = Time.time()
    reduced_chi_squared = chi_squared/(times_obs.size - 3)
    # As written now the arrays above are all 1-D, but it's possible
    # for those routines to deal with multi-d input and output arrays,
    # so to be general, make sure we convert the "best-fit" index into
    # a tuple that matches the dimensions of the arrays.  Still works
    # fine in the 1-D case, too: 
    z_flat = np.argmin(chi_squared)
    z = np.unravel_index(z_flat, chi_squared.shape)
    
    min_chi = reduced_chi_squared[z]
    e = e_array[z]
    P = P_array[z]
    T = T_array[z]
    w = np.rad2deg(w_array[z])
    a = a_array[z]
    i = np.rad2deg(i_array[z])
    Omega = np.rad2deg(Omega_array[z])
    
    if verbose:
        print("Grid search of %d points took %0.1f seconds." % (n**3, end_time - start_time))
        print("Minimum reduced chi-squared is %0.2f." % reduced_chi_squared[z])
        print("Best-fit:\n e: %0.2f \n P: %0.2f \n T: %0.1f \n w: %0.2f \n a: %0.1f \n i: %0.1f \n Omega: %0.1f" % \
          (e_array[z], P_array[z], T_array[z], np.rad2deg(w_array[z]), a_array[z], 
              np.rad2deg(i_array[z]), np.rad2deg(Omega_array[z])))
        
    if refine_grid:        
        # Now refine the grid and do it again.  We throw out anything with a large 
        # delta chi-squared from the minimum and use the remaining values to set 
        # new grid limits to refine the grid a bit: 

        if verbose:
            print("\nRefining grid and re-doing fit...\n")
        delta_chi = 5
        ok = (reduced_chi_squared - min_chi) < delta_chi
        new_e_min = np.min(e_array[ok])
        new_e_max = np.max(e_array[ok])
        new_P_min = np.min(P_array[ok])
        new_P_max = np.max(P_array[ok])
        new_tau_min = np.min((T_array[ok] - data_start)/P_array[ok])
        new_tau_max = np.max((T_array[ok] - data_start)/P_array[ok])
        # Just call the same routine again to make new grid and do new fit, but 
        # *don't* set refine_grid to True this time, so we don't loop endlessly:
        w, a, i, T, e, P, Omega, min_chi = best_fit_orbit(times_obs, x_obs, y_obs, x_errs, y_errs, \
                                                          np.log10(new_P_min), np.log10(new_P_max), 
                                                          new_e_min, new_e_max, refine_grid=False, n=n, \
                                                          verbose=verbose)
        
        
    return w, a, i, T, e, P, Omega, min_chi


def correct_orbit_likelihood(P_in, e_in, T_in, \
                             A_in, B_in, F_in, G_in,\
                             sigma_list, chi_squared, N):

    '''The function Thiele_Innes_optimal gives the best-fit (minimum
    chi-squared) orbit at each point of a grid of P, e, and T,
    following the analysis of Lucy 2014a.  However, according to the
    analysis of Lucy 2014b, this approach does not fully sample the
    posterior likelihood distribution for the resulting orbital
    parameters, and requires some additional sampling in order to give
    unbiased estimates of the posterior likelihoods.  This function
    takes the input grid of P, e, and T used for Thiele_Innes_optimal,
    and the outputs from that function (A, B, F, G arrays and their
    variances and covariances, as well as the chi-squared of those
    orbits), and applies this sampling correction.

    Inputs:

          Basically, the output of Thiele_Innes_optimal can be passed
          directly as input to this function, along with the same
          input arrays of P, e, and T.  See that function and Lucy
          2014a for more detail, but briefly these are the inputs.
          All arrays should have the same dimensions, typically n**3
          from sampling the P, e, and T grid with n elements each. 
    
          - arrays of P, e, T, A, B, F and G, all with the same
            dimensions.
          - the "sigma_list" returned by Thiele_Innes_optimal,
              which is a list of arrays of variances and covariances
              as described in that function.
          - array of chi-squared values of the orbits from the above
            arrays against whatever dataset was passed to
            Thiele_Innes_optimal. 
    
          - Integer N.  This is Lucy 2014b's script N (see Section 3
            and following), the number of random samples of the
            parameter space at each input grid point.

    Outputs:

         All output arrays are augmented from the inputs by an extra
         leading dimension of size N.  So if n**3 is the input 1-D
         array size (i.e. numpy shape is (n**3,) ), then all output
         arrays will be 2D with shape (N, n**3).  
    
         The output orbital parameter arrays are returned
         in the same order as best_fit_orbit, that is  w, a,
         i, T, e, P, Omega.   An array of likelihoods of the same
         dimensions is also returned, which can be used as weights in
         making histograms of the parameters, or calculating
         confidence intervals following Lucy 2014b's prescription.
         Although the P, e, and T arrays are simply larger versions of
         the input arrays (i.e. all values along the leading size-N
         dimension are copies of the input value), the larger arrays
         are returned for convenience in calculating other
         quantities, e.g. mass from a and P.

         Note also that the likelihood array has the same property -
         all values along the leading dimension (for a given value of
         the second dimension) are the same.

         Finally, note that if plotting histograms of these, you want
         to be sure to pass them wrapped in np.ravel() to convert to
         1D, since otherwise it will try to plot N different
         histograms (which will be slow)!'''


    #  For clarity in the code, unpack the sigma_list into separate
    #  arrays:

    sigma_A, sigma_B, sigma_F, sigma_G, cov_AF, cov_BG = sigma_list

    # From definition following Eq. A.1 of Lucy 2014B for AF, and
    # implied analog for BG: 
    rho_AF = cov_AF / (sigma_A*sigma_F)
    rho_BG = cov_BG / (sigma_B*sigma_G)

    # Similar to the comment in Thiele_Innes_optimal above, there is a
    # bare sigma variable used by Lucy that will generally be 1 in the
    # way we typically define weights.  Include here for
    # completeness.

    sigma = 1

    # Eq. A.10 of Lucy 2014B.  (All eq. numbers below refer to this
    # paper unless otherwise specified.) 
    eta = (1/sigma**4) * ( sigma_A * sigma_B * sigma_F * sigma_G) * \
      np.sqrt((1 - rho_AF**2)*(1 - rho_BG**2))

    # Calculate a 1D version of the likelihood, which we'll then expand
    # to 2D.
    # Eq. 14:
    likelihood_1D = (eta/N) * np.exp(-0.5*chi_squared)

    # Now we have a whole bunch of arrays that we need to broadcast to
    # add a leading dimension of size N:

    new_shape =  (N,) + A_in.shape
    P = np.broadcast_to(P_in, new_shape)
    e = np.broadcast_to(e_in, new_shape)
    T = np.broadcast_to(T_in, new_shape)
    A = np.broadcast_to(A_in, new_shape)
    B = np.broadcast_to(B_in, new_shape)
    F = np.broadcast_to(F_in, new_shape)
    G = np.broadcast_to(G_in, new_shape)
    
    sigma_A = np.broadcast_to(sigma_A, new_shape)
    sigma_B = np.broadcast_to(sigma_B, new_shape)
    sigma_F = np.broadcast_to(sigma_F, new_shape)
    sigma_G = np.broadcast_to(sigma_G, new_shape)

    rho_AF = np.broadcast_to(rho_AF, new_shape)
    rho_BG = np.broadcast_to(rho_BG, new_shape)
    eta = np.broadcast_to(eta, new_shape)

    likelihood = np.broadcast_to(likelihood_1D, new_shape)

    # Now we need some Gaussian random deviates, to sample other steps
    # in the parameter space.  These are the script versions of A, B,
    # F, and G from Eqs. A.11 and A.15.

    A_script = np.random.normal(size=new_shape)
    B_script = np.random.normal(size=new_shape)
    F_script = np.random.normal(size=new_shape)
    G_script = np.random.normal(size=new_shape)

    # Now calculate a, b, f, g (lowercase versions) which are the
    # steps in Thiele-Innes space to be applied to each of the input
    # points.

    # Eqs. A.11:
    a = A_script * sigma_A
    f = sigma_F * (rho_AF * A_script + F_script * np.sqrt(1 -rho_AF**2))

    # Eqs. A.15.  Note that there is a correction here (factor of
    # G_script) to make it parallel with above equation - personal
    # communication from L. Lucy: 
    b = B_script * sigma_B
    g = sigma_G * (rho_BG * B_script + G_script * np.sqrt(1 -rho_BG**2))

    # Now apply these offsets to the arrays to get new samples:
    A_out = A + a
    B_out = B + b
    F_out = F + f
    G_out = G + g

    # And that's it!  Use these to get new orbital elements:
    w, a, i, Omega = Campbell_from_Thiele_Innes(A_out, B_out, F_out, G_out)

    # Only return these for testing:
    script_ABFG = [A_script, B_script, F_script, G_script]
    
    # And then return everything:

    return w, a, i, T, e, P, Omega, likelihood, script_ABFG


def credible_interval(parameter, likelihood, n_sigma=1, log = False):
    '''For the input orbital parameter array and likelihood array,
    find the likelihood-weighted mean and +/- n_sigma credible
    interval, assuming the "sigma" here denotes probability intervals
    from a Gaussian distribution.  (It's not necessary that the
    underlying likelihood distribution is actually Gaussian - we
    can still calculate, e.g., the "one sigma" (68%) credible
    interval.)

    Note: if you want to do the weighted mean and intervals in
    log space for the parameter, pass in the log of the parameter
    array, i.e. take the log outside this function. 

    Inputs:  equal-length arrays of parameter values (which could
    be some function of derived orbital parameters, e.g. it
    could be the mass calculated from the semimajor axis
    and period) and corresponding likelihoods.  Both arrays
    should have the same dimensions, but may be 1D or multi-
    dimensional.  Also (optional) is the number of "sigmas" in
    the Gaussian distribution to use to denoted the desired
    credible interval.

    Outputs: likelihood-weighted mean value, lower bound of
    interval, upper bound of interval.
    '''

    from scipy.stats import norm

    # Get the probability bounds from the cumulative
    # probability distribution function: 
    upper_prob = norm.cdf(abs(n_sigma))
    lower_prob = 1. - upper_prob

    # Make the input arrays one-dimensional so they
    # are easier to work with:
    flat_param = np.ravel(parameter)
    # Normalize the likelihood at the same time as we flatten: 
    flat_likelihood = np.ravel(likelihood)/np.sum(likelihood)

    # Normalize the likelihood
    # Sort the parameter values and get the sort
    # indices so we can sort the likelihood accordingly:
    # Experiments show that 'mergesort' is a bit faster than the
    # default 'quicksort' used by numpy:
    sort_inds = np.argsort(flat_param, kind="mergesort")
    sorted_param = flat_param[sort_inds]
    sorted_likelihood = flat_likelihood[sort_inds]

    # Now get the cumulative sum along the likelihood array
    # so we can find the desired probability points:
    cum_likelihood = np.cumsum(sorted_likelihood)

    low_ind = np.searchsorted(cum_likelihood, lower_prob)
    high_ind = np.searchsorted(cum_likelihood, upper_prob)

    low_val = sorted_param[low_ind]
    high_val = sorted_param[high_ind]

    # And get the likelihood-weighted mean value:
    mean_val = np.sum(flat_param * flat_likelihood)
    
    if log == True:
        # reverse log calculation
        low_val = 10**low_val
        high_val = 10**high_val
        mean_val = 10**mean_val
        
    return mean_val, low_val, high_val

def angle_shift(parameter):
    ''' The angle shift moves values down by a factor of 180 degrees,
    that are above 90 degrees, in order for the means to be centered around 
    the same area of degrees.
    '''
    for j in range(len(parameter)):
        for (k, item) in enumerate(parameter[j]):
            if item > 90:
                parameter[j][k] = item - 180
            elif item < 0:
                parameter[j][k] = item + 360
            else:
                continue
                    
    return parameter

def solar2jup(value):
    M_sun = 1.989e+30
    M_jup = 1.898e+27
    value = value*M_sun/M_jup
    return value


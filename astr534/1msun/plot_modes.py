# load necessary modules
import os
from pudb import set_trace as pause
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from nonad.fileio import read_gyre
from scipy import integrate
import pandas as pd
import string
# load some tools from SYDOSU to do asteroseismic things
from SYDOSU.scaling_relation import logg_MR, loggteff2numax, numax, dnu_MR, freq_dyn, numax_sun, dnu_sun, L_sun, R_sun, M_sun
from SYDOSU.dnu_util import dnu_e
from SYDOSU.dnu_correlation import get_model_full
from SYDOSU.smooth_c import smooth_c
from SYDOSU.util import freq_cut

def surface_correct(f, I, a1, a3, nuac):
    '''
    takes modelled freqs and returns an example of what the surface-corrected modelled freqs might look like. not meant to be a universal surface correction at all because a1 and a3 should be fitted to observed frequencies.
    Inputs
    f : frequency
    I : mode inertia
    a1 : a dimensionless constant for the f^{-1} term. may be set to zero if you don't want this term.
    a3 : a dimensionless constant for the f^{3} term. may be set to zero if you don't want this term.
    nuac : acoustic cutoff frequency 
    Outputs
    Surface effect--corrected frequencies
    '''
    return f + ( a1*(f/nuac)**(-1) + a3*(f/nuac)**3 )/I
def dnu_theory(ells, freqs, numax=None, use_all=False, method='weighted', surface_corr=False, E=None, debug=False, use_all_weighted=True, analyze=False, extra=None, get_n_avg=False, get_n_mode=False):
    '''
    uses the radial modes to compute an average dnu...
    Inputs
    [ get_n_avg : bool ]
     Average radial order used in dnu calc.
    [ get_n_mode: bool ]
     Approx. number of modes used in dnu calc (weighted number).
    [ analyze : bool ]
     whether or not to weight the extra thing and return (dnu, extra).
    ells : pd DataFrame or Series
     the mode degrees.
    freqs : pd DataFrame or Series
     the mode frequencies.
    [ numax : scalar ]
     If provided, will compute dnu around numax +/- 3 dnu ish. Otherwise, uses all of the ell = 0 modes to compute dnu, no matter what use_all is.
    [ use_all_weighted : bool ]
     only applies if method == 'weighted', in which case all the modes will be sued, not just those around +/-3dnu of numax. Default True.
    [ use_all : bool ]
     if True, uses all of the modes to compute dnu. Otherwise use all to compute a dnu and then compute dnu using numax _/- 3dnu unless use_all_weighted is set, in which case weighted scheme is used but using all the freqs.
    [ method : str ['weighted', 'average'] ]
     if 'weighted', will weight the dnu calculatin by a Gaussian envelope centered around numax. so numax must be passed. Default 'weighted'. otherwise, use all (up to 26 at the moment --- hard-coded at the moment) if method == 'average'
    [ surface_corr : bool ]
     if True, apply surface corrections from Gall and Gizon 2014 based on the parameters for the ~1M sugiant (star F in Ball and Gizon 2017). Need to provide mode intertias for each frequency in the <E> kwarg.
    [ E : pd DataFrame or pd Series ]
     Mode inertias for the frequencies
    Outputs
    dnu : scalar
     large frequency separation
    '''
    # compute an initial guess for the large frequency separation using l == 0 modes
    dnu = np.mean(np.diff(freqs.loc[ells == 0])).real

    # weighted is an internal boolean, so need to define it based on what the method is.
    if method == 'weighted':
        weighted = True
    else:
        weighted = False

    # if using all the frequencies, the initial guess for dnu is the average difference of the l == 0 modes.
    if use_all:
        return dnu.real
    # if a numax is provided...
    if numax is not None:
        # take the l == 0 frequencies
        
        _freqs = freqs.loc[ells == 0]
        if analyze:
            _extra = extra.loc[ells == 0]

            # assign them radial orders --- this first takes the real parts of the frequencies, multiplies them by zeros, and then assigns radial orders starting with n == 0.
        
        ns = _freqs.apply(lambda x: x.real)*0.0 + np.arange(len(_freqs))
        # i think only allowing certain ffrequencies to be used in tne dnu calculation leads to steps in the resulting dnuad/dnunad: as dnu decreases, at certain models, a mode will be rejected from analysis, and that may change the dnu calculation in a discontinuous way. so allowing all the modes, and since all the GYRE reuslts have the same number of modes, that sholdn't cause dicontinuous jumps. there will still be an effect, i think, because in the weighted method, you basically will 'see' frewer and fewer modes in the dnu calc., but that should be a smooth and not discontinuous jump. so, putting n_dnu = np.inf to avoid discontinuities.
        # JCZ 040221
        # there are still some discontinuities, and i think they corespond to jumps in the  number of frequencies that are found in the solution. so i'm only considering a set number of modes:

        # !!! may need to take away these restrictions on which modes to use. could do this by setting n_max = np.inf
        
        # n_max = 26
        # _freqs = _freqs.iloc[0:n_max]
        # ns = ns.iloc[0:n_max]

        # if correcting the frequencies for surface effects...
        if surface_corr:
            
            # need the mode intertias --- and make them line up with _freqs
            E = E.iloc[0:n_max]
            
            nuac_sun = 5000. # this value is from the B&G17 analysis, where a1 and a3 are taken from. note that numax_sun is taken to be 3090. so there is some inconsistency in how numax and numax_sun are defined here and in B&G, but the point is to just 
            a1 = 1.0
            a3 = -10.0
            nuac = numax/3076.*nuac_sun

            # correct the frequencies
            _freqs = surface_correct(_freqs, E, a1, a3, nuac)

        # how large of a frequency range should we use to compute the dnu. use the initial guess of dnu (dnu.real) to define this range. plausible values for <n_dnu> may be 3 or np.inf (to use all the frequencies).
        if use_all_weighted:
            n_dnu = np.inf
        else:
            n_dnu = 3.0
        
        ns = ns[np.where(np.abs(_freqs.apply(lambda x: x.real)  - numax) < dnu.real*n_dnu)[0]] + 1
        _freqs = _freqs.loc[np.abs(_freqs.apply(lambda x: x.real)  - numax) < dnu.real*n_dnu]


        if len(ns) == 0:
            
            print("WARNING: No modes were found within +/-3 dnu (dnu = {}) of numax (numax = {}). This may be because it is a highly evolved (dnu < 1) star.".format(dnu, numax))
            if analyze or get_n_avg or get_n_mode:
                return (np.nan, np.nan)

                
            return np.nan
        if analyze:
            _extra = _extra.loc[np.abs(_freqs.apply(lambda x: x.real)  - numax) < dnu.real*n_dnu]


        # try to compute dnus in the case where use_all is False.
        try:
            # this is the first guess for dnu. it differs from the other guess because frequencies are potentially bounded by a region defined by <n_dnu>. if method == 'average' and use_all = False, this will be the output dnu.
            dnu = np.mean(np.diff(_freqs.apply(lambda x: x.real)))
            
            if weighted:
                # JCZ 130919
                # these are the meid-points between the modes
                centers = _freqs.iloc[0:-1].apply(lambda x: x.real) + np.diff(_freqs.apply(lambda x: x.real))/2.0
                # width of the Gaussian. from BAM paper Table 1
                # JCZ 130919
                # !!! multiplying by a factor of 3 to basically relax the wieghting. if wnat the full weight, just put to 1.
                # !!! figure out effect of surface effect........
                b = np.exp(1.05*np.log(numax) - 1.91)*1.0
                weights = np.exp( - (centers - numax)**2/2.0/b**2)

                if debug:
                    print('centers:')
                    print (centers)
                    print('weights:')
                    print (weights)
                    print('numax:')                    
                    print (numax)
                    
                dnu = np.sum( np.diff(_freqs.apply(lambda x: x.real))*weights)/np.sum(weights)
                if get_n_avg:
                    n = np.sum( ns*weights)/np.sum(weights)
                    return (dnu.real, n)

                if get_n_mode:
                    ns *= 0
                    ns += 1
                    n = np.sum( ns*weights)
                    return (dnu.real, n)
                if analyze:
                    extra = np.sum(extra[0:-1]*weights)/np.sum(weights)
                    return (dnu.real, extra)
                # OK actually changing this to do least-squares fitting like white+ 2011, as dennis recommended.
                # JCZ 240919
                # they still weight by exponential, so that still works, but they weight the actual frequencies, not the center between radial modes, and they use 0.25*numax as the width
                # JCZ 040221
                # the below still works just not sure which one to use so currently it's commented out in favor of the center between radial modes technique.
                # b = 0.25*numax
                # weights = np.exp( - (_freqs.apply(lambda x: x.real) - numax)**2/2.0/b**2)
                # import scipy.optimize as optimization
                # def func(x, m,b):
                #     return m*x + b
                # x0 = np.array([dnu, dnu*0.2])
                # fit, cov = optimization.curve_fit(func, ns, _freqs.apply(lambda x: x.real), x0, 1./weights)
                # print ('dnu and epsilon*dnu from LS fittign:')
                # print (fit)
                # print ('dnu from using mean:')
                # print (dnu)
                # dnu  = fit[0]

        # if this fails for some reason, return NaN.
        except:
            dnu = np.nan

    return dnu.real
def small_dnu_theory(ells, freqs, numax=None, use_all=False, method='weighted', surface_corr=False, E=None, debug=False):
    '''
    uses the radial modes to compute an average dnu...
    Inputs
    ells : pd DataFrame or Series
     the mode degrees.
    freqs : pd DataFrame or Series
     the mode frequencies.
    [ numax : scalar ]
     If provided, will compute dnu around numax +/- 3 dnu ish. Otherwise, uses all of the ell = 0 modes to compute dnu, no matter what use_all is.
    [ use_all : bool ]
     if True, uses all of the modes to compute dnu. Otherwise use all to compute a dnu and then compute dnu using numax _/- 3dnu
    [ method : str ['weighted', 'average'] ]
     if 'weighted', will weight the dnu calculatin by a Gaussian envelope centered around numax. so numax must be passed. Default 'weighted'. otherwise, use all (up to 26 at the moment --- hard-coded at the moment) if method == 'average'
    [ surface_corr : bool ]
     if True, apply surface corrections from Gall and Gizon 2014 based on the parameters for the ~1M sugiant (star F in Ball and Gizon 2017). Need to provide mode intertias for each frequency in the <E> kwarg.
    [ E : pd DataFrame or pd Series ]
     Mode inertias for the frequencies
    Outputs
    dnu : scalar
     large frequency separation
    '''
    # first need to make sure there are the same number of 0 and 2 modes to compute the small frequency separation...
    # added from sarah medina's work JCZ 150123
    small_dnu = np.nan
    l_0 = len(freqs[ells == 0].values)
    l_2 = len(freqs[ells == 2].values)
    if l_0 != l_2:
        
        if l_0 > l_2:
            ind = np.max( ells.loc[ells == 0].index.values)
            ells.loc[ind] = -1
        else:
            ind = np.max( ells.loc[ells == 2].index.values)
            ells.loc[ind] = -1
    
    # compute an initial guess for the large frequency separation using l == 0 modes
    if not (l_0 > 0 and l_2 > 0 and l_0 == l_2):
        return small_dnu
    small_dnu = np.mean(freqs[ells == 0].values - freqs[ells == 2].values)
    
    # weighted is an internal boolean, so need to define it based on what the method is.
    if method == 'weighted':
        weighted = True
    else:
        weighted = False

    # if using all the frequencies, the initial guess for dnu is the average difference of the l == 0 modes.
    if use_all:
        return dnu.real
    # if a numax is provided...
    if numax is not None:
        # take the l == 0 frequencies
        
        # _freqs = freqs.loc[ells == 0]
        # _ells = ells.loc[ells == 0]
        # assign them radial orders --- this first takes the real parts of the frequencies, multiplies them by zeros, and then assigns radial orders starting with n == 0.
        
        ns = freqs.apply(lambda x: x.real)*0.0 + np.arange(len(freqs))
        # i think only allowing certain ffrequencies to be used in tne dnu calculation leads to steps in the resulting dnuad/dnunad: as dnu decreases, at certain models, a mode will be rejected from analysis, and that may change the dnu calculation in a discontinuous way. so allowing all the modes, and since all the GYRE reuslts have the same number of modes, that sholdn't cause dicontinuous jumps. there will still be an effect, i think, because in the weighted method, you basically will 'see' frewer and fewer modes in the dnu calc., but that should be a smooth and not discontinuous jump. so, putting n_dnu = np.inf to avoid discontinuities.
        # JCZ 040221
        # there are still some discontinuities, and i think they corespond to jumps in the  number of frequencies that are found in the solution. so i'm only considering a set number of modes:

        # !!! may need to take away these restrictions on which modes to use. could do this by setting n_max = np.inf
        
        n_max = len(freqs) - 1
        _freqs = freqs.iloc[0:n_max]
        _ells = ells.iloc[0:n_max]
        ns = ns.iloc[0:n_max]

        # if correcting the frequencies for surface effects...
        if surface_corr:
            
            # need the mode intertias --- and make them line up with _freqs
            E = E.iloc[0:n_max]
            
            nuac_sun = 5000. # this value is from the B&G17 analysis, where a1 and a3 are taken from. note that numax_sun is taken to be 3090. so there is some inconsistency in how numax and numax_sun are defined here and in B&G, but the point is to just 
            a1 = 1.0
            a3 = -10.0
            nuac = numax/3076.*nuac_sun

            # correct the frequencies
            _freqs = surface_correct(_freqs, E, a1, a3, nuac)

        # how large of a frequency range should we use to compute the dnu. use the initial guess of dnu (dnu.real) to define this range. plausible values for <n_dnu> may be 3 or np.inf (to use all the frequencies).
        if use_all:
            n_dnu = np.inf
        else:
            n_dnu = 3.0
        dnu = dnu_theory(ells, freqs, numax=numax, use_all=use_all, method=method, surface_corr=surface_corr)
        ns = ns[np.where(np.abs(_freqs.apply(lambda x: x.real)  - numax) < dnu.real*n_dnu)[0]]


        #_ells = ells.loc[np.abs(_freqs.apply(lambda x: x.real)  - numax) < dnu.real*n_dnu]
        _freqs = _freqs.loc[np.abs(_freqs.apply(lambda x: x.real)  - numax) < dnu.real*n_dnu]
        if len(ns) == 0:
            raise Exception("No modes were found within +/-3 dnu (dnu = {}) of numax (numax = {}). This may be because it is a highly evolved (dnu < 1) star.".format(dnu, numax))
        # try to compute dnus in the case where use_all is False.
        if True:
            # this is the first guess for dnu. it differs from the other guess because frequencies are potentially bounded by a region defined by <n_dnu>. if method == 'average' and use_all = False, this will be the output dnu.

            
            if weighted:
                # JCZ 130919
                # these are the meid-points between the modes
                centers = (_freqs.loc[_ells == 0].values + _freqs.loc[_ells == 2].values)/2.0
                # width of the Gaussian. from BAM paper Table 1
                # JCZ 130919
                # !!! multiplying by a factor of 3 to basically relax the wieghting. if wnat the full weight, just put to 1.
                # !!! figure out effect of surface effect........
                b = np.exp(1.05*np.log(numax) - 1.91)*1.0
                weights = np.exp( - (centers - numax)**2/2.0/b**2)

                if debug:
                    print('centers:')
                    print (centers)
                    print('weights:')
                    print (weights)
                    print('numax:')                    
                    print (numax)
                if l_0 > 0 and l_2 > 0:                    
                    small_dnu = np.sum( (_freqs[_ells == 0].values - _freqs[_ells == 2].values)*weights)/np.sum(weights)
                # OK actually changing this to do least-squares fitting like white+ 2011, as dennis recommended.
                # JCZ 240919
                # they still weight by exponential, so that still works, but they weight the actual frequencies, not the center between radial modes, and they use 0.25*numax as the width
                # JCZ 040221
                # the below still works just not sure which one to use so currently it's commented out in favor of the center between radial modes technique.
                # b = 0.25*numax
                # weights = np.exp( - (_freqs.apply(lambda x: x.real) - numax)**2/2.0/b**2)
                # import scipy.optimize as optimization
                # def func(x, m,b):
                #     return m*x + b
                # x0 = np.array([dnu, dnu*0.2])
                # fit, cov = optimization.curve_fit(func, ns, _freqs.apply(lambda x: x.real), x0, 1./weights)
                # print ('dnu and epsilon*dnu from LS fittign:')
                # print (fit)
                # print ('dnu from using mean:')
                # print (dnu)
                # dnu  = fit[0]

        # if this fails for some reason, return NaN.
#        except:
 #           small_dnu = np.nan

    return small_dnu.real

def main(profile = 'profile3.data', model=0, debug=False, ad='ad', plot=False, surface_corr=False, method='weighted', dir='./'):
    '''
    Will save a file called '<profile>.<ad>.astero.in', which has dnu and numax information.
    Will also optionally plot the modes, if plot = True.

    Inputs
    [ dir : str ]
     The directory to put the astero.in file in. Default './'.
    [ profile ] 
     the name of the basename for the GYRE files. Should be of the form 'profileN.data'.
    [ ad : str ]
     Either 'ad' or 'nad', depending on whether or not you awant the adiabatic or non-adiabatic results. This is used to find the summary file.
    [ plot : bool ]
     Plot up the modes
    [ surface_corr : bool ]
     whether or not to apply *token* surface corrections --- to do the real thing would depend on a star-by-star basis. default False.
    [ method : str ]
     'weighted': weight using the white weights around numax; 'average': no weighting --- use all the modes (up to <n_max> at the moment --- this is hard-coded to be 26 as of JCZ 250621) and just take the average difference. Default 'weighted'.
    Outputs
    pd DataFrame
    with keys like 'dnu_obs', 'dnu_scal', 'numax', etc.
    '''

    # this is the file that has the frequency information.
    eigval_file = profile + '.gyre_'+ad+'.eigval.h5'
    print('reading in')
    print(eigval_file)
    
    # # If there is no GYRE output file, compute one
    # gyre_command = '$GYRE_DIR/bin/gyre' # how to call gyre
    # gyre_infile_tmp = 'gyre_tmp.in' # file to create on the fly with the profile you want
    # gyre_infile_template = 'gyre_pms_template.in' # the template file with all the settings you want plus 
    # it should have a line like:
    # file = '{profile}.GYRE'
    # which will then be replaced with the profile you want (this also runs it):
    # if not os.path.isfile(eigval_file):
    #     # Creat the GYRE file
    #     with open(gyre_infile_tmp,'w') as outfile:
    #         with open(gyre_infile_template, 'r') as infile:
    #             outfile.write("".join(infile.readlines()).format(profile=profile))
    #     subprocess.call(gyre_command + ' ' + gyre_infile_tmp, shell=True)

    # JCZ 231018
    # actually, if there is no file, then just print out nans for the model and scaling dnu and for numax

    if not os.path.isfile(eigval_file):
        dnu = [np.nan]
        small_dnu = [np.nan]
        numax = [np.nan]
        dnu_scal = [np.nan]
        attrs = {'R_star':np.nan}
        arrs = {'freq':[np.nan]}
        print('WARNING: no file found')
    # but if GYRE has been run on that profile...
    else:
        # read in the GYRE info.
        attrs, arrs = read_gyre(eigval_file)
        try:
            # make a pandas DataFrame with relevant stellar property info, this is used to compute a numax from scaling relations.
            attrs = pd.DataFrame({'M_star':arrs['M_star'][0], 'R_star':arrs['R_star'][0], 'L_star':arrs['L_star'][0]}, index=[0])
        except:
            pass
        arrs = pd.DataFrame(arrs)
        arrs['count'] = np.arange(len(arrs['freq']))

        count_0 = 0
        count_1 = 0
        count_2 = 0
        
        # need to read in the teff from the original YREC model... which actually is put in the mode file. so NEED A MODE FILE FOR ALL THIS TO WORK
        
        # teff = 10.**(starl.loc[starl['model_number'] == model_number]['log_Teff'])
        # logg = logg_MR(attrs['M_star'].__array__(), attrs['R_star'].__array__())
        # print 'star has logg = {} and Teff = {}'.format(logg, teff)
        # loggteff2numax(logg, teff).__array__()[0]
        # actually don't need temperature, but can rather just use luminosity with the fact that
        # numax = M R^{-7/4} L^{-1/8}
        # in solar units
        # numax =  36.0
        
        # compute a numax from scaling relations
        numax = (attrs['M_star']/M_sun)*(attrs['R_star']/R_sun)**(-7./4.)*(attrs['L_star']/L_sun)**(-1./8.)*numax_sun
        numax = numax[0]

        # if attrs['R_star'].values/R_sun < 5:
            # pause()
        # either compute dnu from theoretical radial modes
        # or from the mean density scaling relation.
        #     dnu = arrs['Delta_p'].__array__()[0]*freq_dyn(attrs['M_star'].__array__(), attrs['R_star'].__array__())[0]
        #     dnu = dnu_MR(attrs['M_star'].__array__(), attrs['R_star'].__array__())[0]

        # JCZ 231018
        # by default, method is weighted.if you set use_all = True and method = 'weighted', then will use weighting applied to all modes. otherwise, if use_all = False and method = 'weighted', then will use modes within +/-<n_max>*dnu of numax with weighting. if method = 'average' and use_all = True, will use simple average of all the modes. if method = 'average' and use_all = False, will use +/-<n_max>*dnu modes.
        use_all = False
        dnu = dnu_theory(arrs['l'], arrs['freq'], numax=numax, use_all=use_all, method=method,E=arrs['E'], surface_corr=surface_corr)


        
        # this is the dnu expected from scaling relations.
        dnu_scal = dnu_MR(attrs['M_star'].__array__(), attrs['R_star'].__array__(), emp=True)
        count = 1
        ratio = []
        eta = []
        inertia = []
        eta2 = []
        sigir = []

        for count in arrs['count']:
            # JCZ 300817
            # read in the mode's eignfunction and calculate the ratio of the mode's amplitude near the surface to that near the core. based on <visibility_thresh>, this will define which of the model modes we expect to be actually able to see at the surface.
            # for a given frequency, get its actual radial structure by reading in the "mode" file. this will give the horizontal displacement of the mode, xi_h, and the vertical displacement of the mode, xi_r (i.e., in the radial direction), as a function of the stellar depth, x.
            ind_file = profile + '.gyre_'+ad+'.mode-{:05d}.h5'.format(count+1)
            if os.path.exists(ind_file):
                meta_mode, mode = read_gyre(profile + '.gyre_'+ad+'.mode-{:05d}.h5'.format(count+1))
            else:
                ratio.append(np.nan)
                eta.append(np.nan)
                inertia.append(np.nan)
                eta2.append(np.nan)
                sigir.append(np.nan)
                continue
            # thresh defines how much of the surface you want to use in computing the ratio of surface to core mode amplitude
            thresh = 0.98
            lo_thresh = 0.02
            # this is a ratio of the integral of the mode amplitude sqrt(xi_h^2 + xi_r^2)dx approximated by the rectangle rule (ish --- doesnt use the center of each rectangle...) in the surface to that near the core. x is the radial coordinate, where 0 is the center and 1 is the surface. if this ratio is large, that means the mode has most of its amplitude in the surface compared to the core, and should be visible. visibility_thresh, below, defines the visibility threshold. !!! this should be replaced by a better integration.
            # ratio.append(np.sum(np.diff(mode['x'])[np.where(mode['x'] > thresh)[0][0]:]*np.sqrt(mode['xi_h']**2 + mode['xi_r']**2)[np.where(mode['x'] > thresh)[0][0]+1:])                         /np.sum(np.diff(mode['x'])[0:np.where(mode['x'] < lo_thresh)[0][-1]]*np.sqrt(mode['xi_h']**2 + mode['xi_r']**2)[0:np.where(mode['x'] < lo_thresh)[0][-1]]))
            
            ind = np.where(mode['x'] > thresh)[0][0]
            ind_outer = np.argmin(np.abs(mode['x'] - 1.0))
            h = mode['xi_h'].real
            r = mode['xi_r'].real
            num = integrate.simps(np.sqrt(h**2 + r**2)[ind:], mode['x'][ind:])
            ind = np.where(mode['x'] < lo_thresh)[0][-1]
            den = integrate.simps(np.sqrt(h**2 + r**2)[0:ind], mode['x'][0:ind])
            ratio.append(num/den)

            
            if ad == 'nad':
                eta.append(integrate.simps(mode['dW_dx'], mode['x'])/integrate.simps(np.abs(mode['dW_dx']), mode['x']))
                R = attrs['R_star'].values[0]
                # M = integrate.simps(4.0*np.pi*mode['rho']*mode['x']**2*R**2, mode['x']*R)
                M = attrs['M_star'].values[0]
                # print(M)
                i = integrate.simps((r**2 + h**2)*4.0*np.pi*mode['rho']*mode['x']**2*R**2, mode['x']*attrs['R_star'].values[0])/attrs['M_star'].values[0]/r[ind_outer]**2
                W = integrate.simps(mode['dW_dx'], mode['x'])

                G = 6.6741e-8
                freq0 = np.sqrt(G*M/R**3)
                omega = arrs['freq'].iloc[count].real*1e-6/(freq0)*2.*np.pi
                _eta2 = -W/i/2.0/omega/r[ind_outer]**2#/attrs['M_star'].values[0]/R**2
                # !!! let's try just integrating the function in the outside of the star
                # _W = integrate.simps(mode['dW_dx'][0:ind], mode['x'][0:ind])
                # _eta2 = -_W/i/2.0/omega/r[ind_outer]**2#/attrs['M_star'].values[0]/R**2
                sigir.append(arrs['freq'].iloc[count].imag/arrs['freq'].iloc[count].real)
                inertia.append(i)
                eta2.append(_eta2)
            else:
                eta.append(np.nan)
                inertia.append(np.nan)
                eta2.append(np.nan)
                sigir.append(np.nan)
            


        arrs['vis'] = ratio
        arrs['eta'] = eta
        arrs['inertia'] = inertia
        arrs['eta2'] = eta2
        arrs['sigir'] = sigir
        # JCZ 240921
        # getting average ns, average visibility, approx. number of modes used for computation of dnu, and average eta.
        _, eta = dnu_theory(arrs['l'], arrs['freq'], numax=numax, use_all=use_all, method=method,E=arrs['E'], surface_corr=surface_corr, analyze=True, extra=arrs['eta'])
        _, vis = dnu_theory(arrs['l'], arrs['freq'], numax=numax, use_all=use_all, method=method,E=arrs['E'], surface_corr=surface_corr, analyze=True, extra=arrs['vis'])
        _, inertia = dnu_theory(arrs['l'], arrs['freq'], numax=numax, use_all=use_all, method=method,E=arrs['E'], surface_corr=surface_corr, analyze=True, extra=arrs['inertia'])
        _, eta2 = dnu_theory(arrs['l'], arrs['freq'], numax=numax, use_all=use_all, method=method,E=arrs['E'], surface_corr=surface_corr, analyze=True, extra=arrs['eta2'])
        _, sigir = dnu_theory(arrs['l'], arrs['freq'], numax=numax, use_all=use_all, method=method,E=arrs['E'], surface_corr=surface_corr, analyze=True, extra=arrs['sigir'])
        _, n_mode = dnu_theory(arrs['l'], arrs['freq'], numax=numax, use_all=use_all, method=method,E=arrs['E'], surface_corr=surface_corr, get_n_mode=True)
        _, n_avg = dnu_theory(arrs['l'], arrs['freq'], numax=numax, use_all=use_all, method=method,E=arrs['E'], surface_corr=surface_corr, get_n_avg=True)


        # !!! currently not working.
        small_dnu = small_dnu_theory(arrs['l'], arrs['freq'], numax=numax, use_all=use_all, method=method,E=arrs['E'], surface_corr=surface_corr)


        # if plotting...
        if plot:
            plt.clf()
            # JCZ 250621


            # at the moment, don't use the mode intertias E_p and E_g nor the gravity mode orders, n_g. also don'nt use the imaginary components of the frequencies at the momenet (set to <damping>).

            for count,l,freq,damping,n_g,E_p,E_g,ratio in zip(arrs['count'], arrs['l'], arrs['freq'].apply(lambda x: x.real), arrs['freq'].apply(lambda x: x.imag), arrs['n_g'], arrs['E_p'], arrs['E_g'], arrs['vis']):
                ind_file = profile + '.gyre_'+ad+'.mode-{:05d}.h5'.format(count+1)
                if os.path.exists(ind_file):
                    # JCZ 300817
                    # read in the mode's eignfunction and calculate the ratio of the mode's amplitude near the surface to that near the core. based on <visibility_thresh>, this will define which of the model modes we expect to be actually able to see at the surface.
                    # for a given frequency, get its actual radial structure by reading in the "mode" file. this will give the horizontal displacement of the mode, xi_h, and the vertical displacement of the mode, xi_r (i.e., in the radial direction), as a function of the stellar depth, x.
                    meta_mode, mode = read_gyre(ind_file)
                    # thresh defines how much of the surface you want to use in computing the ratio of surface to core mode amplitude
                    thresh = 0.98
                    lo_thresh = 0.02
                    # this is a ratio of the integral of the mode amplitude sqrt(xi_h^2 + xi_r^2)dx approximated by the rectangle rule (ish --- doesnt use the center of each rectangle...) in the surface to that near the core. x is the radial coordinate, where 0 is the center and 1 is the surface. if this ratio is large, that means the mode has most of its amplitude in the surface compared to the core, and should be visible. visibility_thresh, below, defines the visibility threshold. !!! this should be replaced by a better integration.
                    ratio = np.sum(np.diff(mode['x'])[np.where(mode['x'] > thresh)[0][0]:]*np.sqrt(mode['xi_h']**2 + mode['xi_r']**2)[np.where(mode['x'] > thresh)[0][0]+1:])/np.sum(np.diff(mode['x'])[0:np.where(mode['x'] < lo_thresh)[0][-1]]*np.sqrt(mode['xi_h']**2 + mode['xi_r']**2)[0:np.where(mode['x'] < lo_thresh)[0][-1]])
                else:
                    print("WARNING: no individual mode file found: {}. assuming all modes are visible for the *freqs.png plot.".format(ind_file))
                    ratio = np.inf
                # print 'E_p : {}\nE_g : {}'.format(E_p, E_g)


                visibility_thresh = 50.0
                # each mode degree is plotted in a different color.
                if l == 0:
                    color = 'red'
                    _label = 'l = 0'
                    count_0 += 1
                if l == 1:
                    color = 'blue'
                    _label = 'l = 1'
                    count_1 += 1
                if l == 2:
                    color = 'green'
                    _label = 'l = 2'
                    count_2 += 1

                if count_0 == 1 or count_1 == 1 or count_2 == 1:
                    label = _label
                else:
                    label = None


                if ratio > visibility_thresh:
                    plt.axvline(x=freq, label=label, linestyle='dashed', color=color)
            # plot +/- 3 orders around numax

            width = 3.0
            plt.xlim([numax - width*dnu, numax + width*dnu])
            plt.xlabel(r'Frequency [$\mu$Hz]')
            plt.ylabel(r'Amplitude [a.u.]')
            # add a red giant pattern on top, for comparison. this is the observed pattern seen empirically in giant stars. each of the three degrees is modelled as a laurentzian function with different widths and amplitudes, and the mode degrees are separated according to an observed small frequency separation, d02

            # # OVERPLOT OBSERVED SPECTRUM
            # # smooth it with <width> element--wide boxcar
            # f, p = np.genfromtxt('/home/ritchey/zinn.44/pinsonneault/pms/hlsp_k2sff_k2_lightcurve_44-c02_kepler_v1_llc.txt.clean.rel.hipass.fill.psd', unpack=True)
            # _, p_cut = freq_cut(f, p, (70, 150))
            # _, f_cut = freq_cut(f, f, (70, 150))
            #_, p_cut = freq_cut(f, p, (90, 120))
            #_, f_cut = freq_cut(f, f, (90, 120))
            # width_sm = 10000
            # # subtract off a smooth background
            # p_sm = smooth_c.smooth_c(p, width_sm, 'boxcar')
            # p -= p_sm

            # f, p = f_cut, p_cut
            # plt.plot(f, p)

            # EITHER  ONE OR THE OTHER -- CHANGE TO FIRST LINE IF OVERPLOTTING OBSERVED SPECTRUM
            # amp = np.max(p)/4.0
            amp = 1.0



            # epislon needs to be -1.7ish for the low freqs and -0.75 around 110. this will shift the location of all the modes up or down in frequency, by an additive amount equal to <epsilon>.
            # <i> is the inclination we see the star at.
            # if <envelope_width> = np.inf, then the modes will all have fixed amplitudes. otherwise, you can make the modes to have a Gaussian shape by specifying a non-np.inf value (e.g., 3.0*dnu), which is the default if you set <envelope_width> to None.
            # rot is the rotational splitting of the modes, in uHz. set to 0 for all the azimuthal orders, m, to overlap for a given l,n combination.
            # the fourth input (ell) says to which order you want to get the model. if set to ell = 0, will only show the radial modes.
            # <n> says how many radial orders to show.
            # how many points to plot for the RGB model. this just determines how smooth to pattern looks
            N_grid = 1000
            f_grid = np.arange(numax - width*dnu, numax + width*dnu, (width*2*dnu)/N_grid)
            ell = 3
            n = int(width*2*2)
            plt.plot(f_grid, get_model_full(f_grid, dnu, numax, ell, n, rot=0.0, epsilon=-dnu*1.58, i=20.)*amp, color='black') # to make constant amplitudes for all the modes, add keyword argument of envelope_width=np.inf
            plt.legend()
            plt.tight_layout()
            plt.savefig(profile + '_freqs.png', format='png')


    # save to a file the dnu_obs, dnu_scal, and period spacings:
    # SARAH: could add a 'd02' field here to save the d02 info.
    import re
    # make a dataframe with relevant info. dnu_obs is the dnu from the GYRE model. dnu_scal is the theoretical scaling relation dnu. numax is the theoretical scaling relation numax. the index is set to be the model number (e.g., if profile3.data is the profile,  then model number = index = 3).


    df = pd.DataFrame({'model':int(re.search('[0-9]{1,}', profile).group(0)), 'dnu_obs':dnu, 'dnu_scal':dnu_scal, 'numax':numax, 'R':attrs['R_star'].values/R_sun, 'ad':(ad == 'ad'), 'nfreq':len(arrs['freq']),  'weighted':(method == 'weighted'), 'n_mode':n_mode, 'n_avg':n_avg, 'vis':vis, 'eta':eta, 'eta2':eta2, 'inertia':inertia, 'sigir':sigir, 'small_dnu':small_dnu}, index=[int(re.search('[0-9]', profile).group(0))])

    # df = pd.DataFrame({'model':int(re.search('[0-9]', profile).group(0)), 'dnu_obs':dnu, 'small_dnu':small_dnu, 'dnu_scal':dnu_scal, 'numax':numax, 'R':attrs['R_star']/R_sun, 'ad':(ad == 'ad'), 'nfreq':len(arrs['freq'])}, index=[int(re.search('[0-9]', profile).group(0))])

    
    class MyFormatter(string.Formatter):                                        
        '''
        accepts E format a la Fortran, but must specify with a number before the : in the format str:
            print MyFormatter().format("{0:8.0f}", 123.12333) 
        don't know why.
        '''
        def format_field(self, value, format_spec):                                   
            # print format_spec, value                                         
            ss = string.Formatter.format_field(self,value,format_spec)                         
            if format_spec.endswith('E'):                                        
                if ( 'E' in ss):                                            
                    # print ss                                             
                    mantissa, exp = ss.split('E')                                    
                    return mantissa + 'E'+ exp[0] + '0' + exp[1:]                            
                return ss                 
            return ss
                
                  
    fmt  = " {0:40.16E}"  
    # JCZ 221018
    # made default w+ not w so it will append to the file
    with open(dir+'{}.{}.astero.in'.format(profile,ad), 'w+') as f:
        # SARAH: would want to add 'd02' to the header.

        f.write('model ad weighted dnu_scal dnu_obs numax n_mode n_avg vis eta eta2 inertia sigir small_dnu'+"\n") 

        # f.write('model small_dnu dnu_scal dnu_obs numax'+"\n") 

        for i in range(len(df)): 
            
            row = df.iloc[i]
            fmt_row = ' '

            fmt_row += MyFormatter().format('{0:8.0f}', row['model'])
            fmt_row += MyFormatter().format(' {0:1.0f} ', row['ad'])
            fmt_row += MyFormatter().format(' {0:1.0f} ', row['weighted'])
            # SARAH: would want to add 'd02' here to save to a file.

            for key in ['dnu_scal','dnu_obs','numax', 'n_mode', 'n_avg', 'vis', 'eta', 'eta2', 'inertia', 'sigir', 'small_dnu']:

            # for key in ['small_dnu', 'dnu_scal','dnu_obs','numax']:

                fmt_row += MyFormatter().format(fmt, row[key].real) 
            f.write(fmt_row+"\n") 
    # JCZ 090819
    # added this so can call this program directly.
    return df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
            description='append dnu_scal, dnu_obs, and numax to a file.')
    parser.add_argument('profile',nargs='?',help='results will be stored in <profile>.<ad>.astero.in. of the form profileN.data')
    parser.add_argument('--plot',action='store_true',help='plot of the GYRE modes in dashed lines and a typical RGB spectrum in solid curve, and saved to file <profile> + _freqs.png')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--dir', default='./', help='directory to save things to')
    args = parser.parse_args()
    print(args.dir)
    
    main(debug=args.debug, profile=args.profile, plot=args.plot, dir=args.dir)

# -*- coding: utf-8 -*-
r"""
Read and write files from stellar evolution and pulsation codes.

Summary
========

**ADIPLS and ASTEC**

.. autosummary::
    
    read_ngong
    read_fgong
    read_emdl
    read_ssm
    read_gsm
    read_aeig
    read_amdl

**MESA**

.. autosummary::

    read_mesa
    list_mesa_data

**GYRE**    

.. autosummary::

    read_gyre

Useful relations
=================

.. math::
    \delta \equiv -\left(\frac{\partial\ln\rho}{\partial\ln T}\right)_p^{(\dagger,\star)}

    A (A^*) \equiv -\frac{d\ln\rho}{d\ln r}+\frac{1}{\Gamma_1}\frac{d\ln p}{d\ln r}^{(\dagger)}

    A = N^2 \frac{r}{g}

    U \equiv \frac{4\pi\rho r^3}{m}^{(\dagger)}

    V_g \equiv -\frac{1}{\Gamma_1}\frac{d\ln P}{d\ln r} = \frac{G m\rho}{\Gamma_1 p r}^{(\dagger)}

    S_l = \frac{\ell(\ell+1) c^2}{r^2}^{(\dagger)}

    D_5 = - \left(\frac{1}{\Gamma_1 p}\frac{d^2p}{dx^2}\right)_c^{(\dagger)}

    D_6 = - \left(\frac{1}{\rho}\frac{d^2\rho}{dx^2}\right)_c^{(\dagger)}

    \alpha \equiv \left(\frac{\partial\ln\rho}{\partial\ln p}\right)_T = \frac{1}{\Gamma_1} + \delta \nabla_\mathrm{ad}^{(\ddagger)}

    c_V = c_p - \frac{p\delta^2\Gamma_1}{\rho T(1+\delta\Gamma_1\nabla_\mathrm{ad})}^{(\ddagger)}

    \nabla_\mathrm{ad} = \frac{p\delta}{\rho T c_p} = \left(\frac{\partial\ln T}{\partial\ln p}\right)_S^{(\star)}

    \nabla = \frac{d\ln T}{d\ln p}

    \Gamma_1 = \left(\frac{\partial\ln p}{\partial \ln\rho}\right)_S^{(\star)}

    c_1 = \frac{r^3}{R^3}\frac{M}{m} = \frac{x^3}{q}  ^{(\star)}
    
    w = -\frac{m(r)}{m(r)-M} ^{(\star)}


Notes:

 * :math:`H` is enthalpy
 
References:

 * :math:`\dagger`: Notes on using ASTEC and ADIPLS, November 2010, Joergen Christensen-Dalsgaard
 * :math:`\ddagger`: Notes on evolution programme, August 2008, Joergen Christensen-Dalsgaard
 * :math:`\star`: GYRE user manual


Generating this page
====================

::
    
    $:> sphinx-apidoc -f -o doc stellarmodels
    $:> sphinx-quickstart
    $:> make html


Full documentation with examples
=================================
    
"""
import os
import struct
import numpy as np
try:
    import h5py
except ImportError:
    print("h5py package not installed, unable to read/write HDF5 files")
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed")

GG = 6.6742800000e-08

def read_array(content, size, fmt):
    """
    Read a binary array.
    
    @param content: binary data
    @type content: binary data
    @param size: size of the array
    @type size: int
    @param fmt: format of the array
    @type fmt: str
    @return: the array, rest of the binary data
    @rtype: ndarray, binary data
    """
    array_fmt = '<'+size*fmt
    array_size = struct.calcsize(array_fmt)
    this_content = content[:array_size]
    this_array = struct.unpack(array_fmt, this_content)
    content = content[array_size:]
    return np.array(this_array), content

def read_ngong(filename):
    """
    Read in a GONG file as output by ASTEC.
    
    **Example usage:**
    
    >>> cdata,inter,datmod,idatmd,datgng,bccoef,yvar =read_ngong('example.gong')
    >>> print('\\n'.join(cdata))
     date and time     :Dummy_date_and_time
     trial model file  :emdl.0100.Z2.01.s
     output model file :/dev/null
     GH interpolation, version v9. OPAL92, Kurucz91 tables
    >>> print(inter)
    (1, -2, 642, 200, 80, 70, 30, 87)
    
    **Explanation of contents:**
    
    Array ``datmod``
    
    * ``datmod[0]``: :math:`Z_0` initial heavy element abundance
    * ...
    
    @param filename: name of the GONG file
    @type filename: str
    @return: cdata, integer flags, datmod, idatmd, datgng, bccoef, yvar
    @rtype: list of strings, 8-tuple int, 1D float array, 1D integer array, 1D float array, 1D float array, 2D float array
    """
    with open(filename,'rb') as open_file:
        #-- read in all the bytes at once
        content = open_file.read()
    
    #-- weird integer header
    header_fmt = '<i'
    head = struct.calcsize(header_fmt)
    __ = content[:head]
    content = content[head:]

    #-- start of cdata
    cdata = []
    cdata_fmt = 80*'s'
    cdata_size = struct.calcsize(cdata_fmt)

    for i in range(4):
        this_content = content[:cdata_size]
        cdata.append(("".join(struct.unpack(cdata_fmt, this_content))).rstrip())
        content = content[cdata_size:]
                
    #-- intermediate info
    inter_fmt = '<'+8*'i'
    inter_size = struct.calcsize(inter_fmt)
    this_content = content[:inter_size]
    inter = struct.unpack(inter_fmt, this_content)
    content = content[inter_size:]
    nmod, iform, nn, nrdtmd, nidtmd, ndtgng, nvar, nbccf = inter

    #-- the arrays
    datmod, content = read_array(content, nrdtmd, 'd')
    idatmd, content = read_array(content, nidtmd, 'i')
    datgng, content = read_array(content, ndtgng, 'd')
    bccoef, content = read_array(content, nbccf, 'd')
    yvar, content = read_array(content, nn*nvar, 'd')
    yvar = yvar.reshape((nn, nvar))
    
    #-- now put the arrays in something workable?
    #---> seems to much work just return the arrays as they are
    
    return cdata, inter, datmod, idatmd, datgng, bccoef, yvar


def read_emdl(filename):
    """
    Read in an EMDL file from ASTEC.
    
    @param filename: name of emdl-file
    @type filename: str
    @return: some output
    @rtype: some type
    """
    with open(filename,'rb') as open_file:
        #-- read in all the bytes at once
        content = open_file.read()
    
    #-- weird integer header
    header_fmt = '<i'
    head = struct.calcsize(header_fmt)
    __ = content[:head]
    content = content[head:]
    
    #-- intermediate info
    inter_fmt = '<'+6*'i'
    inter_size = struct.calcsize(inter_fmt)
    this_content = content[:inter_size]
    inter = struct.unpack(inter_fmt, this_content)
    content = content[inter_size:]
    iform, nn, nrdtmd, nidtmd, nvarfl, nbccf = inter
    
    #-- the arrays
    datmod, content = read_array(content, nrdtmd, 'd')
    idatmd, content = read_array(content, nidtmd, 'i')
    yvar, content = read_array(content, nn*(nvarfl+1), 'd')
    yvar = yvar.reshape((nn, nvarfl+1))
    x = yvar[:, 0]
    yvar = yvar[:, 1:]
    bccoef, content = read_array(content, nbccf, 'd')
    return inter, datmod, idatmd, x, yvar, bccoef

def read_fgong(filename):
    """
    Read in an FGONG file.
    
    This function can read in FONG versions 250 and 300.
    
    The output gives the center first, surface last.
    
    **Example usage**
    
    >>> starg,starl = read_fgong('example.fgong')
    
    *Global stellar properties*
    
    The global stellar properties are stored in a normal dictionary:
    
    >>> print(starg)
    {'initial_x': 0.0, 'lam': 0.0, 'initial_z': 0.02, 'xi': 0.0, 'ddrho_drr_c': -121.4597632, 'na15': 0.0, 'na14': 18596.197489999999, 'photosphere_r': 371909348600.0, 'ddP_drr_c': -172.6751611, 'phi': 0.0, 'beta': 0.0, 'photosphere_L': 1.178675752e+37, 'star_age': 35245914.009999998, 'alpha': 1.8, 'star_mass': 1.39244e+34}

    You can get a list of all the keys in this dictionary, and access individual
    values via:
    
    >>> print(list(starg.keys()))
    ['initial_x', 'lam', 'initial_z', 'xi', 'ddrho_drr_c', 'na15', 'na14', 'photosphere_r', 'ddP_drr_c', 'phi', 'beta', 'photosphere_L', 'star_age', 'alpha', 'star_mass']
    >>> print(starg['star_mass'])
    1.39244e+34
    
    *Local stellar properties*
    
    The local stellar properties are stored in a record array. The benefit of
    a record array is that you can access the columns by there name. You can list
    all the column names with:
    
    >>> print(starl.dtype.names)
    ('radius', 'ln(m/M)', 'temperature', 'pressure', 'density', 'X', 'luminosity', 'opacity', 'eps_nuc', 'gamma1', 'grada', 'delta', 'cp', 'free_e', 'brunt_A', 'rx', 'Z', 'R-r', 'eps_grav', 'Lg', 'xhe3', 'xc12', 'xc13', 'xn14', 'xo16', 'dG1_drho', 'dG1_dp', 'dG1_dY', 'xh2', 'xhe4', 'xli7', 'xbe7', 'xn15', 'xo17', 'xo18', 'xne20', 'xh1', 'na38', 'na39', 'na40')
    
    An entire column is accessed via e.g. ``starl['radius']``. To get only
    the first 5 entries, you can do
    
    >>> print(starl['radius'][:5])
    [  0.00000000e+00   2.63489963e+08   3.31977388e+08   4.18267440e+08
       5.26988298e+08]
    
    You can sub select multiple columns simultaneously, and have a nice string
    representation via
    
    >>> print(plt.mlab.rec2txt(starl[['radius','temperature']][:5]))
              radius    temperature
               0.000   32356218.190
       263489963.200   32355749.110
       331977388.500   32355443.030
       418267440.400   32354966.580
       526988298.100   32354211.870
       
    Record arrays support array indexing, so if you only want to select that
    part of the star with a temperature between 200000K and 5000000K, you can do
    (we only print the last 9 lines but plot everything):
    
    >>> keep = (200000<starl['temperature']) & (starl['temperature']<5000000)
    >>> print(plt.mlab.rec2txt(starl[keep][-9:]))
                 radius   ln(m/M)   temperature       pressure   density       X                                   luminosity   opacity   eps_nuc   gamma1   grada   delta               cp   free_e   brunt_A       rx       Z               R-r   eps_grav      Lg    xhe3    xc12    xc13    xn14    xo16   dG1_drho   dG1_dp   dG1_dY     xh2    xhe4    xli7    xbe7    xn15    xo17    xo18   xne20     xh1    na38    na39    na40
       355098749500.000    -0.000    217776.342   28893242.570     0.000   0.700   11786757520000000148929153677249216512.000     6.093    -0.057    1.512   0.295   1.979   1116313577.000    0.843     4.780   -0.000   0.020   16810599020.000     -0.057   0.000   0.000   0.003   0.000   0.001   0.010      0.000    0.000    0.000   0.000   0.280   0.000   0.000   0.000   0.000   0.000   0.002   0.000   0.000   0.000   0.000
       355254331000.000    -0.000    215959.487   27989491.870     0.000   0.700   11786757520000000148929153677249216512.000     6.146    -0.055    1.512   0.295   1.977   1114375987.000    0.843     4.440   -0.000   0.020   16655017510.000     -0.055   0.000   0.000   0.003   0.000   0.001   0.010      0.000    0.000    0.000   0.000   0.280   0.000   0.000   0.000   0.000   0.000   0.002   0.000   0.000   0.000   0.000
       355386669700.000    -0.000    214408.595   27237358.180     0.000   0.700   11786757520000000148929153677249216512.000     6.189    -0.053    1.512   0.295   1.975   1112603487.000    0.843     4.144   -0.000   0.020   16522678890.000     -0.053   0.000   0.000   0.003   0.000   0.001   0.010      0.000    0.000    0.000   0.000   0.280   0.000   0.000   0.000   0.000   0.000   0.002   0.000   0.000   0.000   0.000
       355603756700.000    -0.000    211839.378   26036219.670     0.000   0.700   11786757520000000148929153677249216512.000     6.258    -0.049    1.512   0.295   1.971   1109132366.000    0.843     3.641   -0.000   0.020   16305591810.000     -0.049   0.000   0.000   0.003   0.000   0.001   0.010      0.000    0.000    0.000   0.000   0.280   0.000   0.000   0.000   0.000   0.000   0.002   0.000   0.000   0.000   0.000
       355827789500.000    -0.000    209133.425   24838430.420     0.000   0.700   11786757520000000148929153677249216512.000     6.329    -0.045    1.513   0.295   1.966   1104338120.000    0.843     3.088   -0.000   0.020   16081559030.000     -0.045   0.000   0.000   0.003   0.000   0.001   0.010      0.000    0.000    0.000   0.000   0.280   0.000   0.000   0.000   0.000   0.000   0.002   0.000   0.000   0.000   0.000
       355971535600.000    -0.000    207374.731   24091371.390     0.000   0.700   11786757520000000148929153677249216512.000     6.372    -0.043    1.513   0.295   1.962   1100769436.000    0.843     2.721   -0.000   0.020   15937812930.000     -0.043   0.000   0.000   0.003   0.000   0.001   0.010      0.000    0.000    0.000   0.000   0.280   0.000   0.000   0.000   0.000   0.000   0.002   0.000   0.000   0.000   0.000
       356088698500.000    -0.000    205931.643   23494524.550     0.000   0.700   11786757520000000148929153677249216512.000     6.406    -0.041    1.513   0.295   1.959   1097649965.000    0.843     2.418   -0.000   0.020   15820650060.000     -0.041   0.000   0.000   0.003   0.000   0.001   0.010      0.000    0.000    0.000   0.000   0.280   0.000   0.000   0.000   0.000   0.000   0.002   0.000   0.000   0.000   0.000
       356207860400.000    -0.000    204455.624   22898391.190     0.000   0.700   11786757520000000148929153677249216512.000     6.438    -0.038    1.513   0.296   1.955   1094296115.000    0.843     2.111   -0.000   0.020   15701488190.000     -0.038   0.000   0.000   0.003   0.000   0.001   0.010      0.000    0.000    0.000   0.000   0.280   0.000   0.000   0.000   0.000   0.000   0.002   0.000   0.000   0.000   0.000
       356390516500.000    -0.000    202170.683   22005682.050     0.000   0.700   11786757520000000148929153677249216512.000     6.483    -0.035    1.514   0.296   1.949   1088675748.000    0.843     1.638   -0.000   0.020   15518832090.000     -0.035   0.000   0.000   0.003   0.000   0.001   0.010      0.000    0.000    0.000   0.000   0.280   0.000   0.000   0.000   0.000   0.000   0.002   0.000   0.000   0.000   0.000

    You can easily plot this:
    
    >>> p = plt.figure()
    >>> p = plt.plot(starl['radius'],starl['temperature'],'k-')
    >>> p = plt.plot(starl[keep]['radius'],starl[keep]['temperature'],'r-',lw=2)
    
    .. image:: fileio_read_fgong_profile.png
    
    @param filename: name of the FGONG file
    @type filename: str
    @return: global parameters, local parameters
    @rtype: dict, record array
    """
    #   these are the standard definitions and units for FGONG
    glob_pars = [('star_mass', 'g'),
                 ('photosphere_r', 'cm'),
                 ('photosphere_L', 'erg/s'),
                 ('initial_z', None),
                 ('initial_x', None),
                 ('alpha', None),
                 ('phi', None),
                 ('xi', None),
                 ('beta', None),
                 ('lam', None),
                 ('ddP_drr_c', None),
                 ('ddrho_drr_c', None),
                 ('star_age', 'yr'),
                 ('na14', None),
                 ('na15', None)]
    loc_pars = [('radius', 'cm'),
                ('ln(m/M)', None),
                ('temperature', 'K'),
                ('pressure', 'kg/m/s2'),
                ('density', 'g/cm3'),
                ('X', None),
                ('luminosity', 'erg/s'),
                ('opacity', 'cm2/g'),
                ('eps_nuc', None),
                ('gamma1', None),
                ('grada', None),
                ('delta', None),
                ('cp', None),
                ('free_e', None),
                ('brunt_A', None),
                ('rx', None),
                ('Z' ,None),
                ('R-r', 'cm'),
                ('eps_grav', None),
                ('Lg', 'erg/s'),
                ('xhe3', None),
                ('xc12', None),
                ('xc13', None),
                ('xn14', None),
                ('xo16', None),
                ('dG1_drho', None),
                ('dG1_dp', None),
                ('dG1_dY', None),
                ('xh2', None),
                ('xhe4', None),
                ('xli7', None),
                ('xbe7', None),
                ('xn15', None),
                ('xo17', None),
                ('xo18', None),
                ('xne20', None),
                ('xh1', None),
                ('na38', None),
                ('na39', None),
                ('na40', None)]
    #-- start reading the file
    ff = open(filename,'r')
    lines = ff.readlines()
    #-- skip lines until we are at the definitions line
    while not len(lines[0].strip().split())==4:
        lines = lines[1:]
    #-- take care of data dimensions
    NN, ICONST, IVAR, IVERS = [int(i) for i in lines[0].strip().split()]
    if not ICONST == 15:
        raise ValueError('cannot interpret FGONG file: wrong ICONST')
    if not IVERS in [300, 250]:
        raise ValueError('cannot interpret FGONG file:'
                           ' wrong IVERS ({})'.format(IVERS))
    data = []
    #-- read in all the data
    for line in lines[1:]:
        data.append([line[0*16:1*16], line[1*16:2*16], line[2*16:3*16],
                     line[3*16:4*16], line[4*16:5*16]])
    data = np.ravel(np.array(data, float))
    starg = {glob_pars[i][0]:data[i] for i in range(ICONST)}
    data = data[15:].reshape((NN, IVAR)).T
    #-- reverse the profile to get center ---> surface
    data = data[:, ::-1]
    #-- make it into a record array and return the data
    starl = np.rec.fromarrays(data, names=[lp[0] for lp in loc_pars])
    return starg, starl


def read_ssm(ssm_file):
    """
    Read in a short summary file from ADIPLS.
    
    **Example usage:**
    
    >>> glob,loc = read_ssm('example.ssm')
    
    >>> print(glob)
    {'R': 335394129223.22, 'M': 1.7895599999987e+34, 'P_c': 4.1195872838613e+16, 'rho_c': 11.1267054491}
    >>> print(plt.mlab.rec2txt(loc))
           l        nz    sigma2       E   frequency
       2.000   -12.000     0.192   0.005       0.012
       2.000   -11.000     0.225   0.018       0.013
       2.000   -10.000     0.249   0.013       0.014
       2.000    -9.000     0.316   0.015       0.016
       2.000    -8.000     0.420   0.023       0.018
       2.000    -7.000     0.508   0.068       0.020
       2.000    -6.000     0.638   0.024       0.023
       2.000    -5.000     0.964   0.020       0.028
       2.000    -4.000     1.515   0.025       0.035
       2.000    -3.000     1.924   0.023       0.039
       2.000    -2.000     3.663   0.005       0.054
       2.000    -1.000    10.198   0.000       0.090
       2.000     1.000    14.965   0.000       0.110
       2.000     2.000    17.784   0.000       0.119
       2.000     3.000    26.922   0.000       0.147
       2.000     4.000    38.717   0.000       0.176
       2.000     5.000    53.377   0.000       0.207
       2.000     6.000    71.477   0.000       0.239
       2.000     7.000    92.856   0.000       0.273
       2.000     8.000   117.163   0.000       0.306
       2.000     9.000   143.938   0.000       0.340
        
    @param ssm_file: name of the short summary file
    @type ssm_file: str
    @return: global parameters, local parameters
    @rtype: dict, record array
    """
    with open(ssm_file,'rb') as ff:
        contents = ff.read()
    
    #-- global parameters
    fmt = '<i' + 6*'d'
    size = struct.calcsize(fmt)
    out = struct.unpack(fmt, contents[:size])
    contents = contents[size:]
    M, R, P_c, rho_c = out[3:7]
    starg = dict(M=M, R=R, P_c=P_c, rho_c=rho_c)
    
    #-- local parameters
    fmt = '<'+4*'i'+5*'d'+'ii'
    size = struct.calcsize(fmt)
    starl = np.zeros((5, len(contents)/size))
    for i in range(starl.shape[1]):
        out = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        starl[:, i] = out[4:-2]
    
    starl = np.rec.fromarrays(starl,
                              names=['l', 'nz', 'sigma2', 'E', 'frequency'])
    return starg, starl
    
def read_gsm(gsm_file):
    """
    Read in a grand summary file from ADIPLS.
    
    **Example usage:**
    
    >>> data = read_gsm('example.gsm')
    >>> print(plt.mlab.rec2txt(data))
       l     n    sigma2       E   varfreq   beta_nl   freqR       m
       2   -12     0.192   0.005     0.012     0.000   0.000   0.000
       2   -11     0.225   0.018     0.013     0.000   0.000   0.000
       2   -10     0.249   0.013     0.014     0.000   0.000   0.000
       2    -9     0.316   0.015     0.016     0.000   0.000   0.000
       2    -8     0.420   0.023     0.018     0.000   0.000   0.000
       2    -7     0.508   0.068     0.020     0.000   0.000   0.000
       2    -6     0.638   0.024     0.023     0.000   0.000   0.000
       2    -5     0.964   0.020     0.028     0.000   0.000   0.000
       2    -4     1.515   0.025     0.035     0.000   0.000   0.000
       2    -3     1.924   0.023     0.039     0.000   0.000   0.000
       2    -2     3.663   0.005     0.054     0.000   0.000   0.000
       2    -1    10.198   0.000     0.090     0.000   0.000   0.000
       2     1    14.965   0.000     0.110     0.000   0.000   0.000
       2     2    17.784   0.000     0.119     0.000   0.000   0.000
       2     3    26.922   0.000     0.147     0.000   0.000   0.000
       2     4    38.717   0.000     0.176     0.000   0.000   0.000
       2     5    53.377   0.000     0.207     0.000   0.000   0.000
       2     6    71.477   0.000     0.239     0.000   0.000   0.000
       2     7    92.856   0.000     0.273     0.000   0.000   0.000
       2     8   117.163   0.000     0.306     0.000   0.000   0.000
       2     9   143.938   0.000     0.340     0.000   0.000   0.000
    
    You can easily plot the contents of a GSM file:
    
    >>> p = plt.figure()
    >>> p = plt.vlines(np.sqrt(data['sigma2']),0,data['E'])
    
    .. image:: fileio_read_gsm_profile.png
       
    @param gsm_file: name of the grand summary file
    @type gsm_file: str
    @return: contents of the GSM file
    @rtype: record array
    """
    with open(gsm_file,'rb') as open_file:
        contents = open_file.read()
    
    fmt = '<i'
    size = struct.calcsize(fmt)
    out = struct.unpack(fmt, contents[:size])
    contents = contents[size:]
    
    pars = []
    while 1:
        #-- global parameters
        fmt = '<' + 38*'d'
        size = struct.calcsize(fmt)
        out = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        
        mode = list(out)
        
        fmt = '<'+8*'i'
        size = struct.calcsize(fmt)
        out = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        mode += list(out)[:-1]
        pars.append(mode)
        
        if len(contents)<72:
            break
        fmt = '<'+18*'i'
        size = struct.calcsize(fmt)
        out = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
    
    columns = np.array(['xmod', 'M', 'R', 'P_c', 'Rho_c', 'D5', 'D6', 'mu',
                        'D8', 'A2_xs', 'A5_xs', 'x1', 'sigma_Omega', 'xf',
                        'fctsbc', 'fcttbc', 'lambda', 'l', 'n', 'sigma2',
                        'sigmac2', 'y1max', 'xmax', 'E', 'Pe', 'PV', 'varfreq',
                        'ddsig', 'ddsol', 'y1_xs', 'y2_xs', 'y3_xs', 'y4_xs',
                        'z1max', 'xhatmax', 'beta_nl', 'freqR', 'm', 'in',
                        'nn', 'rich', 'ivarf', 'icase', 'iorign', 'iekinr'])
    pars = list(np.array(pars).T)
    pars[17] = np.array(pars[17], int)
    pars[18] = np.array(pars[18], int)
    keep = np.array([17, 18, 19, 23, 26, 35, 36, 37])
    data = np.rec.fromarrays([pars[i] for i in keep], names=list(columns[keep]))
    
    fmt = '<'+17*'i'
    size = struct.calcsize(fmt)
    out = struct.unpack(fmt, contents[:size])
    contents = contents[size:]
    
    return data
    
def read_aeig(aeig_file, nfmode=1):
    """
    Read in a simple eigenfunction file written by ADIPLS.
    
    **Example usage:**
    
    >>> data = read_aeig('example.aeig')
    >>> print(data[0])
    [  0.00000000e+00   8.94402416e-04   1.12688269e-03 ...,   9.99975382e-01
       9.99985580e-01   1.00000000e+00]
    >>> print(data[1])
    [ -0.00000000e+00   1.76464376e-07   2.22333235e-07 ...,   1.00002950e+00
       1.00002168e+00   1.00000000e+00]

    You can easily plot the contents of eigenfunction file:
    
    >>> p = plt.figure()
    >>> p = plt.plot(data[0],data[1],'k-')
    
    .. image:: fileio_read_aeig_profile.png
    
    @param aeig_file: name of the eigenfunction file
    @type aeig_file: str
    @return: x,y1,y2,y3,y4,z1,z2
    @rtype: 7xarray
    """
    with open(aeig_file, 'rb') as ff:
        contents = ff.read()
    
    if nfmode == 2:
        #-- read in the size of the arrays in the file
        fmt = '<'+2*'i'
        size = struct.calcsize(fmt)
        N1, N2 = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        #-- read in the 'x' axis
        fmt = '<'+N2*'d'
        size = struct.calcsize(fmt)
        x = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        x = np.array(x)
        #-- read in the size of the following arrays in the file
        fmt = '<'+2*'i'
        size = struct.calcsize(fmt)
        out = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        #-- some weird intermezzo that I don't understand: global parameters?
        fmt = '<'+50*'d'
        size = struct.calcsize(fmt)
        y = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        #-- the actual eigenfunctions
        N = 2
        fmt = '<'+N*N2*'d'
        size = struct.calcsize(fmt)
        z = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        z = np.array(z)
        y1 = z[0::N]
        y2 = z[1::N]
        #-- and an end integer
        fmt = '<'+'i'
        size = struct.calcsize(fmt)
        y = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        output = x, y1, y2
    
    elif nfmode == 1:
        fmt = '<'+1*'i'
        size = struct.calcsize(fmt)
        N1 = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        fmt = '<'+50*'d'
        size = struct.calcsize(fmt)
        y = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        #-- read in the size of the following arrays in the file
        fmt = '<'+1*'i'
        size = struct.calcsize(fmt)
        out = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        N2, = out
        #-- the actual eigenfunctions
        N = 7
        fmt = '<'+N*N2*'d'
        size = struct.calcsize(fmt)
        z = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        z = np.array(z)
        x = z[0::N]
        y1 = z[1::N]
        y2 = z[2::N]
        y3 = z[3::N]
        y4 = z[4::N]
        z1 = z[5::N]
        z2 = z[6::N]
        #-- and an end integer
        fmt = '<'+'i'
        size = struct.calcsize(fmt)
        y = struct.unpack(fmt, contents[:size])
        contents = contents[size:]
        output = x, y1, y2, y3, y4, z1, z2
    
    if len(contents):
        raise IOError("Error reading ADIPLS eigenfrequency file: "
                       "uninterpreted bytes remaining")
    return output

def read_amdl(filename):
    """
    Read in a binary AMDL file from ADIPLS.
    
    **Example usage:**
    
    >>> glob,loc = read_amdl('example.amdl')
    >>> print(glob)
    {'rho_c': 14.199697670849, 'M': 1.3919829e+34, 'P_c': 4.4188133989039e+16, 'mu': -1.0, 'R': 372356970623.75, 'D6': 111.46870869897, 'D5': 111.46714384251}
    >>> print(plt.mlab.rec2txt(loc[:5]))
           x      q/x3      Vg      G1        A       U
       0.000   220.980   0.000   1.588    0.000   2.995
       0.002   220.957   0.000   1.588    0.000   2.995
       0.004   220.883   0.001   1.588   -0.000   2.994
       0.007   220.652   0.005   1.588    0.000   2.993
       0.010   220.171   0.012   1.589   -0.000   2.991
    
    You can easily plot the contents of an AMDL file:
    
    >>> p = plt.figure()
    >>> p = plt.plot(loc['x'],loc['q/x3'],'k-')
    
    .. image:: fileio_read_amdl_profile.png
    
    @param filename: name of the file
    @type filename: str
    @return: global parameters, local parameters
    @rtype: dict, rec array
    """
    with open(filename,'rb') as ff:
        #-- read in all the bytes at once
        content = ff.read()
    #-- have a look at the header of the file
    header_fmt = '<iii'
    line_fmt = '<'+8*'d'
    head = struct.calcsize(header_fmt)
    line = struct.calcsize(line_fmt)
    header = content[:head]
    content = content[head:]
    model_name, icase, nn = struct.unpack(header_fmt, header)
    
    #-- read in the first line of the data. This contains
    #   global information on the model, and also the number
    #   of columns in the datafile.
    record = content[:line]
    content = content[line:]
    M, R, P_c, rho_c, D5, D6, mu, D8 = struct.unpack(line_fmt, record)
    
    #-- derive the number of variables in this model file
    if D8 >= 100:
        nvars = 9
    elif D8 >= 10:
        nvars = 7
    else:
        nvars = 6
    
    #-- then read in the rest, line by line
    line_fmt = '<'+nvars*'d'
    line = struct.calcsize(line_fmt)
    starl = []
    for i in range(nn):
        record = content[:line]
        content = content[line:]
        data = struct.unpack(line_fmt, record)
        starl.append(data)
    
    #-- end of the file
    line_fmt = '<i'
    record = content[:line]
    content = content[line:]
    data = struct.unpack(line_fmt, record)
    
    #-- wrap up the local and global information on the star
    starl = np.rec.fromarrays(np.array(starl).T,
                              names=['x','q/x3','Vg','G1','A','U'])
    starg = {'M':M, 'R':R, 'P_c':P_c, 'rho_c':rho_c,
             'D5':D5, 'D6':D6, 'mu':mu}
    if len(content):
        raise ValueError('uninterpreted data in file').with_traceback(filename)
    
    return starg, starl



def read_mesa(filename, only_first=False):
    """
    Read *.data files from MESA.
    
    This returns a record array with the global and local parameters (the latter
    can also be a summary of the evolutionary track instead of a profile if
    you've given a 'history.data' file.
    
    The stellar profiles are given from surface to center.
    
    @param filename: name of the log file
    @type filename: str
    @param only_first: read only the first model (or global parameters)
    @type only_first: bool
    @return: list of models in the data file (typically global parameters, local parameters)
    @rtype: list of rec arrays
    """
    models = []
    new_model = False
    header = None
    #-- open the file and read the data
    with open(filename, 'r') as ff:
        #-- skip first 5 lines when difference file
        if os.path.splitext(filename)[1]=='.diff':
            for i in range(5):
                line = ff.readline()
            models.append([])
            new_model = True
        while 1:
            line = ff.readline()
            if not line:
                break # break at end-of-file
            line = line.strip().split()
            if not line:
                continue
            #-- begin a new model
            if line[0] == '1' and line[1] == '2':
                #-- wrap up previous model
                if len(models):
                    model = np.array(models[-1], float).T
                    models[-1] = np.rec.fromarrays(model, names=header)
                    if only_first:
                        break
                models.append([])
                new_model = True
                continue
            #-- next line is the header of the data, remember it
            if new_model:
                header = line
                new_model = False
                continue
            models[-1].append(line)
    if len(models)>1:
        model = np.array(models[-1], float).T
        models[-1] = np.rec.fromarrays(model, names=header)
    
    #-- remove duplicate models
    if os.path.basename(filename)=='history.data':
        keep = np.hstack([np.diff(models[-1]['model_number'])>0, True])
        models[-1] = models[-1][keep]
    
    return models


def read_mesa_gyre(filename):
    """
    Read in a GYRE model from MESA.
    
    Note: epsilon_T and epsilon_rho in the MESA GYRE format columns are
    actually epsilon_T*eps in the GYRE format (see "read_mesa" in
    "gyre_eqmodel_io.f90")!
    """
    
    with open(filename,'r') as ff:
        # global variables
        head = ff.readline().strip().split()
        starg = dict(n=int(head[0]), star_mass=float(head[1]),
                     photosphere_r=float(head[2]),
                     photosphere_L=float(head[3]))
        # local variables
        starl = np.zeros((19, starg['n']))
        for i, line in enumerate(ff.readlines()):
            line = line.replace('D','E')
            if i < (starg['n']-1):
                starl[:, i] = [float(col) for col in line.strip().split()]
            # there is a typo in the MESA format
            else:
                contents = [col for col in line.strip().split()]
                contents[2] = 1e100
                contents = [float(col) for col in contents]
                starl[:, i] = contents
        starl = [list(col) for col in starl]
        starl[0] = np.array(starl[0], int)
        starl = np.rec.fromarrays(starl, names=['zone', 'radius', 'w',
              'luminosity', 'pressure', 'temperature', 'density', 'nabla',
              'brunt_N2', 'cv', 'cp', 'chiT', 'chiRho', 'opacity', 'opacity_T',
              'opacity_rho', 'epsilon', 'epsilon_T', 'epsilon_rho'])
    return starg, starl

def write_mesa_gyre(filename, starg, starl):
    """
    Write a GYRE model from MESA to a file.
    
    This is the output counterpart of "read_mesa_gyre".
    """
    
    names = ['zone', 'radius', 'w',
              'luminosity', 'pressure', 'temperature', 'density', 'nabla',
              'brunt_N2', 'cv', 'cp', 'chiT', 'chiRho', 'opacity', 'opacity_T',
              'opacity_rho', 'epsilon', 'epsilon_T', 'epsilon_rho']
    
    template = '{:6.0f}' + '{:20.12E}'*(len(names)-1) + '\n'
    
    with open(filename,'w') as ff:
        # Write the header
        ff.write('{:6.0f}{:20.12E}{:20.12E}{:20.12E}\n'.format(len(starl),
              starg['star_mass'], starg['photosphere_r'],
              starg['photosphere_L']))
        # Write the body
        for i in range(len(starl)):
            ff.write(template.format(*[starl[name][i] for name in names]))
    # That's it!        
            


def list_mesa_data(filename='profiles.index'):
    """
    Return a chronological list of *.data files in a MESA LOG directory
    """
    number, priority, lognr = np.loadtxt(filename, skiprows=1).T
    logfiles = [os.path.join(os.path.dirname(filename),
                             'profile{:.0f}.data'.format(nr)) for nr in lognr]
    return list(np.array(number, int)), logfiles




def read_gyre(filename, add_extra=False):
    """
    Read a GYRE HDF5-format output file.
    
        
    **Example usage**

    >>> data,local = fileio.read_gyre('eigvals.h5')
    
    You can access information by column:
    
    >>> local['omega']
    array([  2.01844773+0.j,   3.51781378+0.j,   5.02055334+0.j,
             6.38583592+0.j,   7.93991267+0.j,   9.50455841+0.j,
            11.01185876+0.j,  12.63161261+0.j,  14.26029392+0.j,
            15.84866171+0.j,  17.43948807+0.j,  19.04329558+0.j,
            20.66636443+0.j,  22.29938853+0.j,  23.91005215+0.j,
            25.52563311+0.j,  27.14974168+0.j])
    >>> local['n_g']
    array([ 3,  4,  5,  6,  7,  7,  9, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=int32)
    
    But also by row:
    
    >>> local[0]
    (1.0, 3, 0, (0.35264730283221724+0j), 2, (2.018447726710389+0j))        
    
    You can easily select for example all frequencies above a certain threshold:
    
    >>> selection = local[ (local['omega'].real > 10.) ]
    >>> selection['n_p']
    array([ 9, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=int32)
    
    If you just want to get information on the names of the columns, simply do:
    
    >>> local.dtype.names
    
    For a list of possible column names, see `GYRE (Output Files) <https://bitbucket.org/rhdtownsend/gyre/wiki/Output%20Files>`_
    
    @param filename: Input file
    @type filename: string
    @param add_extras: add extra information on kinetic energy density,
                       acoustic energy, buoyancy and gravitational energy
                       (available as ``E_T``, ``E_C``, ``E_N`` and ``E_G``)
    @return: Dictionary containing the scalar information from the GYRE file. In the
             case of eigenvalue files, this dictionary is empty, and
             Record array containing array information with the column names from
             the GYRE file. For eigenvalue files, this contains all the frequency
             information. For eigenfunction files, this contains all the
             eigenfunctions.
    @rtype: dictionary, record array
    """

    # Open the file

    open_file = h5py.File(filename, 'r')

    # Read attributes

    data = dict(list(zip(list(open_file.attrs.keys()), list(open_file.attrs.values()))))

    # Read datasets

    for k in list(open_file.keys()) :
        data[k] = open_file[k][...]

    # Close the file

    open_file.close()

    complex_dtype = np.dtype([('re', '<f8'), ('im', '<f8')])
    
    # Convert the data into a record array and the global information in a dict
    
    local = []
    local_names = []
    
    for k in list(data.keys()):
        
        # Convert items to complex
        
        if(data[k].dtype == complex_dtype) :
            data[k] = data[k]['re'] + 1j*data[k]['im']
        
        # Save array information to local record array, keep scalar information
        
        if not np.isscalar(data[k]):
            local.append(data.pop(k))
            local_names.append(k)
            
    
    
    # Add some extra variables:
    if add_extra:
        local_ = np.rec.fromarrays(local, names=local_names)
        R = data['R_star']
        M = data['M_star']
        t_dynamic = np.sqrt(R**3 / (GG*M))
        data['t_dyn'] = t_dynamic
        
        norm = np.sqrt(4*np.pi)
        norm = 1.0
        
        l = data['l']
        dr = local_['xi_r'] * R * norm
        r = local_['x'] * R
        p = local_['p']
        rho = local_['rho']
        phip = local_['phip'] * GG*M/R * norm
        dphip_dr = local_['dphip_dx'] * GG*M/R**2 * norm
        delp = local_['delp'] * norm
        g = GG * local_['m'] / r**2
        omega = data['omega'] / t_dynamic
        N2 = local_['As'] * g / r
        G1 = local_['Gamma_1']
                
        # Compute Eulerian pressure perturbation:
        
        # pressure gradient from hydrostatic equilibrium
        dp_dr = -rho * g 
        # Eulerian pressure perturbation from Lagrangian
        pp = delp*p - dr * dp_dr

        # T is proportional to the kinetic energy density
        T = dr**2 + l*(l+1) / (r**2 * omega**2) * (pp / rho + phip)
        T = T * rho * r**2
        
        # C is a measure for the acoustic energy
        Sl2 = G1 * p / (r**2 * rho) # omitted the l*(l+1) thing, it is cancelled in C anyway
        C = 1.0 / Sl2 * (pp / rho)**2 / r**2
        C = C * rho * r**2
        
        # N is a measure for the buoyancy energy
        N = N2 * dr**2
        N = N * rho *r **2
        
        # G is a measure for the gravitational energy
        G = -1.0 / (4 * np.pi * rho * GG) * (dphip_dr + l*(l+1) * phip / r)
        G = G * rho * r**2
        
        charpinet = dr**2*N2 + pp**2/(G1*p*rho) + (pp/(G1*p) + dr * N2/g)
        charpinet = charpinet * rho * r**2
        
        T[np.isnan(T)] = 0
        C[np.isnan(C)] = 0
        N[np.isnan(N)] = 0
        G[np.isnan(G)] = 0
        charpinet[np.isnan(charpinet)] = 0
        
        total = (C+N+G)        
        
        for arr in [T, C, N, G]:
            arr[0] = 0.0
        
        local = local + [T, C, N, G, charpinet, N2, Sl2*l*(l+1)]
        local_names = local_names + ['E_T','E_C','E_N','E_G','charpinet','brunt_N2','Sl2']
        
    local = np.rec.fromarrays(local, names=local_names)    
    # Return the global and local information

    return data, local




if __name__ == "__main__":
    
    import doctest
    import matplotlib.pyplot as plt
    doctest.testmod()
    plt.show()
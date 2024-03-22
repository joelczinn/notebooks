import numpy as np
from nonad.plot_modes import main as _plot_modes
import glob
import sys
import os
def main(filename_root='rgb', ad='ad', dir=dir):
    
    # that will make a <filename>.astero file with dnu_scal and dnu_obs columns
    # append all the *.astero files together:
    
    # JCZ 221018
    # removed these lines because want to actually append, so don't want to delete existing files.
    # print 'removing {}.astero.in if it exists.'.format(filename_root)
    # os.system('rm {}.astero.in'.format(filename_root))
    
    try:
        # JCZ 231018
        # picks out the model number, given a name like:
        # m100_good.mesa02_0247
        # (would pick out 247)
        model = int(filename_root.split('_')[-1])
    except:
        model = 0
    # run GYRE on this file --- if it hasn't already been run. !!! change this to overwrite things
    _plot_modes(profile=filename_root, model=model, ad=ad, dir=dir)

    # os.system('cat {}*.astero.in > {}.astero.in'.format(filename_root, filename_root))
    print('appended to {}.{}.astero.in!'.format(filename_root, ad))

if __name__ == '__main__':
        filename_root = sys.argv[1]
        ad = sys.argv[2]
        try:
            dir = sys.argv[4]
        except:
            dir = './'
        main(filename_root, ad=ad, dir=dir)

import pandas as pd
import numpy as np

from astropy import units as u, constants as const

def read_marcs_to_fv(fname, atom_data, final_atomic_number=30):
    marcs_model1 = pd.read_csv(fname,skiprows=24, nrows=56, 
                               delim_whitespace=True, index_col='k')
    marcs_model2 = pd.read_csv(fname,skiprows=81, nrows=56, 
                               delim_whitespace=True, index_col='k')
    del marcs_model2['lgTauR']
    marcs_model = marcs_model1.join(marcs_model2)
    marcs_model.columns = [item.lower() for item in marcs_model.columns]
    marcs_model['lgtaur'] = 10**marcs_model['lgtaur']
    marcs_model['lgtau5'] = 10**marcs_model['lgtau5']
    
    marcs_model = marcs_model.rename(columns={'lgtaur':'tau_ref', 'lgtau5':'tau_500'})
    with open(fname) as fh:
        marcs_lines = fh.readlines()
    marcs_abundance_scale_str = ' '.join([item.strip() for item in marcs_lines[12:22]])
    marcs_abundances = pd.DataFrame(data=map(np.float64, marcs_abundance_scale_str.split()), columns=['abundance_scale'])
    marcs_abundances.replace({-99:np.nan}, inplace=True)
    marcs_abundances = marcs_abundances.set_index(np.arange(1, len(marcs_abundances)+1))
    marcs_abundances.index.name = 'atomic_number'
    marcs_abundances['mass_density'] = ((10**marcs_abundances['abundance_scale']) * 
                                        atom_data.atom_data.mass)
    marcs_abundances['mass_abundance'] = marcs_abundances['mass_density'] / marcs_abundances['mass_density'].sum()
    marcs_model = marcs_model[::-1]
    
    temps = marcs_model.t.values[None].T
    
    marcs_model_fv = pd.DataFrame(data=0.5 * (marcs_model.iloc[:-1].values + marcs_model.iloc[1:].values), 
                                columns=marcs_model.columns)
    marcs_model_fv['r_inner'] = marcs_model['depth'].iloc[:-1].values
    marcs_model_fv['r_outer'] = marcs_model['depth'].iloc[1:].values
    marcs_model_fv['cell_length'] = -(marcs_model_fv['r_outer'].values - marcs_model_fv['r_inner'].values)
    
    marcs_abundances_all = pd.DataFrame(columns=marcs_model_fv.index.values, 
                    index=marcs_abundances.index)
    for i in range(len(marcs_abundances_all.columns)):
        marcs_abundances_all[i] = marcs_abundances['mass_abundance']
    marcs_abundances_all = marcs_abundances_all
    return marcs_model_fv, marcs_abundances_all[:final_atomic_number], temps

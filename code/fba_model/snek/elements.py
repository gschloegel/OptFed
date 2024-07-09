'''
This moduel contains several functions for the elemental analysis of molecules and
the calculation of molecular weights from sum formulas.
'''

import pkg_resources
import logging
import numpy as np
import pandas as pd
import cobra
from .core import sensitive_optimize
from collections import Counter

def _check_for_nonetype(formula):
    '''
    Check for None types in metabolite formulae.
    '''

    if formula is None:
        return True
    else:
        return False

def get_unique_elements(obj):
    '''
    Get a list of unique elments from either

    * a chemical formula
    * a CobraPy metabolite object
    * a CobraPy reaction object
    * a CobraPy model object.

    When a reaction or a model is parsed, all metabolites of the reaction/model
    are considered.

    Parameters
    ----------
        obj : either Str or cobra.core.metabolite.Metabolite or cobra.core.reaction.Reaction or cobra.core.model.Model

    Returns
    -------
    list_elements :  List
                     List of unique elements, e.g. ``['H','O']``.
    '''

    # If the chemical formulae of reactions are not define, a None type is returned.
    # These are excepted, tracked, and a warning is printed.
    nr_none_types = 0

    # test if metabolite
    if isinstance(obj,cobra.core.metabolite.Metabolite):
        tmp_formula = obj.formula
        if _check_for_nonetype(tmp_formula):
            nr_none_types += 1
            list_elements = []
        else:
            list_elements = unique_elements(tmp_formula)

    # test if reaction or model
    elif isinstance(obj,cobra.core.reaction.Reaction) or isinstance(obj,cobra.core.model.Model):
        list_elements = []

        for metabolite in obj.metabolites:
            tmp_formula = metabolite.formula
            if _check_for_nonetype(tmp_formula):
                nr_none_types += 1
            else:
                tmp_element_list = unique_elements(tmp_formula)
                for element in tmp_element_list:
                    if element not in list_elements:
                        list_elements.append(element)

    # test if string
    elif isinstance(obj,str):
        if _check_for_nonetype(obj):
            nr_none_types += 1
        else:
            list_elements = unique_elements(obj)

    # print warning if metabolites are missing chemical formulae
    if nr_none_types == 1:
        logging.warning(f"There is {nr_none_types} metabolite with missing formula.")
    elif nr_none_types > 1:
        logging.warning(f"There are {nr_none_types} metabolites with missing formulae.")

    return list_elements

def count_atom(formula,element):
    """
    Counts the number of atoms of an element in a chemical formula.
    Works with multiple-letter element names and floats.
    Case sensitive!

    Parameters
    ----------
        formula  :  Str
                    Chemical formula, e.g. ``'H2O'``.
        element  :  Str
                    Element name, e.g. ``'H'``.

    Returns
    -------
        nr       :  float
                    Number of atoms as float, e.g. ``2.0``.
    """
    found = 0
    while formula.find(element) != -1:
        idx = formula.find(element)+len(element)-1
        tmp = ''
        for f in formula[idx+1:]:
            if f.isupper() == False:
                tmp += f
            else:
                break
        if len(tmp) == 0:
            nr = '1'
            found = 1
            break

        elif tmp[0].islower() == False:
            nr = tmp
            found = 1
            break
        else:
            found = 0
            formula = formula[idx+1:]

    if found == 0:
        nr = '0'
    return float(nr)

def unique_elements(formula):
    '''
    Get a list of unique elements from a chemical formula.

    Parameters
    ----------
        formula   : Str
                    Chemical formula, e.g. ``'H2O'``.

    Returns
    -------
        list_elements :  List
                         List of unique elements, e.g. ``['H','O']``.
    '''

    list_elements = []

    tmp = ''
    for letter in formula:
        if not letter.isnumeric() and not letter == '.':
            if letter.isupper():
                if len(tmp) > 0 and tmp not in list_elements:
                    list_elements.append(tmp)
                tmp = ''
                tmp += letter
            else:
                tmp += letter

    if len(tmp) > 0 and tmp not in list_elements:
        list_elements.append(tmp)

    return list_elements

def formula_to_dictionary(formula):
    '''
    Get the elemental composition of a molecule as dictionary.
    Keys correspond to elements, values to their number of appearance.

    Parameters
    ----------
        formula  : Str
            Chemical formula, e.g. ``'H2O'``.

    Returns
    -------
        dictionary : Dict
            Dictionary with unique elements as keys
            and number of atoms per element as items,
            e.g. ``{'H': 2.0, 'O': 1.0}``.

    See Also
    --------
    dictionary_to_formula : Inverse function to ``formula_to_dictionary``.
    '''

    list_elements = unique_elements(formula)
    dictionary  = {}
    for element in list_elements:
        dictionary[element] = count_atom(formula,element)

    return dictionary

def dictionary_to_formula(dictionary):
    '''
    Get the chemical sum formula of a molecule as string from a dictionary.

    Parameters
    ----------
        dictionary : Dict
            Dictionary with unique elements as keys
            and number of atoms per element as items,
            e.g. ``{'H': 2.0, 'O': 1.0}``.
    Returns
    -------
    formula  :  Str
                    Chemical formula, e.g. ``'H2O'``.

    See Also
    --------
    formula_to_dictionary : Inverse function to ``dictionary_to_formula``.
    '''

    formula = ''
    for element_name,element_count in sorted(dictionary.items()):
        formula += element_name
        if np.isclose(element_count,round(element_count),rtol=0):
            formula += str(round(element_count))
        else:
            formula += str(element_count)

    return formula

def molecular_weight(formula):
    '''
    Calculates molecular weight from the chemical sum formula. The mass table is derived from the CobraPy package [1]_.

    Parameters
    ----------
    formula : Str
        String of chemical sum formula. Capitalization has to be correct.

    Returns
    -------
    molecular_weight : Float
        Float of molecular weight in g/mol.

    Notes
    -----
    .. [1] Ebrahim, A., Lerman, J. A., Palsson, B. O., & Hyduke, D. R. (2013).
        COBRApy: constraints-based reconstruction and analysis for python.
        BMC systems biology, 7(1), 1-6.
        https://doi.org/10.1186/1752-0509-7-74
    '''

    dictionary = formula_to_dictionary(formula)

    # the molecular masses of the elements is downloaded from CobraPy
    # https://github.com/opencobra/cobrapy/blob/devel/src/cobra/core/formula.py
    location = pkg_resources.resource_stream(__name__,'data/mol_mass_table.csv')
    mol_mass_table = pd.read_csv(location)

    # check if all elements are covered in the mol_mass_table
    for element in dictionary:
        assert element in mol_mass_table.loc[:,'Symbol'].unique(), f'Element {element} not in molecular mass table.'

    molecular_weight = 0
    for element,number_of_atoms in dictionary.items():
        molecular_weight += mol_mass_table[mol_mass_table['Symbol']==element]['AtomicMass'].values[0]*number_of_atoms

    return molecular_weight

def get_pDNA_formula(sequence,return_dict=False):
    '''
    Returns the chemical sum formula of the non-charged pDNA molecule.

    Note
    ----
    The sum formula is only correct for circular, double stranded DNA.

    Parameters
    ----------
    sequence : Str
        String of pDNA sequence (only ``['A','T','C','G']`` are allowed).
    return_dict : Bool, default=False
        If True, additionally a dictionary with elements and counts in returned.

    Returns
    -------
    sum_formula : Str
        Chemical sum formula of the pDNA.
    sum_formula_dic : Dict
        Dictionary with elements and counts, only returned if return_dic=True.
    '''

    # H2O was subtracted from all dNMPs
    dAMP_delta_H2O = 'C10H12N5O5P'
    dTMP_delta_H2O = 'C10H13N2O7P'
    dCMP_delta_H2O =  'C9H12N3O6P'
    dGMP_delta_H2O = 'C10H12N5O6P'

    base_names = ['A','T','C','G']
    base_formulas_fw = [dAMP_delta_H2O,dTMP_delta_H2O,dCMP_delta_H2O,dGMP_delta_H2O]
    base_formulas_rv = [dTMP_delta_H2O,dAMP_delta_H2O,dGMP_delta_H2O,dCMP_delta_H2O]

    assert len(sequence) == sum([sequence.count(i) for i in base_names])

    sum_formula_dict = {'C':0,'H':0,'N':0,'O':0,'P':0}

    # forward strand
    for name, formula in zip(base_names,base_formulas_fw):
        tmp_count = sequence.count(name)
        tmp_elements = formula_to_dictionary(formula)
        for element in tmp_elements:
                sum_formula_dict[element] += tmp_count * tmp_elements[element]
    # reverse strand
    for name, formula in zip(base_names,base_formulas_rv):
        tmp_count = sequence.count(name)
        tmp_elements = formula_to_dictionary(formula)
        for element in tmp_elements:
                sum_formula_dict[element] += tmp_count * tmp_elements[element]

    sum_formula = ''
    for element in sum_formula_dict:
        sum_formula += element
        sum_formula += str(round(sum_formula_dict[element]))

    if return_dict:
        return sum_formula,sum_formula_dict
    else:
        return sum_formula

from collections import Counter

def get_protein_formula(sequence,return_dict=False):
    '''
    Returns the chemical sum formula of the non-charged protein.

    Parameters
    ----------
    sequence : Str
        String of amino acid sequence (only one-letter code of 20 conventional amino acids allowed).
    return_dic : Bool, default=False
        If True, additionally a dictionary with elements and counts in returned.

    Returns
    -------
    sum_formula : Str
        Chemical sum formula of the protein.
    sum_formula_dic : Dict
        Dictionary with elements and counts, only returned if ``return_dic = True``.
    '''

    # chemical formulas of all amino acids
    formula_dic = {'A': 'C3H7NO2', 'C': 'C3H7NO2S', 'D': 'C4H6NO4', 'E': 'C5H8NO4', 'F': 'C9H11NO2', 'G': 'C2H5NO2',
               'H': 'C6H9N3O2', 'I': 'C6H13NO2', 'K': 'C6H15N2O2', 'L': 'C6H13NO2', 'M': 'C5H11NO2S', 'N': 'C4H8N2O3',
               'P': 'C5H9NO2', 'Q': 'C5H10N2O3', 'R': 'C6H15N4O2', 'S': 'C3H7NO3', 'T': 'C4H9NO3', 'V': 'C5H11NO2',
               'W': 'C11H12N2O2', 'Y': 'C9H11NO3'}
    # initialize total sum formula
    sum_formula_dict = {'C':0,'H':0,'N':0,'O':0,'S':0}
    # initialize sum of amino acids
    sum_aa = 0
    # number of amino acids in sequence
    sequence_dic = Counter(sequence)

    # check if all amino acids are covered by formula_dic
    for AA in sequence:
        assert AA in formula_dic.keys(), f"Non-conventional amio acids are not supported: {AA}."


    # just sum up all atoms per amino acid and its count in the protein
    for AA, AA_count in sequence_dic.items():
        sum_aa      += AA_count
        tmp_elements = formula_to_dictionary(formula_dic[AA])
        for element,element_count in tmp_elements.items():
            sum_formula_dict[element] += element_count * AA_count

    # remove one H2O for every peptide synthesis reaction (i.e., the number of amino acids in the protein -1)
    sum_formula_dict['O'] -= sum_aa-1
    sum_formula_dict['H'] -= (sum_aa-1) * 2

    sum_formula = dictionary_to_formula(sum_formula_dict)

    if return_dict:
        return sum_formula,sum_formula_dict
    else:
        return sum_formula

def element_flux_coefficient(model,element,reaction):
    """
    Returns the amount of an user-defined chemical element consumed/produced in a CobraPy reaction.

    Parameters
    ----------
    model : cobra.core.model.Model
        CobraPy model.
    element : Str
        Chemical element as string, e.g. 'C'.
    reaction : Str
        Cobra reaction id as string.

    Returns
    -------
    element_flux_coefficient : Float
        Elemental flux of the reaction (only non-0 when not mass balanced).
    """

    element_flux_coefficient = 0
    for meta in model.reactions.get_by_id(reaction).metabolites:
        stoichiometric_coefficient = model.reactions.get_by_id(reaction).metabolites[meta]
        number_of_atoms = count_atom(meta.formula,element)
        element_flux_coefficient += stoichiometric_coefficient * number_of_atoms

    return element_flux_coefficient

def element_fluxes(model,element,solution=None):
    """
    Returns dictionary of non-zero element fluxes.
    Element fluxes are only non-zero when a reaction is not mass balanced,
    typically exchange, drain, and sink reactions.

    If no solution object is provided a pFBA is performed.
    If a solution object is provided metabolite fluxes are extracted from there.
    We recommend to do this analysis only solutions from pFBA.

    Parameters
    ----------
    model : cobra.core.model.Model
            CobraPy model.
    element : Str
        Chemical element as string, e.g. 'C'.
    solution : cobra.core.solution.Solution, default=None
        CobraPy solution object.

    Returns
    -------
    element_fluxes : pandas.DataFrame
               Pandas dataframe of non-zero element fluxes. The sign of fluxes
               corresponds to the sign of fluxes in the solution object.
               The index of the data frame corresponds to reaction ids,
               the first column to the absolute element flux in mmol/(g biomass h),
               and the second column to the relative element flux in %.
    """

    if isinstance(solution,type(None)):
        solution = sensitive_optimize(model,pFBA=True)

    # calculate fluxes
    element_fluxes = {}
    for r in model.reactions:
        rate = element_flux_coefficient(model,element,r.id)
        if rate != 0 and solution.fluxes[r.id] != 0:
            element_fluxes[r.id] = [solution.fluxes[r.id]*rate]

    # raise error if there are no fluxes in the solution
    if len(element_fluxes) == 0:
        raise ValueError(f'No {element} fluxes were found.')

    # do a sanity check
    control = 0
    for flux in element_fluxes:
        control += element_fluxes[flux][0]
    assert np.isclose(control,0,atol=model.tolerance,rtol=0), 'Fluxes do not add up. Are all non-boundary reactions mass balanced?'

    # create nice dataframe
    element_fluxes = pd.DataFrame(element_fluxes).T
    abs_col_name = f'{element}_flux'
    element_fluxes.columns = [abs_col_name]
    element_fluxes = element_fluxes.sort_values(abs_col_name,ascending=False)
    total_flux = np.sum(element_fluxes[element_fluxes[abs_col_name] > 0].values)
    element_fluxes.insert(1,abs_col_name+'%',element_fluxes[abs_col_name]/total_flux*100)

    return element_fluxes

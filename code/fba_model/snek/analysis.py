'''
This module contains several functions for the analysis of CobraPy models.
'''

import cobra
import numpy as np
import pandas as pd
from .core import sensitive_optimize
from .elements import *

def get_constrained_reactions(model):
    '''
    Returns data frames of all reactions that have constrained flux bounds.
    I.e. something else than (lower bound, upper bound)

    * ``(-1000, 1000)``,
    * ``(    0, 1000)``, or
    * ``(-1000,    0)``.

    Parameters
    ----------
        model   : cobra.core.model.Model
                  CobraPy model.

    Returns
    -------
        constrained  :  pandas.DataFrame
                        Pandas data frame of constrained reactions with name,
                        id, lower bound, upper bound as columns.
    '''

    constrained = []
    for reaction in model.reactions:
        if reaction.bounds == (0,1000):
            pass
        elif reaction.bounds == (-1000,1000):
            pass
        elif reaction.bounds == (-1000,0):
            pass
        elif reaction.bounds == (0,0):
            constrained.append([reaction.name,reaction.id,*reaction.bounds])
        else:
            constrained.append([reaction.name,reaction.id,*reaction.bounds])
    constrained = pd.DataFrame(constrained,columns=['name','id','lower_bound','upper_bound'])
    constrained.index = constrained['id'].values
    constrained = constrained.sort_values(['upper_bound','lower_bound'],ascending=False)
    return constrained

def in_bounds(model):
    """
    Returns all boundary reactions that have a non-zero lower bound as data frame.

    Parameters
    ----------
    model : cobra.core.model.Model
        CobraPy Model.

    Returns
    -------
    bounds : pandas.DataFrame
        Data frame of boundary reactions with non-zero lower bound.
    """

    bounds ={}
    for reaction in model.boundary:
        if reaction.lower_bound != 0:
            bounds[reaction.id] = [reaction.name,reaction.id,reaction.lower_bound,reaction.upper_bound]
    bounds = pd.DataFrame(bounds).T
    bounds.columns = columns=['name','id','lower_bound','upper_bound']
    bounds = bounds.sort_values(['lower_bound','upper_bound'],ascending=False)
    return bounds

def out_bounds(model):
    """
    Returns all boundary reactions that have a non-zero upper bound as data frame.

    Parameters
    ----------
    model : cobra.core.model.Model
        CobraPy Model.

    Returns
    -------
    bounds : pandas.DataFrame
        Data frame of boundary reactions with non-zero upper bound.
    """

    bounds ={}
    for reaction in model.boundary:
        if reaction.upper_bound != 0:
            bounds[reaction.id] = [reaction.name,reaction.id,reaction.lower_bound,reaction.upper_bound]
    bounds = pd.DataFrame(bounds).T
    bounds.columns = columns=['name','id','lower_bound','upper_bound']
    bounds = bounds.sort_values(['upper_bound','lower_bound'],ascending=False)
    return bounds

def in_flux(model,solution=None):
    """
    Returns all boundary reactions with flux < 0  as data frame.

    Parameters
    ----------
    model : cobra.core.model.Model
        CobraPy model.
    solution : cobra.core.solution.Solution, default=None
        CobryPy solution object. If None a pFBA is performed.

    Returns
    -------
    flux : pandas.DataFrame
        Data frame with all exchange reactions with flux < 0.

    """

    if isinstance(solution,type(None)):
        solution = sensitive_optimize(model,pFBA=True)

    flux= {}
    for reaction in model.boundary:
        if solution[reaction.id] < 0:
            flux[reaction.id] = [reaction.name,reaction.id,solution[reaction.id]]
    flux = pd.DataFrame(flux).T
    flux.columns = columns=['name','id','flux']
    flux = flux.sort_values('flux',ascending=True)
    return flux

def out_flux(model,solution=None):
    """
    Returns all boundary reactions with flux > 0  as data frame.

    Parameters
    ----------
    model : cobra.core.model.Model
        CobraPy model.
    solution : cobra.core.solution.Solution, default=None
        CobryPy solution object. If None a pFBA is performed.

    Returns
    -------
    flux : pandas.DataFrame
        Data frame with all boundary reactions with flux > 0.

    """

    if isinstance(solution,type(None)):
        solution = sensitive_optimize(model,pFBA=True)

    flux= {}
    for reaction in model.boundary:
        if solution[reaction.id] > 0:
            flux[reaction.id] = [reaction.name,reaction.id,solution[reaction.id]]
    flux = pd.DataFrame(flux).T
    flux.columns = columns=['name','id','flux']
    flux = flux.sort_values('flux',ascending=False)
    return flux

def _michaelis_menten(c,vmax,km):
    """
    Simple implementation of a Michaelis Menten enzyme kinetic.
    v = - vmax*c/(k_M+c)
    Attention: returns negative fluxes.
    Attention: be aware of units.


    Parameters
    ----------
    c : float
        Concentration.
    vmax : float
        Maximum reaction rate.
    km : float
        K_M as defined in Michaelis Menten kinetics.

    Returns
    -------
    v : float
        Reaction rate, (negative sign!).

    """
    if c <= 0:
        v = 0
    else:
        v = -vmax * c / (km + c)
    return v

def _lex_dfba(model,compounds,y_zero,time,objectives,objectives_direction,dynamic_constraints):
    """
    Lexicographic Dynamic FBA

    Parameters
    ----------
    compounds  : list
        List of reactions to track with size n
    y_zero     : list
        List of start concentrations of reaction metabolites with size n
    time       : list
        List with t0, tmax, and dt
    objectives : list
        List of reactions to optimize with size m
    objectives_direction : list
        List of direction of reactions to optimize ('max' or 'min') with size m
    dynamic_constraints  : list
        List of lists in form of ``[['reaction_id',vmax,km],...]``

    Returns
    -------
    y : dict
        Dictionary of tracked reaction concentrations
    f : dict
        Dictionary of tracked reaction fluxes
        """


    y = {}
    f = {}

    compounds = compounds # compound exchange reactions
    y[0] = y_zero # compound concentrations
    t0,tmax,dt = time # start time, max time, time step size
    objectives = objectives # list of reactions for objective function
    objectives_direction = objectives_direction # direction of objetives

    stat = ''
    for n,t in enumerate(np.arange(t0,tmax,dt)):
        with model:
            for dc in dynamic_constraints:
                model.reactions.get_by_id(dc[0]).lower_bound = _michaelis_menten(y[n][compounds.index(dc[0])],dc[1],dc[2])
            try:
                lex_constraints = cobra.util.add_lexicographic_constraints(model, objectives, objectives_direction)
            except:
                stat = 'infeasible'
            solution = model.optimize()
            if model.solver.status == 'infeasible' or stat == 'infeasible':
                break
            tmp_y = []
            tmp_f = []
            for nr, c in enumerate(compounds):
                tmp_y.append(y[n][nr]+solution.fluxes[c]*dt*y[n][0])
                tmp_f.append(solution.fluxes[c])
            y[n+1]=tmp_y
            f[n+1]=tmp_f
    return y,f

def _investigate_reaction(model,reaction_id,silent_metabolites=None):
    '''
    Returns metabolites which are part of the reaction.
    Exempt for metabolite ids listed in silent_metabolites
    (default: [}).

    Parameters
    ----------


    Returns
    -------

    '''
    if silent_metabolites == None:
        silent_metabolites = []
    metabolite_list = []
    for m in model.reactions.get_by_id(reaction_id).metabolites:
        if m.id not in silent_metabolites:
            metabolite_list.append(m.id)

    return metabolite_list

def _investigate_metabolite(model,metabolite_id,silent_reactions=None):
    '''
    Returns reactions in which this metabolite is present.
    Exempt for reaction ids listed in silent_reactions
    (default: []).

    Parameters
    ----------

    Returns
    -------

    '''
    if silent_reactions == None:
        silent_reactions = []
    reaction_list = []
    for r in model.metabolites.get_by_id(metabolite_id).reactions:
        if r.id not in silent_reactions:
            reaction_list.append(r.id)

    return reaction_list

def _mimic_excel():
    'Just creates some letters, Excel style.'
    # https://stackoverflow.com/questions/63875471/enumerate-with-letters-instead-of-numbers
    for i in range(0, 26):
        yield chr(i + 65)

    i, j = [0, 0]

    for j in range(0, 26):
        for i in range(0, 26):
            yield "{}{}".format(chr(j + 65), chr(i + 65))

def investigate_network_solution(model, solution, start_reaction, depth, silent_reactions = None, silent_metabolites = None):
    '''
    See the enviroment of the Metabolic Network around a reaction of interest.

    Parameters
    ----------
    model : cobra.core.model.Model
        Cobrapy model.
    solution : cobra.core.Solution
        A CobraPy model solution.
    start_reaction : Str
        Reaction_id of reaction of interest
    depth : Int
        Reaction depth to investigate
    silent_reactions : list ``default=[]``
        Reactions you are not interested in for whatever reason
    silent_metabolites : list, ``default=['h_c','atp_c','h2o_c','pi_c','h_p','adp_c','o2_p','h2o_c','h2o_p']``
        Metabolites you are not interested in for whatever reason (usually because they are present in almost all reactions).


    Returns
    -------
    None : None
        Prints reactions and metabolites in the network with non-zero flux in the solution object.
    '''

    if silent_reactions == None:
        silent_reactions = []
    if silent_metabolites == None:
        silent_metabolites = ['h_c','atp_c','h2o_c','pi_c','h_p','adp_c','o2_p','h2o_c','h2o_p']

    reaction_list    = [start_reaction]
    metabolite_list  = []
    for i in range(1,depth+1):
        new_reaction_list = []
        for n,r in zip(_mimic_excel(),reaction_list):
            if r in silent_reactions:
                pass
            elif solution[r] == 0:
                pass
            else:
                print('{:1}.{:3} {:17} ({:.5f})'.format(i,n,r,solution[r]))
                metabolite_list = _investigate_reaction(model,r,silent_metabolites)
                silent_reactions.append(r)
                for m in metabolite_list:
                    print('      > {:15} ({:5})'.format(m,model.reactions.get_by_id(r).metabolites[model.metabolites.get_by_id(m)]))
                    new_reaction_list += _investigate_metabolite(model,m,silent_reactions)
        reaction_list = set(new_reaction_list)
    return

def investigate_network(model,start_reaction,depth,silent_reactions = None,silent_metabolites = None):
    '''
    See the enviroment of the Metabolic Network around a reaction of interest.

    Parameters
    ----------
    model : cobra.core.model.Model
        Cobrapy model.
    start_reaction : Str
        Reaction_id of reaction of interest
    depth : Int
        Reaction depth to investigate
    silent_reactions : list ``default=[]``
        Reactions you are not interested in for whatever reason
    silent_metabolites : list, ``default=['h_c','atp_c','h2o_c','pi_c','h_p','adp_c','o2_p','h2o_c','h2o_p']``
        Metabolites you are not interested in for whatever reason (usually because they are present in almost all reactions).

    Returns
    -------
    None : None
        Prints reactions and metabolites in the Network.    '''

    if silent_reactions == None:
        silent_reactions = []
    if silent_metabolites == None:
        silent_metabolites = ['h_c','atp_c','h2o_c','pi_c','h_p','adp_c','o2_p','h2o_c','h2o_p']

    reaction_list    = [start_reaction]
    metabolite_list  = []
    for i in range(1,depth+1):
        new_reaction_list = []
        for n,r in zip(_mimic_excel(),reaction_list):
            if r in silent_reactions:
                pass
            else:
                print('{:1}.{:3} {}'.format(i,n,r))
                metabolite_list = _investigate_reaction(model,r,silent_metabolites)
                silent_reactions.append(r)
                for m in metabolite_list:
                    print('      > {:15} ({:5})'.format(m,model.reactions.get_by_id(r).metabolites[model.metabolites.get_by_id(m)]))
                    new_reaction_list += _investigate_metabolite(model,m,silent_reactions)
        reaction_list = set(new_reaction_list)
    return

'''
The functions can also be called with ``snek.<function>``.
'''

import logging
import pandas as pd
import cobra

def set_objective(model,reaction,direction='max'):
    '''
    Updates the objective function of a model. More importantly, also checks if
    the objective reaction has unusual bounds.
    I.e., if other bounds than (lower bound, upper bound)

        * ``(-1000, 1000)`` for maximization and minimization,
        * ``(    0, 1000)`` for maximization, or
        * ``(-1000,    0)`` for minimization

    are present, a warning is printed.

    Parameters
    ----------
        model : cobra.core.model.Model
            CobraPy Model.
        reaction : Str
            Objective reaction ID.
        direction : Str, default='max'
            Objective direction, either ``'min'`` or ``'max'``.

    Returns
    -------
        None : None
            The given model object is updated.
    '''

    if model.reactions.get_by_id(reaction).upper_bound == 1000 and model.reactions.get_by_id(reaction).lower_bound == -1000:
        pass
    elif direction == 'min' and (model.reactions.get_by_id(reaction).lower_bound != -1000 or model.reactions.get_by_id(reaction).upper_bound != 0):
        logging.warning('The objective reaction has a lower bound at {:.2f} and a upper bound at {:.2f}. This could lead to problems during optimization.'.format(model.reactions.get_by_id(reaction).lower_bound,model.reactions.get_by_id(reaction).upper_bound))
    elif direction == 'max' and (model.reactions.get_by_id(reaction).lower_bound != 0 or model.reactions.get_by_id(reaction).upper_bound != 1000):
        logging.warning('The objective reaction has a lower bound at {:.2f} and a upper bound at {:.2f}. This could lead to problems during optimization.'.format(model.reactions.get_by_id(reaction).lower_bound,model.reactions.get_by_id(reaction).upper_bound))

    model.objective = reaction
    model.objective_direction = direction

def set_bounds(model,reaction,lb=None,ub=None):
    '''
    Updates the bounds of a specific reaction. If None, the bounds are not changed.

    Parameters
    ----------
        model : cobra.core.model.Model
            CobraPy Model.
        reaction : Str
            Reaction ID.
        lb : Float, default=None
            Lower bound. If None, the bound is not updated.
        ub : Float, default=None
            Upper bound. If None, the bound is not updated.

    Returns
    -------
        None : None
            The given model object is updated.
    '''

    if lb is None:
        lb = model.reactions.get_by_id(reaction).lower_bound
    if ub is None:
        ub = model.reactions.get_by_id(reaction).upper_bound

    model.reactions.get_by_id(reaction).bounds = float(lb),float(ub)

def sensitive_optimize(model,pFBA=False):
    '''
    This function does some additional checks before model optimization.

    #. In contrast to the original implementation where fluxes can still be extracted from
       infeasible solutions (e.g. ``solution[reaction_id] = <float>``, even if
       ``solution.status == 'infeasible'``), this function raises an
       Error if the solver is infeasible.

    #. Additionally, this function checks if the objective reaction has unusual bounds.
       I.e., if other bounds than (lower bound, upper bound)

        * ``(-1000, 1000)`` for maximization and minimization,
        * ``(    0, 1000)`` for maximization, or
        * ``(-1000,    0)`` for minimization

       are present, a warning is printed.

    #. Finally, this function warns about inconsistent results that can occur when
       doing pFBA with the GLPK solver.

    Parameters
    ----------
        model : cobra.core.model.Model
            CobraPy Model.
        pFBA : Bool, default=False
            Wether or not to do parsimonious FBA.

    Returns
    -------
        solution : cobra.core.solution.Solution
            Optimized solution if ``solution.status != 'infeasible'``.

    Raises
    ------
        Error
            If the solver status is infeasible.
    '''

    objective_reaction_id = get_objective(model)
    objective_direction = model.objective_direction
    set_objective(model,objective_reaction_id,objective_direction)

    if pFBA:
        if get_solver(model) == 'glpk':
            logging.warning('You are performing pFBA with the GLPK Solver. This can lead to inconsistent results.')
        solution = cobra.flux_analysis.parsimonious.pfba(model)
    else:
        solution = model.optimize()
    if solution.status == 'infeasible':
        raise logging.error(f'Solution status is {solution.status}.')
    else:
        return solution

def find_biomass_reaction(model):
    '''
    Searches 'Biomass' string in reaction names or reaction IDs.
    Prints reaction IDs of found reactions.
    Not sensitive to capitalization.

    Parameters
    ----------
    model : cobra.core.model.Model
        CobraPy Model to be searched.

    Returns
    -------
    reactions : pandas.DataFrame
        Data frame of reaction names and reaction IDs that contain 'Biomass'.
    '''

    return find_reaction(model,'biomass')

def find_reaction(model,name):
    '''
    Searches ``<name>`` string in reaction names or reaction IDs.
    Not sensitive to capitalization.

    Parameters
    ----------
    model : cobra.core.model.Model
        CobraPy Model to be searched.
    name : Str
        Part of reaction name or ID.

    Returns
    -------
    reactions : pandas.DataFrame
        Data frame of reaction names and reaction IDs that contain the searched phrase.
    '''

    # Credits to Martin Völkl for the idea

    reactions = pd.DataFrame(columns=['id','name'])
    for reaction in model.reactions:
        if name.upper() in reaction.name.upper() or name.upper() in reaction.id.upper():
            reactions.loc[reaction.id] = [reaction.id,reaction.name]
    reactions.reset_index(inplace=True,drop=True)

    return reactions

def find_metabolite(model,name):
    '''
    Searches ``<name>`` string in metabolite names or metabolite IDs.
    Not sensitive to capitalization.

    Parameters
    ----------
    model : cobra.core.model.Model
        CobraPy Model to be searched.
    name : Str
        Part of metabolite name or ID.

    Returns
    -------
    metabolites : pandas.DataFrame
        Data frame of metabolite names and metabolite IDs that contain the searched phrase.
    '''

    # Credits to Martin Völkl for the idea

    metabolites = pd.DataFrame(columns=['id','name'])
    for metabolite in model.metabolites:
        if name.upper() in metabolite.name.upper() or name.upper() in metabolite.id.upper():
            metabolites.loc[metabolite.id] = [metabolite.id,metabolite.name]
    metabolites.reset_index(inplace=True,drop=True)

    return metabolites

def get_objective(model):
    '''
    Returns the reaction ID of a single reaction objective of a CobraPy model.

    Parameters
    ----------
    model : cobra.core.model.Model
        CobraPy Model.

    Returns
    -------
    reaction_id : Str
        Reaction id of the objective reaction.
    '''

    reaction_id = str(model.objective.expression).split('*')[1].split(' ')[0]

    return reaction_id

def get_solver(model):
    '''
    Returns the solver name of the model.

    Parameters
    ----------
        model : cobra.core.model.Model
            CobraPy model.

    Returns
    -------
        solver_name : Str
            Solver name as string.

    Raises
    ------
        Error
            If the solver is not in ``['glpk', 'glpk_exact', 'cplex', 'scipy', 'gurobi']``.
    '''

    from optlang import available_solvers
    if available_solvers['GLPK']:
        from optlang.glpk_interface import Model as glpk_Model
        from optlang.glpk_exact_interface import Model as glpk_exact_Model
    else:
        glpk_Model, glpk_exact_Model = None, None
    if available_solvers['CPLEX']:
        from optlang.cplex_interface import Model as cplex_Model
    else:
        cplex_Model = None
    if available_solvers['SCIPY']:
        from optlang.scipy_interface import Model as scipy_Model
    else:
        scipy_Model = None
    if available_solvers['GUROBI']:
        from optlang.gurobi_interface import Model as gurobi_Model
    else:
        gurobi_Model = None

    if isinstance(model.solver,glpk_exact_Model):
        solver_name = 'glpk_exact'
    elif isinstance(model.solver,glpk_Model):
        solver_name = 'glpk'
    elif isinstance(model.solver,cplex_Model):
        solver_name = 'cplex'
    elif isinstance(model.solver,scipy_Model):
        solver_name = 'scipy'
    elif isinstance(model.solver,gurobi_Model):
        solver_name = 'gurobi'
    else:
        raise logging.error('Solver cannot be identified.')

    return solver_name

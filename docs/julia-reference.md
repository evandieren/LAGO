# Julia LAGO-BO reference

LAGO is originally written in Julia. For broader reach, we decided to port it
onto Python.

We discuss below the different steps, and assume throughout a minimization
problem.

## Initialization of the GP
- Initial dataset D0.
- Standardize the GP output (if need be).
- Fit the GP hyperparameters on this initial dataset

- Perform one informed first step, minimizing the posterior mean.

## Trust-region state
- Initialize using the frozen config.
- Evaluating the gradient of the best incumbent found so far.
- Initializing the TR on this incumbent.

## Evaluation history
- Keep in mind local or global steps (useful for counting evaluations in general)
- which points and what values were observed (f and gradient or just f).

## Main iteration

- Some sanity checks (more for ensuring good running of the algorithm)
- Hyperparameter tuning of the GP (at some frequency).

### Local proposal
- Solving the TR subproblem within the TR (can be conditionned on top to be in
the box constraint).

- Check if the TR is terminated, i.e. if the steps were too small. If so, we
adjust the radius according to the paper.

### Global proposal
- Optimizing the acquisition function outside of the TR, for now using a soft
penalty constraint.

### Local/global decision
EI(x_g) > gamma * I_t? -> global, else local.

### Global step
- Evaluate `f`, and condition the GP.
- If new incumbent, reinitialize the TR at this new point.
- If not, then simply add the point and nothing else.

### Local step
- Perform a SR1 TR step, i.e. computing the improvement ratio, updating
accordingly the radius, and Hessian approximation
- Perform the filtering step on the GP dataset, i.e. removing points too close
to the incumbent to avoid ill-conditioning within the trust region.


### Evaluation history
Add the evaluated points, and f and gradient evaluations (if applicable) to a
history.

## GP assimilation rules

Global: evaluate `f`
- non-incumbent: GP receives `f`
- new incumbent:
    GP receives `f`
    evaluate gradient for TR (GP does NOT receive gradient)

Local: evaluate `f` and gradient.
-non-incumbent:
    TR uses `f` for ratio and `f`, gradient for SR1
    GP receives nothing
- new incumbent:
    TR receives both
    GP receives `f`

## Stopping rules

Early stopping: we stop if N acquisition function values are under epsilon_T, concurrently with I_t < epsilon_T.

Normal stopping: function evaluation budget exhaustion.

## Evaluation-cost accounting
This depends how we count gradient cost. In a finite-difference fashion, that would be `d` function evaluations, 
in an adjoint fashion, a unit cost. In AD, this is to be discussed.


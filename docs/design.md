                  ┌──────────────────┐

                  │ Objective        │
                  │ f(x), grad f(x)  │
                  └────────┬─────────┘
                           │
                      Evaluation
                           │
                           ▼
                  ┌──────────────────┐
                  │ Evaluation       │
                  │ Archive          │
                  │                  │
                  │ x                │
                  │ value?           │
                  │ gradient?        │
                  │ step type        │
                  │ accepted?        │
                  └──────┬─────┬─────┘
                         │     │
             ┌───────────┘     └─────────────┐
             ▼                               ▼
      Trust-region state              Surrogate data
      uses f + gradients              selection policy
                                             │
                                             ▼
                                             GP
# GP modelling
## Noise level
Do not assume that `train_Yvar=1e-9` means GPyTorch actually conditions with variance `1e-9`,
it actually floors it to `1e-6`, need to use a wrapper to ask for lower values.
## Gradients and Hessians
torch.func.hessian
    = forward-over-reverse
    = may fail inside some GPyTorch kernels

jacrev(jacrev(...))
    = reverse-over-reverse
    = broader operator coverage

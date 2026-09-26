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

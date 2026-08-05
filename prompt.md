I would like you to draft another experiment because of a rebuttal for neurips we are doing. Here is what we are proposing "learning the step-size and momentum schedules of an accelerated (fast) gradient method, which both adds a harder algorithm class (contrasting learned gradient descent with learned accelerated methods) and directly exercises the signed span coefficients discussed in the theory responses.". Plan carefully for an additional experiment. Draft it carefully here with proper baselines so that we can report an exciting table for openreview.
We are not going to report all the numerics but just something we can show on openreview. Perhaps a good example could be a non-lasso style problem like logistic regression.

the rebuttal description is in the repo https://git.overleaf.com/6967e518e88721b9639dadb4 (read it carefully the paper so that we are planning the experiment correctly).

the additional example should follow the same structure of the previous one. you should have smoke tests to make sure everything runs locally here.
Then you should update the cluster code to the latest changes and once everything is double checked and correct, you should launch the jobs.

We do not have much time to run these so any experiment that could run in just a few hours would be absolutely amazing.

ultrathink

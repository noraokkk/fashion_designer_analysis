import random

import wandb

for i in range(0,10):

    # # #define our custom x axis metric
    # wandb.define_metric("test/step")
    # # set all other test/ metrics to use this step
    # wandb.define_metric("test/*", step_metric="test/step")
    idx = 1
    para_str = str(i)
    for j in range(0,5):
        wandb.init(
            project="fashion_gp_project",
        )
        wandb.log({
            'loss': random.randint(1,100),
            'full_loss': random.randint(1,100),
            'hyper/lengthscale': random.randint(1,100),
            'hyper/outputscale': random.randint(1,100),
            'hyper/second_noise': random.randint(1,100),
            'hyper/fc_eighs': random.randint(1,100),
        }, step=idx)
        idx += 1

    wandb.log({'test/acc_'+str(i): random.randint(1,100)})
    wandb.log({'test/auc_'+str(i): random.randint(1,100)})
    wandb.finish()






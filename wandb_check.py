import wandb

api = wandb.Api()

me = api.viewer
print("✅ Connected as:", me.username)

run = api.run("TON_USERNAME/TON_PROJET_WANDB/RUN_ID")

for f in run.files():
    print(f.name)
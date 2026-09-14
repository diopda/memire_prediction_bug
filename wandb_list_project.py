import wandb

api = wandb.Api()
me = api.viewer

entities = [me.username] + list(getattr(me, "teams", []))

print("✅ Connected as:", me.username)
print("\nEntities détectées:")
for e in entities:
    print("-", e)

print("\n=== Liste des projets par entity ===")
for entity in entities:
    try:
        projects = list(api.projects(entity))
        print(f"\nEntity: {entity}  |  #projects={len(projects)}")
        for p in projects:
            print("  -", p.name)
    except Exception as ex:
        print(f"\nEntity: {entity}  |  ERREUR:", ex)

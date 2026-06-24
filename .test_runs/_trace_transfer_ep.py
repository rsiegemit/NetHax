import numpy as np, jax, torch, importlib.util, os
spec=importlib.util.spec_from_file_location("ppo","/Users/rsiegelmann/Downloads/Projects/nethax/.test_runs/ppo_minihack.py")
ppo=importlib.util.module_from_spec(spec); spec.loader.exec_module(ppo)
from Nethax.minihax.minihax_env import MinihaxEnv
from Nethax.nethax.obs.nle_obs import build_nle_observation
d=torch.load(".test_runs/ppo_MiniHackRoom5x5Monster.pt",map_location="cpu")
ag=ppo.Agent(d["n_act"]); ag.load_state_dict(d["state_dict"]); ag.eval()
ORDS=[107,108,106,104,117,110,98,121]
env=MinihaxEnv("MiniHack-Room-Monster-5x5-v0")
s,info=env.reset(jax.random.key(1001)); fm=info["fired_mask"]; par=info["pits_at_reset"]
for t in range(15):
    ob=build_nle_observation(s); bl=np.asarray(ob["blstats"])
    with torch.no_grad():
        i=int(torch.argmax(ag(torch.tensor(ppo.crop_glyphs(np.asarray(ob["glyphs"]),bl)[None],dtype=torch.long))[0],-1))
    s,r,done,info=env.step(s,ORDS[i],jax.random.key(7000+t),fired_mask=fm,step_count=t,pits_at_reset=par); fm=info["fired_mask"]
    print(f"t{t} act={i}(ord{ORDS[i]}) hp={int(bl[10])} pos=({int(bl[0])},{int(bl[1])}) r={float(r):+.2f} done={bool(done)}")
    if bool(done): print("EPISODE END"); break

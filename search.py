import random, os
def pick(data):
    return random.choice(data)

search_grid = {
    "DISCOUNT" : [0.982, .93],
    "NORMALIZE" : [True, False],
    "STEPS_PER_EPOCH" : [2000],
    "MINI_BATCH_SIZE" : [32],
    "PPO_EPOCHS_PER_UPDATE" : [20, 40],
    "PPO_DELAY_LEARN" : [1000],

    "RPOLYAK":[False, True],
    "KL_MIN":[.1, .001],
    "HINDSIGHT_ACTION":[.3],
    "TD3_GAE":[False],
    "CRITIC_EMPHATIZE_RECENT":[True],
    "PPO_GAE_N":[1, 3],
    "TD3_GAE_N":[1],
    "PPO_HER":[True, False, True],

    "D2RL":[True],
    "PPO_NORM_IN":[True],
    "PPO_TRAIN_ACTOR":[False],
    "PPO_TRAIN_CRITIC":[True, False],
    "GOAL1_SIZE":[10, 20, 40],

    "HRL":[True],
    "ACHIEVED_PUSH_N":[2],

    "ADVANTAGE":[True, False],
    "ADV_NORM":[False],
    "REW_DELTA":[1.],
    "REW_SCALE":[1.],
    "KL_DELTA":[1.],
    "KL_SCALE":[1.],

    "LOOKAHEAD_1":[.5],
    "LOOKAHEAD_K":[.5],

    "SEPERATE_CRITICS":[False],
    "PPO_HER_RATIO":[.33, .55, .22],
    "BLIND":[True, False, True],

    "TD3_PPO_GRADS":[True],#, False, True],#
    "TD3_PPO_CLIP":[True],#, False, True],#
}
lparams = [
    "DISCOUNT",
    "NORMALIZE",
    "STEPS_PER_EPOCH",
    "MINI_BATCH_SIZE",
    "PPO_EPOCHS_PER_UPDATE",
    "PPO_DELAY_LEARN",

    "RPOLYAK",
    "KL_MIN",
    "HINDSIGHT_ACTION",
    "TD3_GAE",
    "CRITIC_EMPHATIZE_RECENT",
    "PPO_GAE_N",
    "TD3_GAE_N",
    "PPO_HER",

    "D2RL",
    "PPO_NORM_IN",
    "PPO_TRAIN_ACTOR",
    "PPO_TRAIN_CRITIC",
    "GOAL1_SIZE",

    "HRL",
    "ACHIEVED_PUSH_N",

    "ADVANTAGE",
    "ADV_NORM",

    "REW_DELTA",
    "REW_SCALE",
    "KL_DELTA",
    "KL_SCALE",
    "LOOKAHEAD_1",
    "LOOKAHEAD_K",
    "SEPERATE_CRITICS",

    "PPO_HER_RATIO",
    "BLIND",
    "TD3_PPO_GRADS",
    "TD3_PPO_CLIP",
]


config = """ENV = "mujoco-pusher"
SEED = 0
MAX_TIMESTEPS = 1e6
# TD3
EXPL_NOISE = 0.1
BATCH_SIZE = 256
DISCOUNT = {}
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
# TD3 + BC
ALPHA = 2.5
NORMALIZE = {}
# OPEN AI TD3 BASELINE TRAINING
EPOCHS = 60
REPLAY_SIZE = 1e7
START_STEPS = 10000
UPDATE_AFTER = 1000
UPDATE_EVERY = 50*2
UPDATE_COUNT = 40
EVAL_FREQ = 50 * UPDATE_EVERY
# HER
HER_PER_EP = 10
HER_RATIO = .75
# PPO
STEPS_PER_EPOCH = {}
MINI_BATCH_SIZE = {}
PPO_EPOCHS_PER_UPDATE = {}
PPO_DELAY_LEARN = {}

RPOLYAK={}
KL_MIN = {}
HINDSIGHT_ACTION={}
TD3_GAE = {}
CRITIC_EMPHATIZE_RECENT = {}
PPO_GAE_N = {}
TD3_GAE_N = {}
PPO_HER = {}
D2RL = {}
PPO_NORM_IN = {}
PPO_TRAIN_ACTOR = {}
PPO_TRAIN_CRITIC = {}


#DLPPOH
TIMEFEAT = False#True#
LEAK2LL = True#False#

# AUXILARY
CLIP_Q = False#True
PIL2_GV = True

PANDA = "panda" in ENV
ERGOJR = "ergojr" in ENV
MUJOCO = not PANDA and not ERGOJR
assert MUJOCO + PANDA + ERGOJR == 1

BACKLASH = False
PUSHER = "usher" in ENV

GOAL_SIZE = 3

if ERGOJR: # no gripper, velo per joint ( #of joints == action_size )
    ACTION_SIZE = 3 + (not PUSHER) * 1#3
    LL_STATE_SIZE = GOAL_SIZE * 2 + ACTION_SIZE * 2 + TIMEFEAT
    STATE_SIZE = GOAL_SIZE + LL_STATE_SIZE + 3*GOAL_SIZE*PUSHER
else: # arm pos, arm prev pos, arm velo, gripper pos + velo + velp
    ACTION_SIZE = 3 + MUJOCO
    LL_STATE_SIZE = GOAL_SIZE * 3 + 4 * MUJOCO + TIMEFEAT
    STATE_SIZE = 2*GOAL_SIZE + LL_STATE_SIZE + 6*GOAL_SIZE*PUSHER# velp + gripper, object velp for pusher

GOAL0_SIZE = GOAL_SIZE
GOAL1_SIZE = {}

HRL = {}
ACHIEVED_PUSH_N = {}

ADVANTAGE = {}
ADV_NORM = {}

REW_DELTA = {}
REW_SCALE = {}
KL_DELTA = {}
KL_SCALE = {}

LOOKAHEAD_1 = {}
LOOKAHEAD_K = {}
SEPERATE_CRITICS = {}

PPO_HER_RATIO = {}
BLIND = {}

TD3_PPO_GRADS = {}
TD3_PPO_CLIP = {}
"""

values = []
for param in lparams:
    value = pick(search_grid[param])
    print(param, value)
    values.append(value)

cfg = config.format(*values)

#print(cfg)

with open("config.py", "w") as f:
    f.write(cfg)
import uuid, subprocess
fname = "test_"+str(uuid.uuid4())

print("test start : ", fname)
import time
start = time.time()
with open(fname + ".log", 'w') as out:
    return_code = subprocess.call(["python3", "duh5_search.py"], stdout=out)
print("test taken : %.2f minutes "%((time.time() - start) // 60))

from gymnasium.envs.registration import register

register(
    id="gym_examples/Veh2CrashEnv-v1",
    entry_point="gym_examples.envs:Veh2CrashEnv",
    max_episode_steps=30000
)

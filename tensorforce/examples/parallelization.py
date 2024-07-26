# Copyright 2020 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from threading import Thread

from tensorforce import Environment, Runner


def main():
    local()
    local_vectorized()
    multiprocessing()
    socket()


def local():
    """
    Train agent on experience collected in parallel from 4 local CartPole environments.

    Typical use case:
        time for batched agent.act() ~ time for agent.act() > time for environment.execute()
    """
    agent = 'benchmarks/configs/ppo.json'
    environment = 'benchmarks/configs/cartpole.json'
    runner = Runner(agent=agent, environment=environment, num_parallel=4)
    # Batch act/observe calls to agent, unless environment.is_vectorizable()
    # (otherwise essentially equivalent to single environment)
    runner.run(num_episodes=100, batch_agent_calls=True)
    runner.close()


def local_vectorized():
    """
    Train agent on experience collected in parallel from one vectorized CartPole environment.

    Typical use case:
        time for vectorized environment < time for sequential execution
    """
    agent = 'benchmarks/configs/ppo.json'
    environment = 'custom_cartpole'
    runner = Runner(agent=agent, environment=environment, max_episode_timesteps=500, num_parallel=4)
    runner.run(num_episodes=100)
    runner.close()


def multiprocessing():
    """
    Train agent on experience collected in parallel from 4 CartPole environments running in
    separate processes.

    Typical use case:
        (a) time for batched agent.act() ~ time for agent.act()
                        > time for environment.execute() + remote communication
            --> batch_agent_calls = True
        (b) time for environment.execute() > time for agent.act() + process communication
            --> batch_agent_calls = False
    """
    agent = 'benchmarks/configs/ppo.json'
    environment = 'benchmarks/configs/cartpole.json'
    runner = Runner(agent=agent, environment=environment, num_parallel=4, remote='multiprocessing')
    runner.run(num_episodes=100, batch_agent_calls=True)  # optional: batch_agent_calls=True
    runner.close()


def socket():
    """
    Train agent on experience collected in parallel from 2 CartPole environments running on
    another machine.

    Typical use case: same as mode 2, but generally remote communication socket > process

    Simulate remote environment, usually run on another machine via:
        python run.py --environment gym --level CartPole-v1 --remote socket-server --port 65432
    """
    agent = 'benchmarks/configs/ppo.json'
    environment = 'benchmarks/configs/cartpole.json'

    def server(port):
        Environment.create(environment=environment, remote='socket-server', port=port)

    server1 = Thread(target=server, kwargs=dict(port=65432))
    server2 = Thread(target=server, kwargs=dict(port=65433))
    server1.start()
    server2.start()

    runner = Runner(
        agent=agent, num_parallel=2, remote='socket-client', host='127.0.0.1', port=65432
    )
    runner.run(num_episodes=100)  # optional: batch_agent_calls=True
    runner.close()

    server1.join()
    server2.join()


if __name__ == '__main__':
    main()

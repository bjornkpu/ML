from absl import app,flags
import gym,pybullet_envs,time,os
import pybullet as p
import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import suite_pybullet,tf_py_environment
from tf_agents.networks import actor_distribution_network,value_network
from tf_agents.policies import policy_saver
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

os.environ["CUDA_VISIBLE_DEVICES"]="1"  # 1 for CPU

flags.DEFINE_integer('policy_nr',500,'What policy to use')
flags.DEFINE_string('policy_dir','checkpoints', 'dir to policy')
flags.DEFINE_string('sfx', None, 'Suffix to model')
FLAGS = flags.FLAGS

def realtime():
    #Konstruer path og load modell
    dir = os.path.join(FLAGS.policy_dir, 'policy_saved_model_'+FLAGS.sfx)
    policynr = FLAGS.policy_nr
    saved_policy = tf.compat.v2.saved_model.load(os.path.join(
        dir, 'policy_' + ('%d' % policynr).zfill(9)))
    policy_state = saved_policy.get_initial_state(batch_size=1)

    #Last inn environment
    py_env = suite_pybullet.load('AntBulletEnv-v0')
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    #Render
    py_env.render(mode='human')
    for i_episode in range(10):
        time_step = tf_env.reset()
        reward = 0;
        for t in range(1111):
            py_env.render(mode='human')
            action_step = saved_policy.action(time_step, policy_state)
            policy_state = action_step.state
            time_step = tf_env.step(action_step.action)
            reward=time_step.reward.numpy()
            if time_step.step_type == ts.StepType.LAST:
                break
            print("timestep={}\treward={}".format(t,reward))
            time.sleep(1/60)
    tf_env.close()


def main(_):
    realtime()


if __name__ == '__main__':
  flags.mark_flag_as_required('sfx')
  app.run(main)

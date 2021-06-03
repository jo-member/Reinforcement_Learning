import sys
import gym
import pylab
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# 정책 신경망과 가치 신경망 생성
class REINFORCE(nn.Module):
    def __init__(self, action_size):
        super(REINFORCE, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        nn.init.uniform_(self.fc3.weight, -1e-3, 1e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = F.softmax(self.fc3(x))
        return q



# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class REINFORCEAgent:
    def __init__(self, action_size):
        # 행동의 크기 정의
        self.action_size = action_size

        # 정책신경망과 가치신경망 생성
        self.model = REINFORCE(self.action_size)
        self.model.load_state_dict(torch.load('./save_model/' + 'model_state_dict.pt'))

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy = self.model(state)[0]
        with torch.no_grad():
            policy = policy.numpy()
        return np.random.choice(self.action_size, 1, p=policy)[0]


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 액터-크리틱(A2C) 에이전트 생성
    agent = REINFORCEAgent(action_size)

    num_episode = 10
    for e in range(num_episode):
        done = False
        score = 0
        state = env.reset()
        state = torch.FloatTensor(np.reshape(state, [1, state_size]))

        while not done:
            env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = torch.FloatTensor(np.reshape(next_state, [1, state_size]))

            score += reward
            state = next_state

            if done:
                print("episode: {:3d} | score: {:3d}".format(e, int(score)))
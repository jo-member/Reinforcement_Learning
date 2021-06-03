
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 상태가 입력, 큐함수가 출력인 인공신경망 생성
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4,24)
        self.fc2 = nn.Linear(24,24)
        self.fc3 = nn.Linear(24,action_size)
        nn.init.uniform_(self.fc3.weight, -1e-3, 1e-3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

# 카트폴 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size)
        self.model.load_state_dict(torch.load('./save_model/' + 'model_state_dict.pt'))

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        q_value = self.model(state)
        return torch.argmax(q_value,dim =1).item()


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # DQN 에이전트 생성
    agent = DQNAgent(state_size, action_size)

    num_episode = 10
    for e in range(num_episode):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = torch.FloatTensor(np.reshape(state, [1, state_size]))

        while not done:
            env.render()

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)
            next_state = torch.FloatTensor(np.reshape(next_state, [1, state_size]))

            score += reward
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:.3f} ".format(e, score))
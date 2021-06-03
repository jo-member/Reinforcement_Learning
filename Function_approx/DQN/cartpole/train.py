import os
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
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
        self.render = True

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = optim.Adam(self.model.parameters(),lr = self.learning_rate)
        self.loss = nn.MSELoss()

        # 타깃 모델 초기화
        self.update_target_model()

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        state = torch.FloatTensor(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                q_value = self.model(state)
            return torch.argmax(q_value,dim =1).item()

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor([np.array(sample[0][0]) for sample in mini_batch])
        actions = [sample[1] for sample in mini_batch]
        rewards = torch.tensor([sample[2] for sample in mini_batch])
        next_states = torch.FloatTensor([np.array(sample[3][0]) for sample in mini_batch])
        dones = torch.IntTensor([sample[4] for sample in mini_batch])
        # 현재 상태에 대한 모델의 큐함수
        output = self.model(states)
        #one_hot_encoding을 해주는 과정
        one_hot_action = torch.zeros(64,2)
        for i,d in enumerate(actions):
            one_hot_action[i][d] =1
        predicts = torch.sum(one_hot_action * output,dim = 1)
        # 다음 상태에 대한 타깃 모델의 큐함수
        with torch.no_grad():
            target_predicts = self.target_model(next_states)
        # 벨만 최적 방정식을 이용한 업데이트 타깃
        max_q = torch.max(target_predicts,dim = 1)[0]
        targets = rewards + (1 - dones) * self.discount_factor * max_q
        self.optimizer.zero_grad()
        loss = self.loss(predicts,targets)
        loss.backward()
        # 오류함수를 줄이는 방향으로 모델 업데이트
        self.optimizer.step()


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # DQN 에이전트 생성
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0

    num_episode = 500
    for e in range(num_episode):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)
            next_state =np.reshape(next_state, [1, state_size])
            # 타임스텝마다 보상 0.5, 에피소드가 중간에 끝나면 -1 보상
            score += reward
            reward = 0.1 if not done or score == 500 else -1

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, action, reward, next_state, done)
            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()
                # 에피소드마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | memory length: {:4d} | epsilon: {:.4f}".format(
                      e, score_avg, len(agent.memory), agent.epsilon))

                # 에피소드마다 학습 결과 그래프로 저장
                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph.png")


                # 이동 평균이 400 이상일 때 종료
                if score_avg > 400:
                    torch.save(agent.model.state_dict(),'./save_model/'+'model_state_dict.pt')
                    sys.exit()

import pylab
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from deepsarsatodqn.environment import Env
import torch.nn.functional as F

# 딥살사 인공신경망
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(15,30)
        self.fc2 = nn.Linear(30,50)
        self.fc3 = nn.Linear(50,action_size)
        nn.init.uniform_(self.fc3.weight, -1e-3, 1e-3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


# 그리드월드 예제에서의 딥살사 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=4000)
        # 딥살사 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.  
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.batch_size = 128
        self.train_start = 2000
        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()
        self.update_target_model()
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            return torch.argmax(q_value,dim =1).item()
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    # <s, a, r, s'>의 샘플로부터 모델 업데이트
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
        # one_hot_encoding을 해주는 과정
        one_hot_action = torch.zeros(self.batch_size, 5)
        for i, d in enumerate(actions):
            one_hot_action[i][d] = 1
        predicts = torch.sum(one_hot_action * output, dim=1)
        with torch.no_grad():
            target_predicts = self.target_model(next_states)
        max_q = torch.max(target_predicts, dim=1)[0]
        targets = rewards + (1 - dones) * self.discount_factor * max_q
        self.optimizer.zero_grad()
        loss = self.loss(predicts, targets)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env(render_speed=0.01)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DQNAgent(state_size, action_size)
    
    scores, episodes = [], []

    EPISODES = 250
    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = torch.FloatTensor(np.reshape(state, [1, state_size]))

        while not done:
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(np.reshape(next_state, [1, state_size]))
            agent.append_sample(state, action, reward, next_state, done)
            # 샘플로 모델 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                agent.update_target_model()
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(
                      e, score, agent.epsilon))

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("score")
                pylab.savefig("./save_graph/graph.png")


        # 100 에피소드마다 모델 저장
        if e % 100 == 0:
            torch.save(agent.model.state_dict(),'./save_model/'+'model_state_dict.pt')
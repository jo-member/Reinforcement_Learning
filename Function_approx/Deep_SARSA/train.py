
import pylab
import random
import numpy as np
from deepsarsa.environment import Env
import torch
import torch.nn as nn
import torch.optim as optim


# 딥살사 인공신경망
class DeepSARSA(nn.Module):
    def __init__(self, action_size):
        super(DeepSARSA, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(15,30),
            nn.ReLU(),
            nn.Linear(30,30),
            nn.ReLU(),
            nn.Linear(30,action_size)
        )
    def forward(self, x):
        q = self.layer(x)
        return q


# 그리드월드 예제에서의 딥살사 에이전트
class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        
        # 딥살사 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.  
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = DeepSARSA(self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(),lr = self.learning_rate)
        self.loss = nn.MSELoss()
    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model(state)
            return torch.argmax(q_values[0])

    # <s, a, r, s', a'>의 샘플로부터 모델 업데이트
    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 학습 파라메터
        one_hot_action = torch.zeros(1,action_size)
        one_hot_action[0][action]=1
        output = self.model(state)
        predict = torch.sum(one_hot_action * output)
        self.optimizer.zero_grad()
        # done = True 일 경우 에피소드가 끝나서 다음 상태가 없음
        next_q = self.model(next_state)[0][next_action]
        target = reward + (1 - done) * self.discount_factor * next_q
        loss = self.loss(predict,target)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env(render_speed=0.01)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)
    
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

            next_action = agent.get_action(next_state)
            # 샘플로 모델 학습
            agent.train_model(state, action, reward, next_state, 
                                next_action, done)
            score += reward
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
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
            torch.save(agent.model.state_dict(),'./save_model_dspt/'+'model_state_dict.pt')
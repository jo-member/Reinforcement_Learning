
import pylab
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ac_gridworld.environment import Env


# 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
class A2C(nn.Module):
    def __init__(self, action_size):
        super(A2C, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(15,24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24,action_size),
            nn.Softmax(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(15,24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24,1)
        )

    def forward(self, x):
        policy = self.layer1(x)
        value = self.layer2(x)
        return policy, value


# 그리드월드 예제에서의 REINFORCE 에이전트
class A2CAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        
        # REINFORCE 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = A2C(self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # 정책신경망으로 행동 선택
    def get_action(self, state):
        policy,_ =self.model(state)
        with torch.no_grad():
            policy = policy[0].numpy()
        return np.random.choice(self.action_size, 1, p=policy)[0]




    # 정책신경망 업데이트
    def train_model(self, state, action, reward, next_state, done):
        policy, value = self.model(state)
        _, next_value = self.model(next_state)
        reward = torch.tensor(reward)
        done = torch.tensor(done, dtype=torch.uint8)
        target = reward + (1 - done) * self.discount_factor * next_value[0]
        one_hot_action = torch.zeros(1, self.action_size)
        one_hot_action[0][action] = 1
        action_prob = torch.sum(one_hot_action * policy, dim=1)
        cross_entropy = - torch.log(action_prob + 1e-5)
        with torch.no_grad():
            advantage = target - value[0]
        actor_loss = torch.mean(cross_entropy * advantage)

        # 가치 신경망 오류 함수 구하기
        with torch.no_grad():
            critic_loss = 0.5 * torch.square(target - value[0])
        critic_loss = torch.mean(critic_loss)

        # 하나의 오류 함수로 만들기
        loss = 0.2 * actor_loss + critic_loss

        # 오류함수를 줄이는 방향으로 모델 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env(render_speed=0.01)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    EPISODES = 200
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
            score += reward
            reward+=-0.1

            agent.train_model(state, action, reward, next_state, done)

            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3d} ".format(
                      e, score,))

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("score")
                pylab.savefig("./save_graph/graph.png")
                

        # 100 에피소드마다 모델 저장
        if e % 100 == 0:
            torch.save(agent.model.state_dict(), './save_model/' + 'model_state_dict.pt')
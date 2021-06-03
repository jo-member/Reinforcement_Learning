
import pylab
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from deepsarsatodqn.environment import Env


# 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
class REINFORCE(nn.Module):
    def __init__(self, action_size):
        super(REINFORCE, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(15,24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24,action_size),
            nn.Softmax(),
        )

    def forward(self, x):
        policy = self.layer(x)
        return policy


# 그리드월드 예제에서의 REINFORCE 에이전트
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        
        # REINFORCE 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = REINFORCE(self.action_size)
        self.states, self.actions, self.rewards = [], [], []
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # 정책신경망으로 행동 선택
    def get_action(self, state):
        policy =self.model(state)[0]

        with torch.no_grad():
            policy = policy.numpy()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # 반환값 계산
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # 한 에피소드 동안의 상태, 행동, 보상을 저장
    def append_sample(self, state, action, reward):
        self.states.append(state[0].numpy())
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)


    # 정책신경망 업데이트
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        discounted_rewards = torch.tensor(discounted_rewards)
        # 크로스 엔트로피 오류함수 계산
        policies = self.model(torch.FloatTensor(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        action_prob = torch.sum(actions * policies, dim=1)
        cross_entropy = - torch.log(action_prob + 1e-5)
        loss = torch.sum(cross_entropy * discounted_rewards)
        entropy = - policies * torch.log(policies)
        # 오류함수를 줄이는 방향으로 모델 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.states, self.actions, self.rewards = [], [], []
        return torch.mean(entropy)


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env(render_speed=0.01)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = REINFORCEAgent(state_size, action_size)

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
            agent.append_sample(state, action, reward)

            state = next_state

            if done:
                # 에피소드마다 정책신경망 업데이트
                entropy = agent.train_model()
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3d}  | entropy: {:.3f} ".format(
                      e, score,entropy))

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("score")
                pylab.savefig("./save_graph/graph.png")
                

        # 100 에피소드마다 모델 저장
        if e % 100 == 0:
            torch.save(agent.model.state_dict(), './save_model/' + 'model_state_dict.pt')
import sys
import gym
import pylab
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
        self.render = False

        # 행동의 크기 정의
        self.action_size = action_size

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        # 정책신경망과 가치신경망 생성
        self.model = REINFORCE(self.action_size)
        self.states, self.actions, self.rewards = [], [], []
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

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
    # 정책신경망의 출력을 받아 확률적으로 행동을 선택


    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
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
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 액터-크리틱(A2C) 에이전트 생성
    agent = REINFORCEAgent(action_size)

    scores, episodes = [], []
    score_avg = 0

    num_episode = 1000
    for e in range(num_episode):
        done = False
        score = 0
        state = env.reset()
        state = torch.FloatTensor(np.reshape(state, [1, state_size]))

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = torch.FloatTensor(np.reshape(next_state, [1, state_size]))

            # 타임스텝마다 보상 0.1, 에피소드가 중간에 끝나면 -1 보상
            score += reward
            reward = 0.1 if not done or score == 500 else -1

            # 매 타임스텝마다 학습
            agent.append_sample(state, action, reward)
            state = next_state

            if done:
                entropy = agent.train_model()
                # 에피소드마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | entropy: {:.3f}".format(
                      e, score_avg, entropy))

                # 에피소드마다 학습 결과 그래프로 저장
                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph.png")

                # 이동 평균이 400 이상일 때 종료
                if score_avg > 400:
                    torch.save(agent.model.state_dict(), './save_model/' + 'model_state_dict.pt')
                    sys.exit()
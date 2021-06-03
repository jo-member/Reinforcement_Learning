import sys
import gym
import pylab
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 정책 신경망과 가치 신경망 생성

class A2C(nn.Module):
    def __init__(self, action_size):
        super(A2C, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(4,24),
            nn.Tanh(),
        )
        self.fc2 = nn.Linear(24,action_size)
        nn.init.uniform_(self.fc2.weight,-1e-3, 1e-3)
        self.fc3 = nn.Sequential(
            nn.Linear(4,24),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(24,24),
            nn.Tanh(),
            nn.Dropout()
        )
        self.fc4 = nn.Linear(24,1)
        nn.init.uniform_(self.fc4.weight, -1e-3, 1e-3)
    def forward(self, x):
        actor_x = self.fc1(x)
        policy = F.softmax(self.fc2(actor_x))

        critic_x = self.fc3(x)
        value = self.fc4(critic_x)
        return policy, value

# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class A2CAgent:
    def __init__(self, action_size):
        self.render = True

        # 행동의 크기 정의
        self.action_size = action_size
        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        # 정책신경망과 가치신경망 생성
        self.model = A2C(self.action_size)
        # 최적화 알고리즘 설정,
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy, _ = self.model(state)
        with torch.no_grad():
            policy = policy[0].numpy()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        policy, value = self.model(state)
        _, next_value = self.model(next_state)
        reward = torch.tensor(reward)
        done = torch.tensor(done, dtype=torch.uint8)
        target = reward + (1 - done) * self.discount_factor * next_value[0]

        # 정책 신경망 오류 함수 구하기
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(action_size)

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
            agent.train_model(state, action, reward, next_state, done)

            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f}".format(
                      e, score_avg))

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
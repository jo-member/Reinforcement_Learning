import numpy as np
import random
from qlearning.environment_largescale import Env
from collections import defaultdict


class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.iteration = 1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s'> 샘플로부터 큐함수 업데이트
    def learn(self, state, action, reward, next_state):
        state, next_state = str(state), str(next_state)
        q_1 = self.q_table[state][action]
        # 벨만 최적 방정식을 사용한 큐함수의 업데이트
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.step_size * (q_2 - q_1)

    # 큐함수에 의거하여 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        new_epsilon = self.epsilon / self.iteration
        if np.random.rand() < new_epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state = str(state)
            q_list = self.q_table[state]
            action = arg_max(q_list)
        self.iteration+=1
        return action


# 큐함수의 값에 따라 최적의 행동을 반환
def arg_max(q_list):
    max_idx_list = np.argwhere(q_list == np.amax(q_list))
    max_idx_list = max_idx_list.flatten().tolist()
    return random.choice(max_idx_list)


if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    for episode in range(1,1000):
        state = env.reset()
        # reward 값들을 담는 list 생성
        reward_table = []
        # return 값들을 담는 list 생성
        return_table = []
        while True:
            # 게임 환경과 상태를 초기화
            env.render()
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            # 행동을 취한 후 다음 상태, 보상 에피소드의 종료여부를 받아옴
            next_state, reward, done = env.step(action)
            #reward_table에 reward들을 차근차근 저장
            reward_table.append(reward)
            # <s,a,r,s'>로 큐함수를 업데이트
            agent.learn(state, action, reward, next_state)
            state = next_state
            
            # 모든 큐함수를 화면에 표시
            env.print_value_all(agent.q_table)

            if done:
                break
        #여기서 부터 리턴값을 출력
        df = agent.discount_factor
        l = len(reward_table)
        for i in range(l):
            return_value = 0
            for j in range(l-i):
                return_value += ((df) ** j) * reward_table[j+i]
            return_table.append(return_value)
        print(episode, "th episode's return vale, number of state:", l, 'final reward = ', reward_table[-1])
import numpy as np
import random
from collections import defaultdict
from sarsa.environment_largescale import Env


class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.iteration = 1 #시간이 지남을 알려주는 변수 생성
        # 0을 초기값으로 가지는 큐함수 테이블 생성
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s', a'>의 샘플로부터 큐함수를 업데이트(여기서 Pass를고친다)
    def learn(self, state, action, reward, next_state, next_action):
        state, next_state = str(state), str(next_state)
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        td = reward + self.discount_factor * next_state_q - current_q
        new_q = current_q + self.step_size * td
        self.q_table[state][action] = new_q

    # 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        new_epsilon = self.epsilon/self.iteration
        if np.random.rand() < new_epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 아니면 큐함수에 따른 행동 반환
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
    env = Env() #객체생성
    agent = SARSAgent(actions=list(range(env.n_actions)))

    for episode in range(1,1000):
        # 게임 환경과 상태를 초기화 환경에서 상태를 받아와
        state = env.reset()
        # 현재 상태에 대한 행동을 선택 그중에서 해애동을 선택
        action = agent.get_action(state)
        # reward 값들을 담는 list 생성
        reward_table = []
        # return 값들을 담는 list 생성
        return_table = []

        while True:
            env.render()

            # 행동을 위한 후 다음상태 보상 에피소드의 종료 여부를 받아옴
            next_state, reward, done = env.step(action)
            # 다음 상태에서의 다음 행동 선택 (이게 a')
            next_action = agent.get_action(next_state)
            # <s,a,r,s',a'>로 큐함수를 업데이트 여기까지가 샘플
            agent.learn(state, action, reward, next_state, next_action)
            # reward_table에 reward들을 차근차근 저장
            reward_table.append(reward)
            state = next_state #다음상태를 가서
            action = next_action#거기서 또 다른 액션

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
        print(episode, "th episode's return vale, number of state:", l,'final reward = ',reward_table[-1])

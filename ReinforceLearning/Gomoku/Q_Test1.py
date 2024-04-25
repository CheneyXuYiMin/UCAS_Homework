import gym
import random
import time
import pandas as pd
import pickle

# 查看所有已注册的环境
# from gym import envs
# print(envs.registry.all()) 

def str2tuple(string): # Input: '(1,1)'
    string2list = list(string)
    return ( int(string2list[1]), int(string2list[4]) ) # Output: (1,1)

class Game():
    def __init__(self, env):
        self.INTERVAL = 0 # 行动间隔
        self.RENDER = False # 是否显示游戏过程
        self.first = 'blue' if random.random() > 0.5 else 'red' # 随机先后手
        self.currentMove = self.first
        self.env = env
        self.agent = Agent()
    
    def switchMove(self): # 切换行动玩家
        move = self.currentMove
        if move == 'blue': self.currentMove = 'red'
        elif move == 'red': self.currentMove = 'blue'
    
    def newGame(self): # 新建游戏
        self.first = 'blue' if random.random() > 0.5 else 'red'
        self.currentMove = self.first
        self.env.reset()
        self.agent.reset()
    
    def run(self): # 玩一局游戏
        self.env.reset() # 在第一次step前要先重置环境，不然会报错
        while True:
            print(f'--currentMove: {self.currentMove}--')
            self.agent.updateQtable(self.env, self.currentMove, False)
            
            if self.currentMove == 'blue':
                self.agent.lastState_blue = self.env.state.copy()
            elif self.currentMove == 'red':
                self.agent.lastState_red = self.agent.overTurn(self.env.state) # 红方视角需将状态翻转
                
            action = self.agent.epsilon_greedy(self.env, self.currentMove)
            if self.currentMove == 'blue':
                self.agent.lastAction_blue = action['pos']
            elif self.currentMove == 'red':
                self.agent.lastAction_red = action['pos']
            
            state, reward, done, info = self.env.step(action)
            if done:
                self.agent.lastReward_blue = reward
                self.agent.lastReward_red = -1 * reward
                self.agent.updateQtable(self.env, self.currentMove, True)
            else:     
                if self.currentMove == 'blue':
                    self.agent.lastReward_blue = reward
                elif self.currentMove == 'red':
                    self.agent.lastReward_red = -1 * reward
            
            if self.RENDER: self.env.render()
            self.switchMove()
            time.sleep(self.INTERVAL)
            if done:
                self.newGame()
                if self.RENDER: self.env.render()
                time.sleep(self.INTERVAL)
                break
                    
class Agent():
    def __init__(self):
        self.Q_table = {}
        self.EPSILON = 0.05
        self.ALPHA = 0.5
        self.GAMMA = 1 # 折扣因子
        self.lastState_blue = None
        self.lastAction_blue = None
        self.lastReward_blue = None
        self.lastState_red = None
        self.lastAction_red = None
        self.lastReward_red = None
    
    def reset(self):
        self.lastState_blue = None
        self.lastAction_blue = None
        self.lastReward_blue = None
        self.lastState_red = None
        self.lastAction_red = None
        self.lastReward_red = None
    
    def getEmptyPos(self, env_): # 返回空位的坐标
        action_space = []
        for i, row in enumerate(env_.state):
            for j, one in enumerate(row):
                if one == 0: action_space.append((i,j)) 
        return action_space
        
    def randomAction(self, env_, mark): # 随机选择空格动作
        actions = self.getEmptyPos(env_)
        action_pos = random.choice(actions)
        action = {'mark':mark, 'pos':action_pos}
        return action
    
    def overTurn(self, state): # 翻转状态
        state_ = state.copy()
        for i, row in enumerate(state_):
            for j, one in enumerate(row):
                if one != 0: state_[i][j] *= -1
        return state_
    
    def addNewState(self, env_, currentMove): # 若当前状态不在Q表中，则新增状态
         state = env_.state if currentMove == 'blue' else self.overTurn(env_.state) # 如果是红方行动则翻转状态
         if str(state) not in self.Q_table:
             self.Q_table[str(state)] = {}
             actions = self.getEmptyPos(env_)
             for action in actions:
                 self.Q_table[str(state)][str(action)] = 0
        
    def epsilon_greedy(self, env_, currentMove): # ε-贪心策略
        state = env_.state if currentMove == 'blue' else self.overTurn(env_.state) # 如果是红方行动则翻转状态
        Q_Sa = self.Q_table[str(state)]
        maxAction, maxValue, otherAction = [], -100, [] 
        for one in Q_Sa:
            if Q_Sa[one] > maxValue:
                maxValue = Q_Sa[one]
        for one in Q_Sa:
            if Q_Sa[one] == maxValue:
                maxAction.append(str2tuple(one))
            else:
                otherAction.append(str2tuple(one))
        
        try:
            action_pos = random.choice(maxAction) if random.random() > self.EPSILON else random.choice(otherAction)
        except: # 处理从空的otherAction中取值的情况
            action_pos = random.choice(maxAction) 
        action = {'mark':currentMove, 'pos':action_pos}
        return action
    
    
    def updateQtable(self, env_, currentMove, done_):
        
        judge = (currentMove == 'blue' and self.lastState_blue is None) or \
                (currentMove == 'red' and self.lastState_red is None)
        if judge: # 边界情况1：若agent无上一状态，说明是游戏中首次动作，那么只需要新增状态就好，无需更新Q值
            self.addNewState(env_, currentMove)
            return
                
        if done_: # 边界情况2：若当前状态S_是终止状态，则无需把S_添加至Q表格中，直接令maxQ_S_a = 0，并同时更新双方Q值
            for one in ['blue', 'red']:
                S = self.lastState_blue  if one == 'blue' else self.lastState_red
                a = self.lastAction_blue if one == 'blue' else self.lastAction_red 
                R = self.lastReward_blue if one == 'blue' else self.lastReward_red
                print('lastState S:\n', S)
                print('lastAction a: ', a)
                print('lastReward R: ', R)
                maxQ_S_a = 0
                self.Q_table[str(S)][str(a)] = (1 - self.ALPHA) * self.Q_table[str(S)][str(a)] \
                                                + self.ALPHA * (R + self.GAMMA * maxQ_S_a)
                print('Q(S,a) = ', self.Q_table[str(S)][str(a)])
            return
          
        # 其他情况下：Q表无当前状态则新增状态，否则直接更新Q值
        self.addNewState(env_, currentMove)
        S_ = env_.state if currentMove == 'blue' else self.overTurn(env_.state)
        S = self.lastState_blue  if currentMove == 'blue' else self.lastState_red
        a = self.lastAction_blue if currentMove == 'blue' else self.lastAction_red 
        R = self.lastReward_blue if currentMove == 'blue' else self.lastReward_red
        Q_S_a = self.Q_table[str(S_)]
        maxQ_S_a = -100 
        for one in Q_S_a:
            if Q_S_a[one] > maxQ_S_a:
                maxQ_S_a = Q_S_a[one]
        print('lastState S:\n', S)
        print('State S_:\n', S_)
        print('lastAction a: ', a)
        print('lastReward R: ', R)
        self.Q_table[str(S)][str(a)] = (1 - self.ALPHA) * self.Q_table[str(S)][str(a)] \
                                        + self.ALPHA * (R + self.GAMMA * maxQ_S_a)
        print('Q(S,a) = ', self.Q_table[str(S)][str(a)])
        print('\n')
                                            
env = gym.make('TicTacToeEnv-v0')
game = Game(env)
for i in range(50000):
    print('episode', i)
    game.run()
Q_table = game.agent.Q_table

# 将字典转换为DataFrame
df = pd.DataFrame(list(Q_table.items()), columns=['State', 'Q-Table Value'])

# 将DataFrame保存为Excel文件
df.to_excel('Q-table.xlsx', index=False)

# 保存字典
with open('Q_table_dict.pkl', 'wb') as f: 
    pickle.dump(Q_table, f)
# moving cartpole!
##  **목표: 좌우로 왔다갔다하면서 막대 세우기**
### **step1. 필수적인 라이브러리 불러오기**
```python
import gymnasium as gym
import numpy as np
import random
from collections import deque 
from keras.layers import Dense
from tensorflow import keras
from keras import Model
import tensorflow as tf
import matplotlib.pyplot as plt
```
### **step2. class**
```python
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(1024, input_dim=4, activation='ReLU')
        self.d2 = Dense(64, activation='ReLU')
        self.d3 = Dense(32, activation='ReLU')
        self.d4 = Dense(2, activation='linear')
        self.optimizer = keras.optimizers.Adam(0.001)

        self.M = []  # 리플레이 버퍼

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x
```
### **step3. cartpole 환경구성**
```python
env = gym.make( 'CartPole-v1')
model = DQN()
```
### **step4. 수치설정 & 리플레이 버퍼**
```python
episode = 1000
step = 2000
memory_length = 5000
minibatch_size = 256
trys = 10
memory = deque(maxlen=memory_length)
```
### **step5. 보상함수**
```python
def reward_function_1(next_state): #state 1x4
    #-1에 위치시키기
   
    cart_pos = state[0][0]
    next_cart_pos = next_state[0][0]
    target_pos = -1

    if -1.1 < next_cart_pos < -0.9:    # 목표 지점에 도달할 경우
        return 5

    elif cart_pos > -0.9:  # 카트가 -1보다 오른쪽에 있는 경우
        if next_cart_pos < cart_pos:  # pos 값은 작아져야 한다!
            return 5/1+abs(next_cart_pos - target_pos)
        else:
            return 0
    else:  # 카트가 -1보다 왼쪽에 있는 경우
        if next_cart_pos > cart_pos:  # pos 값은 커져야 한다!
            return 5/1+abs(next_cart_pos - target_pos)
        else:
            return 0
    if abs(next_state[2]) > env.theta_threshold_radians or abs(next_state[0]) > env.x_threshold:
            r = -10
def reward_function_2(next_state): #state 1x4
    #+1에 위치시키기
    cart_pos = state[0][0]
    next_cart_pos = next_state[0][0]
    target_pos = 1
    if 0.9 < next_cart_pos < 1.1:    # 목표 지점에 도달할 경우
        return 5
    elif cart_pos < 0.9:
        if next_cart_pos > cart_pos:
            return 5/1+abs(next_cart_pos - target_pos)
        else:
            return 0
    else:
        if next_cart_pos < cart_pos:
            return 5/1+abs(next_cart_pos - target_pos)
        else:
            return 0
    if abs(next_state[2]) > env.theta_threshold_radians or abs(next_state[0]) > env.x_threshold:
            r = -10
```
- `env.reset()`:(array([ 0.01660822,  0.0325723 , -0.01963504, -0.04377728], dtype=float32),
 {})

- `env.reset()[0]`:array([-0.03689059,  0.01076892, -0.00720097,  0.00955538], dtype=float32)

- `print(env.theta_threshold_radians)`:0.20943951023931953

- `print(env.x_threshold)`:2.4
### **step6.학습**
```python
for k in range(trys):
    for i in range(episode):
        #시작 포지션을 -1로 설정
        state = env.reset()[0] #env.reset()은 (state,{})를 반환 #[position, velocity, angle, angular velocity]
        if i%2 == 0:
            env.state[0] -= 1
            env.state[1] += 0.5
            state[0] -= 1
            state[1] += 0.5
            cart_goal = 'right'
        else:
            env.state[0] += 1
            env.state[1] -= 0.5
            state[0] += 1
            state[1] -= 0.5
            cart_goal = 'left'
        state = state.reshape(1,4) #state = 1x4
        eps = (5/10)* np.exp(-i/200) #e-greedy 점점 줄어들게 설정
        total_reward = 0
        score = 0

        for t in range(step):
            #e-greedy
            if np.random.rand() < eps:
                action = np.random.randint(0,2)
            else:
                action = np.argmax(model.call(state))
            #다음 상태 및 보상
            next_state, reward, done = env.step(action)[0:3]

            if cart_goal == 'left' and next_state[0] < -0.3:
                score += 1
                cart_goal = 'right'

            if cart_goal == 'right' and next_state[0] > 0.3:
                score += 1
                cart_goal = 'left'

            next_state = next_state.reshape(1,4)
            #reward structure
            if done:
                reward = 0
            else:
                reward = reward_function_1(next_state) + reward_function_2(next_state)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if done or t == step-1:
                print('Episode {}, total_reward : {:.3f}, time : {}, score : {}'.format(i, total_reward, t+1, score))
                break
        
        if i > 50:
            minibatch = random.sample(memory, minibatch_size)
            states = np.array([x[0][0] for x in minibatch])
            actions = np.array([x[1] for x in minibatch])
            rewards = np.array([x[2] for x in minibatch]) 
            next_states = np.array([x[3][0] for x in minibatch])
            dones = np.array([x[4] for x in minibatch])

            targets = rewards + 0.95 * np.max(model.call(next_states).numpy(), axis=1) * (1-dones)
            target_y = model.call(states).numpy()
            target_y[range(minibatch_size), actions] = targets

            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(target_y - model.call(states)))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if (i%100==0 or i%100==1) and i>600:
            env = gym.make('CartPole-v1', render_mode="human")
        else:
            env = gym.make('CartPole-v1')
        
        if score >= 4:
            #가중치 저장
            model.save_weights('moving_cartpole_idea5.h5')


#환경 close
env.close()
```
# 결과

![Animation2](https://github.com/doeunjua/moving_cartpole2/assets/122878319/ff5af317-c511-428c-a961-1f429e883c31)


# 종원 보상 체계로 했을 때 더 잘됨
```python
def reward_function(next_state): #state 1x4
    position = next_state[0][0]
    velocity = next_state[0][1]
    angle = next_state[0][2]
    angular_velocity = next_state[0][3]

    if position <0.5 and velocity>0 and angle>0.05:
        if abs(angular_velocity)<0.5:
            return velocity*angle*200
        elif abs(angular_velocity)<1.5:
            return velocity*angle*100
        else:
            return velocity*angle*50
    elif position >-0.5 and velocity<0 and angle<-0.05:
        if abs(angular_velocity)<0.5:
            return velocity*angle*200
        elif abs(angular_velocity)<1.5:
            return velocity*angle*100
        else:
            return velocity*angle*50
    else:
        return 0.05

#결과
![Animation](https://github.com/doeunjua/moving_cartpole2/assets/122878319/1f4f0801-7011-48ac-928c-a8ad1060b93e)

```
# 결과
![Alt text](Animation.gif)

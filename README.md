# A3C (asynchronous advantage actor-critic)

### paper
REINFORCE, A3C - [Asynchronous Methods for Deep Reinforcement Learning(2016.1.16)](https://arxiv.org/abs/1602.01783)

### Code Environment
[OpenAI-gym](https://www.gymlibrary.dev/)
<br>

### Code
[Worldmodel-A3C](https://github.com/Deepest-Project/WorldModels-A3C)
[Parallel  Distributed  Processing](http://www.aistudy.com/neural/parallel_distributed_processing.htm)
<br>

### Mac setting 
```shell
brew install cmake zlib
pip install 'gymnasium[all]'

pip install autorom # 0.42 버전 설치 (0.55 버전 mac m1 subprocess error)
AutoROM --accept-license (rom license Y 후 ale_py rom으로 파일 이동)
# mv /Users/[user_id]/[anaconda/conda/miniforge]/envs/[env_name]/lib/python3.8/site-pakage/AutoRom/rom /Users/[user_id]/[anaconda/conda/miniforge]/envs/[env_name]/lib/python3.8/site-pakage/ale_py/rom/


```

<br>
[pip install gym error](https://www.pygame.org/wiki/MacCompile)

[딥러닝 분산 학습 관련 연구, Deep learning travels(류성원)](https://lyusungwon.github.io/assets/publications/DistributedDeepLearningTrainingOverview.pdf)

[나만의 GYM 환경 만들기](https://www.youtube.com/watch?v=chVLag1NIAQ)
<br>


## A3C 
         
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B31.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B32.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B33.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B34.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B35.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B36.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B36.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B37.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B38.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B39.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B310.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B311.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B312.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B313.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B314.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B315.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B316.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B317.jpeg" height=80% width=80% >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/ppt/A3C/%E1%84%89%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%83%E1%85%B318.jpeg" height=80% width=80% >


<img src="https://github.com/seohyunjun/RL_A3C/blob/main/data/network_1.gif" width="75%" height="75%" >
<img src="https://github.com/seohyunjun/RL_A3C/blob/main/data/global_reward.png" height=80% width=80% >



--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
require 'gnuplot'
require 'torch'
if not dqn then
    require "initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
--local data_env, actions, agent, opt = setup(opt)
--local game_actions, agent, opt = setup(opt)
local game_actions,agent, opt = setup(opt)
--local game_actions={-1,0,1}
-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end


local s_value={}
  dt = 0.05 
  points=10 
  sin_index=0
  hold_num=0 --dollar
  Account_All=100 --初始本金
  lossRate=0.6  --if lose 40% stop
  Account=Account_All --RMB
  
  price={}
  sindex={}
  shb={}-----plot action

--获取第i个点的sin值
function getSinValue(sin_index, dt)  --RMB/1$
  return math.sin(sin_index*dt)+1
--  x=torch.uniform() +torch.random(1, 5)
--  y=math.pow(-1,torch.random(1,5))
--return math.abs(math.sin(sin_index*dt+0.001)+1+x*y )
  end

--返回初始状态，初始reward,初始terminal
function getState()
   sin_index  = sin_index + 1
   hold_num = hold_num  + 1 -- buy 1$ at point11
   Account  =   Account - getSinValue(11,dt)
  local sinTensor = torch.Tensor(points+2,1):fill(0.01)--状态
   for i  = sin_index , sin_index + points  -1 do 
     sinTensor[ i - sin_index + 1] = getSinValue(  i , dt  )
   end
   sinTensor[11]  = hold_num
   sinTensor[12]  = Account_All
  return  sinTensor,0,false
  end

--根据action返回对应的状态,reward,terminal
function Step(action)
  sin_index = sin_index + 1
  
    shb[sin_index]=action+1 -----plot
  sindex[sin_index]=sin_index
   price[sin_index]=getSinValue(sin_index,dt)
  
  local terminal =  false
  
  local dprice  = getSinValue(sin_index+points  , dt)  - getSinValue(sin_index+points-1 , dt)
  -------------------print___info------------------------
        print (sin_index+points , getSinValue(sin_index+points,dt)  )
        print (sin_index+points-1 , getSinValue(sin_index+points-1,dt)  ) 
        print ("reward=",hold_num,"X",dprice)
  
    
  hold_num  = hold_num  + action --buy/hold/sell 1$ at point 11
  local rw=hold_num  * dprice ---action
  Account  = Account  - action  * getSinValue(  sin_index+points  , dt  )
  
   
  local sinTensor = torch.Tensor(points+2,1):fill(0.01)--状态
   for i=sin_index , sin_index+points-1 do 
     sinTensor[i-sin_index+1]=getSinValue(i,dt)
   end
    sinTensor[11]  = hold_num
    local tmp=Account  + hold_num  * getSinValue( sin_index + points, dt)
    sinTensor[12]  = tmp
    
    
    if tmp <  Account_All *  (1-lossRate)  then
        terminal = true
    end
  return sinTensor, rw, terminal
end

--RandomStep Function
--function gameEnv:_randomStep()
--    return self:_step(self._actions[torch.random(#self._actions)])
--end

function NewState()
  print("here-----------")
  hold_num=0 --dollar
  Account=Account_All --RMB
   local sinTensor,reward,terminal = Step(0)
--   while not terminal do
--        sinTensor,reward,terminal = Step( game_actions[ torch.random(#game_actions) ] )
--    end
  return  sinTensor,reward,terminal
end

function NewRandomState(k)       -------------------what's the mening?
  local sinTensor,reward,terminal = NewState()
   k = k or torch.random(opt.random_starts)
    for i=1,k-1 do
         sinTensor, reward, terminal = Step(0)
        if terminal then
            print(string.format('WARNING: Terminal signal received after %d 0-steps', i))
        end
    end
     sinTensor, reward, terminal =  Step(0)
  return   sinTensor, reward, terminal
  end


local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward

--local screen, reward, terminal = game_env:getState()
local screen, reward, terminal = getState()
 print( "first state: ")
 print(screen )

print("Iteration ..", step)
--print(getSinValue(92,0.05))
local win = nil
--while step < 1000 do
while step < opt.steps do
    step = step + 1
    local action_index = agent:perceive(reward, screen, terminal)
      print ("opt.itera: ",step)
      print("action: ",game_actions[action_index])
     
    -- game over? get next game!
    if not terminal then
      screen, reward, terminal = Step(game_actions[action_index])
      print ("reward: ",reward)
    
      print( "next state: ")
      print(screen ) 
      
--        screen, reward, terminal = game_env:step(game_actions[action_index], true)
--    else
--        if opt.random_starts > 0 then
--            screen, reward, terminal = game_env:nextRandomGame()
--        else
--            screen, reward, terminal = game_env:newGame()
--        end
     else
         if opt.random_starts > 0 then
            screen, reward, terminal = NewRandomState()
         else
            screen, reward, terminal = NewState()
         end
    end

    -- display screen
   --win = image.display({image=screen, win=win})

--调参
--    if step % opt.prog_freq == 0 then
--        assert(step==agent.numSteps, 'trainer step: ' .. step ..
--                ' & agent.numSteps: ' .. agent.numSteps)
--        print("Steps: ", step)
--        agent:report()
--        collectgarbage()
--    end

    if step%1000 == 0 then collectgarbage() end

    if step % opt.eval_freq == 0 and step > learn_start then
     --if step % 125 == 0 and step > 50 then  
        --screen, reward, terminal = game_env:newGame()
        screen, reward, terminal = NewState()
        
        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        local eval_time = sys.clock()
        for estep=1,opt.eval_steps do
            local action_index = agent:perceive(reward, screen, terminal, true, 0.05)
            -- Play game in test mode (episodes don't end when losing a life)
            screen, reward, terminal = Step(game_actions[action_index])

            -- display screen
          --  win = image.display({image=screen, win=win})

            if estep%1000 == 0 then collectgarbage() end

            -- record every reward
            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end

            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                --screen, reward, terminal = game_env:nextRandomGame()
                screen, reward, terminal = NewRandomState()
            end
        end

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time
        agent:compute_validation_statistics()
        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards))
    end

    if step % opt.save_freq == 0 or step == opt.steps then
       --if step % 1000 == 0 or step == opt.steps then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filename = opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
        end
        filename = filename
        torch.save(filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
        if opt.saveNetworkParams then
            local nets = {network=w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, r, s2, term
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
        print('Saved:', filename .. '.t7')
        io.flush()
        collectgarbage()
    end
   -- gnuplot.pngfigure('/home/qxm/plot.png')
--gnuplot.plot({torch.Tensor(sindex), torch.Tensor(price)},{torch.Tensor(sindex), torch.Tensor(shb)})
--print(#sindex)
--print(#price)
--print(#shb)
--gnuplot.plotflush()
end
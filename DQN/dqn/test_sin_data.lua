--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

--gd = require "gd"
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

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-gif_file', '', 'GIF path to write session screens')
cmd:option('-csv_file', '', 'CSV path to write session data')

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
--local game_env, game_actions, agent, opt = setup(opt)
local game_actions,agent, opt = setup(opt)
-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

   dt = 0.05
  points=10
  sin_index=0
  hold_num=0 --dollar
  Account_All=100
  lossRate=0.6   --if lose 40% stop
  Account=Account_All --RMB
  
  price={}
  sindex={}
  shb={}
  trw=0
  own={}
  max=150
  loop=250 --循环次数（打印出的点的数目）
  action_index={}

function getSinValue(sin_index, dt)  --RMB/1$
   --sindex[sin_index]=sin_index
   --price[sin_index]=math.sin(sin_index*dt+0.001)+1
 
--带噪声的正弦数据
--   x=torch.uniform() /10--+torch.random(1, 5)
--   y=math.pow(-1,torch.random(1,5))
--  return math.abs(math.sin(sin_index*dt+0.001)+1+x*y )
  
  --偏移0.001单位的正弦数据
  return math.sin(sin_index*dt+0.001)+1
  --return sin_index*dt+1
  end


function Step(action)
    sin_index = sin_index + 1
  
  shb[sin_index]=action+1 -----plot action
  sindex[sin_index]=sin_index 
  action_index[sin_index]=sin_index+10
   price[sin_index]=getSinValue(sin_index,dt) ---plot price
   
  
  local terminal =  false
  
  local dprice  = getSinValue(sin_index+points  , dt)  - getSinValue(sin_index+points-1 , dt)
  -------------------print___info------------------------
        print (sin_index+points , getSinValue(sin_index+points,dt)  )
        print (sin_index+points-1 , getSinValue(sin_index+points-1,dt)  ) 
        print ("reward=",hold_num,"X",dprice)
        
    
  hold_num  = hold_num  + action --buy/hold/sell 1$ at point 11
  local rw=hold_num  * dprice---action
  trw=trw+rw
  own[sin_index]=trw/max
  
  
  Account  = Account  - action  * getSinValue(  sin_index+points  , dt  )
   
   
  local sinTensor = torch.Tensor(points+2,1):fill(0.01)
   for i=sin_index , sin_index+points-1 do 
     sinTensor[i-sin_index+1]=getSinValue(i,dt)
   end
    sinTensor[11]  = hold_num
    local tmp=Account  + hold_num  * getSinValue( sin_index + points, dt)
    sinTensor[12]  = tmp
    
    print(tmp)
    print(Account_All * (1-lossRate))
    if tmp <  Account_All * (1-lossRate) then
        terminal = true
    end
  return sinTensor, rw, terminal
end

function NewState()
 print("here-----------")
  hold_num=0 --dollar
  Account=Account_All --RMB
   local sinTensor,reward,terminal = Step(1)
--   while not terminal do
--        sinTensor,reward,terminal = Step( game_actions[ torch.random(#game_actions) ] )
--    end
  return  sinTensor,reward,terminal
end




-- file names from command line
local gif_filename = opt.gif_file

-- start a new game
local screen, reward, terminal = NewState()

-- compress screen to JPEG with 100% quality
--local jpg = image.compressJPG(screen:squeeze(), 100)
-- create gd image from JPEG string
--local im = gd.createFromJpegStr(jpg:storage():string())
-- convert truecolor to palette
--im:trueColorToPalette(false, 256)

-- write GIF header, use global palette and infinite looping
--im:gifAnimBegin(gif_filename, true, 0)
-- write first frame
--im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)

-- remember the image and show it first
--local previm = im
--local win = image.display({image=screen})

print("Started playing...")
N_reward=0
p_reward=0
n_reward=0
T_reward=0
-- play one episode (game)
--while not terminal do
   for i=1,loop do
     print(terminal)
     if  terminal then
       break
       end
   -- if action was chosen randomly, Q-value is 0
    agent.bestq = 0
    -- choose the best action
    print()
    print("Loop----------------------",i)
    --print("Looping----------------------")
    local action_index = agent:perceive(reward, screen, terminal, true, 0.05)
    print("next action 1/2/3 for S/H/B",action_index)
    -- play game in test mode (episodes don't end when losing a life)
    --screen, reward, terminal = game_env:step(game_actions[action_index], false)
   screen, reward, terminal = Step(game_actions[action_index])
   
   T_reward=reward+T_reward
   
   if(reward~=0) then 
      N_reward=N_reward+1
    end
   
   if(reward>0) then 
      p_reward=p_reward+1
    else
      n_reward=n_reward+1
    end
   
      
   
   
    -- display screen
   -- image.display({image=screen, win=win})

    -- create gd image from tensor
    --jpg = image.compressJPG(screen:squeeze(), 100)
    --im = gd.createFromJpegStr(jpg:storage():string())
    
    -- use palette from previous (first) image
    --im:trueColorToPalette(false, 256)
   -- im:paletteCopy(previm)

    -- write new GIF frame, no local palette, starting from left-top, 7ms delay
   -- im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)
    -- remember previous screen for optimal compression
    --previm = im

end
 print("the num of positive rewards is ", p_reward)
    print("the num of negative rewards is ",n_reward)
    print("total reward is ",T_reward)
  
    
    
gnuplot.pngfigure('/home/qxm/result/plot.png')
--下面代码依次画出：
--价格走势
--对应所采取的动作
--总回报(own=total_reward/max)，除以系数max是为了结果展示方便
gnuplot.plot({torch.Tensor(sindex), torch.Tensor(price)},{torch.Tensor(action_index), torch.Tensor(shb)} , {torch.Tensor(action_index),torch.Tensor(own)})

print(#sindex)
print(#price)
print(#shb)
print(#own)
gnuplot.plotflush()
-- end GIF animation and close CSV file
--gd.gifAnimEnd(gif_filename)

print("Finished playing, close window to exit!")
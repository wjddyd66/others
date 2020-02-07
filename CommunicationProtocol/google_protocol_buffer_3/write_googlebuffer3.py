#!/usr/bin/env python
# coding: utf-8

# In[5]:


import monster_pb2
import sys


# In[6]:


try:
  raw_input          # Python 2
except NameError:
  raw_input = input  # Python 3


# In[9]:


def PromptForMonster(monster):
    monster.mana = int(raw_input("Enter monster mana: "))
    monster.hp = int(raw_input("Enter monster hp: "))
    monster.name = raw_input("Enter monster name: ")
    monster.friendly = bool(raw_input("Enter monster friendly: "))
    weapons =[]
    
    while True:
        weapon = raw_input("Enter a weapon name,dmage (or leave blank to finish): ")
        if weapon == "":
          break
        name, damage = weapon.split(',')
        imsi = [name, damage]
        weapons.append(imsi)
        monster_weapon = monster.weapons.add()
        monster_weapon.name = name
        monster_weapon.damage = int(damage)
        
    monster_inven = []
    while True:
        inven = raw_input("Enter a inventory (or leave blank to finish): ")
        if inven == "":
          break
        
        monster_inven.append(int(inven))
        
    monster.inventory[:] = monster_inven
            
    type = raw_input("Is this a Red, Green, or Blue? ")
    if type == "Red":
      monster.color = monster_pb2.Monster.Red
    elif type == "Green":
      monster.color = monster_pb2.Monster.Green
    elif type == "Blue":
      monster.color = monster_pb2.Monster.Blue
    else:
      print "Unknown Color type; leaving as default value."
    
    
    pos = raw_input("Enter monster pos(x,y,z): ")
    x,y,z = pos.split(',')
    monster_pos = monster.pos
    monster_pos.x = int(x)
    monster_pos.y = int(y)
    monster_pos.z = int(z)
    
    while True:
        path = raw_input("Enter a path(x,y,z) (or leave blank to finish): ")
        if path == "":
          break
        
        x,y,z = path.split(',')
        monster_path = monster.path.add()
        monster_path.x = int(x)
        monster_path.y = int(y)
        monster_path.z = int(z)
    
    equip = monster.equipped
    for w in weapons:
        equip.weapon.add(name=w[0],damage=int(w[1]))
        


# In[8]:


if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "SAVE_FILE_NAME")
  sys.exit(-1)


# In[ ]:


address_monster = monster_pb2.AddressMonster()


# In[ ]:


try:
  with open(sys.argv[1], "rb") as f:
    address_monster.ParseFromString(f.read())
except IOError:
  print(sys.argv[1] + ": File not found.  Creating a new file.")


# In[ ]:


PromptForMonster(address_monster.monster.add())


# In[ ]:


with open(sys.argv[1], "wb") as f:
  f.write(address_monster.SerializeToString())


#!/usr/bin/env python
# coding: utf-8

# In[3]:


import monster_pb2
import sys


# In[4]:


def ListMonster(address_monster):
  for monster in address_monster.monster:
    print("Monster pos: ",monster.pos.x, monster.pos.y, monster.pos.z)
    print("Monster mana:", monster.mana)
    print("Monster hp:", monster.hp)
    print("Monster name:", monster.name)
    print("Monster friendly:", monster.friendly)
    
    print("Monster inventory: ")
    for monster_inventory in monster.inventory:
        print(monster_inventory)
    
    print("Monster color: ",monster.color)
    
    print("Monster weapon: ")
    for w in monster.weapons:
        print("Name: ",w.name)
        print("Damage: ",w.damage)
    
    print("Monster equipped")
    for e in monster.equipped.weapon:
        print(e)
    
    print("Monster path:")
    for p in monster.path:
        print(p)


# In[5]:


if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "Save_FILE")
  sys.exit(-1)


# In[6]:


address_monster = monster_pb2.AddressMonster()


# In[7]:


with open(sys.argv[1], "rb") as f:
  address_monster.ParseFromString(f.read())

ListMonster(address_monster)


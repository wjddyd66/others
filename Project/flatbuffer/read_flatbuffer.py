#!/usr/bin/env python
# coding: utf-8

# In[7]:


import MyGame.Sample.Monster as example
import MyGame.Sample.Equipment as equip
import MyGame.Sample.Weapon as weapon
import sys


# In[9]:


buf = open(sys.argv[1], 'rb').read()
buf = bytearray(buf)
monster = example.Monster.GetRootAsMonster(buf, 0)

print('Monster Weapon_list')
weapons_length = monster.WeaponsLength()
for i in range(weapons_length):
    print('Weapon Name: ',monster.Weapons(i).Name())
    print('Weapon Damage: ',monster.Weapons(i).Damage())   
print("Monster Name",monster.Name())
print('Monster Inventory')
for i in range(monster.InventoryLength()):
    print(monster.Inventory(i))
print('Monster Pos')
pos = monster.Pos()
print('Pos.x: ',pos.X())
print('Pos.y: ',pos.Y())
print('Pos.z: ',pos.Z())
print('Monster Hp',monster.Hp())
color = monster.Color()
if color == 0:
    print('Monter Color: Red')
elif color ==1:
    print('Monster Color: Green')
else:
    print('Monster Color: Blue')

print('Monster Mana',monster.Mana())


print('Monster Equipped List')
union_type = monster.EquippedType()
if union_type == equip.Equipment().Weapon:
  # `monster.Equipped()` returns a `flatbuffers.Table`, which can be used to
  # initialize a `MyGame.Sample.Weapon.Weapon()`.
    union_weapon = weapon.Weapon()
    union_weapon.Init(monster.Equipped().Bytes, monster.Equipped().Pos)
    weapon_name = union_weapon.Name()     
    weapon_damage = union_weapon.Damage()
    print('Weapon name: ',weapon_name)
    print('Weapon damage: ',weapon_damage)


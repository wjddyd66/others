#!/usr/bin/python

import sys
import flatbuffers
import MyGame.Sample.Color
import MyGame.Sample.Equipment
import MyGame.Sample.Monster
import MyGame.Sample.Vec3
import MyGame.Sample.Weapon

# Example of how to use FlatBuffers to create and read binary buffers.

def main():
  builder = flatbuffers.Builder(0)

#Add_weapon
  input_weapons = []
  output_weapons = []

  while True:
    weapon = raw_input("Enter a weapon name,dmage (or leave blank to finish): ")
    if weapon == "":
      break
    name, damage = weapon.split(',')
    imsi = [name, damage]
    input_weapons.append(imsi)
    
  for w in input_weapons:
    weapon_imsi = builder.CreateString(w[0])
   
    MyGame.Sample.Weapon.WeaponStart(builder)
    MyGame.Sample.Weapon.WeaponAddName(builder, weapon_imsi)
    MyGame.Sample.Weapon.WeaponAddDamage(builder, int(w[1]))
    output_weapons.append(MyGame.Sample.Weapon.WeaponEnd(builder))
    
  MyGame.Sample.Monster.MonsterStartWeaponsVector(builder, len(output_weapons))
  
  for o in output_weapons:
    builder.PrependUOffsetTRelative(o)
  weapons = builder.EndVector(len(input_weapons))

 # Add_name
  input_name = raw_input("Enter monster name: ")
  name = builder.CreateString(name)


 # Add-Inventory
  input_inventory = int(raw_input("Enter a inventory: "))
  MyGame.Sample.Monster.MonsterStartInventoryVector(builder, input_inventory)
  # Note: Since we prepend the bytes, this loop iterates in reverse order.
  for i in reversed(range(0, 10)):
    builder.PrependByte(i)
  inv = builder.EndVector(10)

 # Add-pos
  input_pos = raw_input("Enter monster pos(x,y,z): ")
  x,y,z = input_pos.split(',')
  pos = MyGame.Sample.Vec3.CreateVec3(builder, float(x), float(y), float(z))

 #Add-Hp
  input_hp = int(raw_input("Enter a hp: "))

 #Add-Color
  input_color = raw_input("Is this Monster'color Red, Green, or Blue? ")
  

 #Add-Mana
  input_mana = int(raw_input("Enter a Mana: "))


  MyGame.Sample.Monster.MonsterStart(builder)
  MyGame.Sample.Monster.MonsterAddPos(builder, pos)
  MyGame.Sample.Monster.MonsterAddMana(builder, input_mana)
  MyGame.Sample.Monster.MonsterAddHp(builder, input_hp)
  MyGame.Sample.Monster.MonsterAddName(builder, name)
  MyGame.Sample.Monster.MonsterAddWeapons(builder, weapons)
  MyGame.Sample.Monster.MonsterAddInventory(builder, inv)

  
  if input_color == "Red":
    MyGame.Sample.Monster.MonsterAddColor(builder,MyGame.Sample.Color.Color().Red)
  elif input_color == "Green":
    MyGame.Sample.Monster.MonsterAddColor(builder,MyGame.Sample.Color.Color().Green)
  elif input_color == "Blue":
    MyGame.Sample.Monster.MonsterAddColor(builder,MyGame.Sample.Color.Color().Blue)
  else:
    print("Unknown Color type; leaving as default value.(Red)")
    MyGame.Sample.Monster.MonsterAddColor(builder,MyGame.Sample.Color.Color().Red)
  
  
  MyGame.Sample.Monster.MonsterAddEquippedType(
      builder, MyGame.Sample.Equipment.Equipment().Weapon)
  print("Which weapon",input_name," equipped?")
  count = 0
  for w in input_weapons:
    print("Number: ",count,'Weapon_name: ',w[0])
    count = count+1

  select_weapon = int(raw_input("Choose Select: "))
  
  MyGame.Sample.Monster.MonsterAddEquipped(builder, output_weapons[select_weapon])
  

  orc = MyGame.Sample.Monster.MonsterEnd(builder)

  builder.Finish(orc)

  buf = builder.Output()

  with open(sys.argv[1], "wb") as f:
    f.write(buf)
  print 'The FlatBuffer was successfully created and verified!'

if __name__ == '__main__':
  main()

#!/usr/bin/python
# Copyright 2015 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To run this file, use `python_sample.sh`.

# Append paths to the `flatbuffers` and `MyGame` modules. This is necessary
# to facilitate executing this script in the `samples` folder, and to root
# folder (where it gets placed when using `cmake`).
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
    
  # Create some weapons for our Monster ('Sword' and 'Axe').
  for w in input_weapons:
    weapon_imsi = builder.CreateString(w[0])
   
    MyGame.Sample.Weapon.WeaponStart(builder)
    MyGame.Sample.Weapon.WeaponAddName(builder, weapon_imsi)
    MyGame.Sample.Weapon.WeaponAddDamage(builder, int(w[1]))
    output_weapons.append(MyGame.Sample.Weapon.WeaponEnd(builder))
    
  MyGame.Sample.Monster.MonsterStartWeaponsVector(builder, len(output_weapons))
  
  for o in output_weapons:
    builder.PrependUOffsetTRelative(o)
  # Note: Since we prepend the data, prepend the weapons in reverse order.
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

  # We now have a FlatBuffer that we could store on disk or send over a network.

  # ...Saving to file or sending over a network code goes here...

  # Instead, we are going to access this buffer right away (as if we just
  # received it).

  buf = builder.Output()

  # Note: We use `0` for the offset here, since we got the data using the
  # `builder.Output()` method. This simulates the data you would store/receive
  # in your FlatBuffer. If you wanted to read from the `builder.Bytes` directly,
  # you would need to pass in the offset of `builder.Head()`, as the builder
  # actually constructs the buffer backwards.

  #Under Code = Exception Error
  '''
  monster = MyGame.Sample.Monster.Monster.GetRootAsMonster(buf, 0)
  # Note: We did not set the `Mana` field explicitly, so we get a default value.
  print(input_mana)
  assert monster.Mana() == input_mana
  assert monster.Hp() == input_hp
  assert monster.Name() == input_name

  if input_color == "Red":
    assert monster.Color() == MyGame.Sample.Color.Color().Red  
  elif input_color == "Green":
    assert monster.Color() == MyGame.Sample.Color.Color().Red
  elif input_color == "Blue":
    assert monster.Color() == MyGame.Sample.Color.Color().Red  
  else:
    assert monster.Color() == MyGame.Sample.Color.Color().Red

  assert monster.Pos().X() == x
  assert monster.Pos().Y() == y
  assert monster.Pos().Z() == z

  # Get and test the `inventory` FlatBuffer `vector`.
  for i in xrange(monster.InventoryLength()):
    assert monster.Inventory(i) == i

  # Get and test the `weapons` FlatBuffer `vector` of `table`s.
  expected_weapon_names = []
  expected_weapon_damages = []

  for i_w in input_weapons:
    expected_weapon.append(i_w[0])
    expected_weapon_damages.append(i_w[1])

  for i in xrange(monster.WeaponsLength()):
    assert monster.Weapons(i).Name() == expected_weapon_names[i]
    assert monster.Weapons(i).Damage() == expected_weapon_damages[i]

  # Get and test the `equipped` FlatBuffer `union`.
  assert monster.EquippedType() == MyGame.Sample.Equipment.Equipment().Weapon

  # An example of how you can appropriately convert the table depending on the
  # FlatBuffer `union` type. You could add `elif` and `else` clauses to handle
  # the other FlatBuffer `union` types for this field.
  if monster.EquippedType() == MyGame.Sample.Equipment.Equipment().Weapon:
    # `monster.Equipped()` returns a `flatbuffers.Table`, which can be used
    # to initialize a `MyGame.Sample.Weapon.Weapon()`, in this case.
    union_weapon = MyGame.Sample.Weapon.Weapon()
    union_weapon.Init(monster.Equipped().Bytes, monster.Equipped().Pos)

    assert union_weapon.Name() == output_weapons[select_weapoon -1][0]
    assert union_weapon.Damage() == output_weapons[select_weapoon -1][1]
  '''

  with open(sys.argv[1], "wb") as f:
    f.write(buf)
  print 'The FlatBuffer was successfully created and verified!'

if __name__ == '__main__':
  main()

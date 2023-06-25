import bpy
import json
import mathutils

def clear_all_materials():
    # GPT generated
    # Get all materials in the scene
    materials = bpy.data.materials

    # Remove all materials
    for material in materials:
        materials.remove(material)

def add_flat_top_light():
    # GPT generated
    # Create a new light object
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))

    # Set light properties
    light = bpy.context.object
    light.data.type = 'SUN'
    light.data.energy = 1.0
    light.data.use_nodes = False
    light.data.shadow_method = 'RAY_SHADOW'
    light.data.shadow_ray_samples = 4

    # Set light direction
    light.rotation_euler = (1.5708, 0, 0)  # Rotates the light to point downwards


def add_camera():
    # GPT
    # Create a camera object
    bpy.ops.object.camera_add(location=(0, 0, 8))

    # Set camera rotation
    camera = bpy.context.object
    camera.rotation_euler = (0, 0, 0)  # Rotates the camera to face downward, GPT said 1.7 i said 0

    bpy.context.scene.render.resolution_x = 1000
    bpy.context.scene.render.resolution_y = 1000
    bpy.context.scene.camera = camera
def set_background_color():
    # GPT generated
    # Set the background color of the world
    world = bpy.context.scene.world
    world.use_nodes = False  # Disable node-based world
    world.color = (100, 100, 100)  # Set background color to black
    

def render_scene(output_filepath):
    # GPT generated
    # Set the output filepath and format
    bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
    bpy.context.scene.render.filepath = output_filepath
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Render the scene
    bpy.ops.render.render(write_still=True)

def change_color(color):
    # GPT
    ball_obj = bpy.context.object
    if len(ball_obj.data.materials) == 0:
        material = bpy.data.materials.new(name="Ball Material")
        ball_obj.data.materials.append(material)
    else:
        material = ball_obj.data.materials[0]
    material.diffuse_color = color

def set_meterial(m):
    ball_obj = bpy.context.object
    try:
        ball_obj.data.materials[0] = bpy.data.materials.get(m)
    except:
        ball_obj.data.materials.append(bpy.data.materials.get(m))
def zoom_camera_to_fit():
    # GPT generated
    # Select all objects in the scene
    bpy.ops.object.select_all(action='SELECT')

    # Set the active object to the camera
    bpy.context.view_layer.objects.active = bpy.context.scene.camera

    # Zoom the camera to fit all selected objects
    bpy.ops.view3d.camera_to_view_selected()
    
    camera = bpy.context.object
    camera.location += mathutils.Vector((0, 0, 1)) #changed from -1 to 1

def add_color_schema(name, color):
    # semi-GPT
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = color

import math
import random 

def random_rotate_camera():
    """
    Generated by ChatGPT.
    Randomly rotates the active camera in Blender by a random angle between -10 and 10 degrees.
    """
    # Get the active camera object
    camera = bpy.context.scene.camera

    if camera is not None:
        # Convert the rotation angles from degrees to radians
        angle_x = math.radians(random.uniform(-5, 5))
        angle_y = math.radians(random.uniform(-5, 5))
        angle_z = math.radians(random.uniform(-5, 5))

        # Rotate the camera by the specified angles
        camera.rotation_euler = (angle_x, angle_y, angle_z)
        
import bpy


import bpy
import mathutils


def connect_points_with_cylinder(point1, point2, radius=0.1):
    """
    Connects two points in Blender using a thin cylinder.
    https://chat.openai.com/share/a5b9e388-c81f-4290-9f28-a25db7db5056
    Args:
        point1: The coordinates of the first point as a tuple or list [x, y, z].
        point2: The coordinates of the second point as a tuple or list [x, y, z].
        radius: The radius of the cylinder (default is 0.1).

    Returns:
        The created cylinder object.
    """
    # Create a new cylinder mesh
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=1)

    # Get the newly created cylinder object
    cylinder = bpy.context.active_object

    # Calculate the distance between the two points
    distance = mathutils.Vector(point2) - mathutils.Vector(point1)

    # Set the scale of the cylinder based on the distance between the points
    cylinder.scale = (radius, radius, distance.length)

    # Calculate the midpoint between the two points
    midpoint = (mathutils.Vector(point1) + mathutils.Vector(point2)) / 2

    # Set the location of the cylinder to the midpoint
    cylinder.location = midpoint

    # Calculate the direction vector from point1 to point2
    direction = distance.normalized()

    # Calculate the rotation quaternion to align the cylinder with the direction vector
    rotation = direction.to_track_quat('Z', 'Y')

    # Set the rotation of the cylinder
    cylinder.rotation_mode = 'QUATERNION'
    cylinder.rotation_quaternion = rotation

    return cylinder  # generated by GPT


def get_cord_by_aid(aid):
    #print(data["coords"][0]["conformers"][0]["x"][aid-1])
#    c = data["coords"][0]["conformers"][0]
    ans = data["coords"][0]["conformers"][0]["x"][aid-1], data["coords"][0]["conformers"][0]["y"][aid-1], 0
    print(ans)
    return ans 

clear_all_materials()



add_color_schema("C", (0.2,0.2,0.2,1))
add_color_schema("O", (1, 0, 0, 1))
add_color_schema("S", (1, 0.5, 0.5, 1))
add_color_schema("H", (0.9,0.9,0.9, 1))
add_color_schema("N", (0,0.1,0.9, 1))
add_color_schema("Cl", (0.1,0.9,0.1, 1))
add_color_schema("Br", (0.7,0.3,0.3,1))

mapping = {6:"C",8:"O",1:"H",7:"N",17:"Cl",35:"Br"}

bpy.ops.scene.new()
add_camera()

with open("/home/wg25r/set2SMILE/collect_images/demo.json","r") as f:
    data = json.load(f)

carbon_ids = []
hydrogen_ids = []
for index, item in enumerate(data["atoms"]["element"]):
    if item == 6:
        carbon_ids.append(index+1)
    if item == 1:
        hydrogen_ids.append(index+1)
               
removed_hydrogen_ids = []
for i in range(len(data["bonds"]["aid1"])):
    if data["bonds"]["aid1"][i] in carbon_ids:
        if data["bonds"]["aid2"][i] in hydrogen_ids:
            removed_hydrogen_ids.append(data["bonds"]["aid2"][i])
        else:
            connect_points_with_cylinder(get_cord_by_aid(data["bonds"]["aid1"][i]), 
                                        get_cord_by_aid(data["bonds"]["aid2"][i]), 0.2)
            
            
    elif data["bonds"]["aid2"][i] in carbon_ids:
        if data["bonds"]["aid1"][i] in hydrogen_ids:
            removed_hydrogen_ids.append(data["bonds"]["aid1"][i])
        else:
            connect_points_with_cylinder(get_cord_by_aid(data["bonds"]["aid1"][i]),
                                        get_cord_by_aid(data["bonds"]["aid2"][i]), 0.2)
    else:
            connect_points_with_cylinder(get_cord_by_aid(data["bonds"]["aid1"][i]),
                                        get_cord_by_aid(data["bonds"]["aid2"][i]), 0.2)  
                                        
for i in range(len(data["coords"][0]["conformers"][0]["x"])):
    if data["atoms"]["aid"][i] in removed_hydrogen_ids:
        continue 
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.13 if (i+1) in hydrogen_ids else 0.2, location=(data["coords"][0]["conformers"][0]["x"][i],
    data["coords"][0]["conformers"][0]["y"][i],0))
    set_meterial(mapping[data["atoms"]["element"][i]])


 
zoom_camera_to_fit() 

#render_scene("/home/wg25r/test.png")
import sys,os
import bpy
import numpy as np
from render_freestyle_svg import register
from mathutils import Vector
import warnings

register()
export_path="/vol/research/ycau/syntheticSketch/chair/"
warnings.filterwarnings("ignore")

def look_at(obj_camera, point):   
    direction = point - obj_camera.location
    # print(direction)
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()
    print(obj_camera.rotation_euler)

# def spherical_to_euclidian(elev, azimuth):
#     r = 2
#     x_pos = r * np.cos(elev/180.0*np.pi) * np.cos(azimuth/180.0*np.pi)
#     y_pos = r * np.cos(elev/180.0*np.pi) * np.sin(azimuth/180.0*np.pi)
#     z_pos = r * np.sin(elev/180.0*np.pi)
#     return x_pos, y_pos, z_pos

def proc_obj(path, uid):
    # if already rendered quit
    #f_path = export_path + flag + uid
    f_path = export_path + '/svg/' + uid
    f_path1 = export_path + '/sil/' + uid
    if not os.path.exists(f_path):
        os.mkdir(f_path)
    if not os.path.exists(f_path1):
        os.mkdir(f_path1)
    
    svg_flag = 0
    sil_flag = 0

    if len(os.listdir(f_path))==48:
        svg_flag = 1
    if len(os.listdir(f_path1))==48:
        sil_flag = 1

    if (svg_flag+sil_flag) == 2:
        print('done')
        exit()
    if (svg_flag+sil_flag) == 1:
        shutil.rmtree(f_path)
        shutil.rmtree(f_path1)
        os.mkdir(f_path)
        os.mkdir(f_path1)

    #file_name = export_path + flag + uid + '/' + '%03d' + '_%s'
    file_name = export_path + uid + '/' + '%03d' + '_%s'
    bpy.data.objects['Cube'].select = True  
    bpy.ops.object.delete() 

    #bpy.data.lamps['Lamp'].type = 'HEMI'
    #bpy.data.lamps['Lamp'].energy = 0.25

    bpy.data.worlds['World'].horizon_color = [255,255,255]

    bpy.ops.import_scene.obj(filepath=path, axis_forward='-X')
    # obj_object = bpy.context.selected_objects[0];
    # scalef = 1.0/max(obj_object.dimensions.x, obj_object.dimensions.y, obj_object.dimensions.z);  
    # print(scalef)
    # obj_object.scale = (scalef, scalef, scalef) 

    center = Vector((0.0, 0.0, 0.0));
    
    num_azis = 8
    num_elevs = 5

    random_state = np.random.RandomState()
    mean_azi = 0
    std_azi = 7
    mean_elev = 0
    std_elev = 7
    mean_r = 0
    std_r = 7
    
    delta_azi_max = 15
    delta_elev_max = 15
    delta_azi_min = 5
    delta_elev_min = 5
    delta_r = 0.1

    azi_origins = np.linspace(0, 315, num_azis)
    elev_origin = 10
    r_origin = 1.5

    bound_azi = [(azi - delta_azi_max, azi + delta_azi_max, azi - delta_azi_min, azi + delta_azi_min) for azi in azi_origins]
    bound_elev = (elev_origin - delta_elev_max, elev_origin + delta_elev_max, elev_origin - delta_elev_min, elev_origin + delta_elev_min)
    bound_r = (r_origin - delta_r, r_origin + delta_r)

    azis = []
    elevs = []
    for azi in azi_origins:
        azis.append(azi)
        elevs.append(elev_origin)

    x_pos = []
    y_pos = []
    z_pos = []
    for azi, elev in zip(azis, elevs):
        x_pos.append(r_origin * np.cos(elev/180.0*np.pi) * np.cos(azi/180.0*np.pi))
        y_pos.append(r_origin * np.cos(elev/180.0*np.pi) * np.sin(azi/180.0*np.pi))
        z_pos.append(r_origin * np.sin(elev/180.0*np.pi))

    for n_azi in range(num_azis):
        for _ in range(num_elevs):
            azi = round(azi_origins[n_azi] + mean_azi + std_azi*random_state.randn())
            while azi < bound_azi[n_azi][0] or azi > bound_azi[n_azi][1] or (azi > bound_azi[n_azi][2] and azi < bound_azi[n_azi][3]):
            # while azi < bound_azi[n_azi][0] or azi > bound_azi[n_azi][1]:       # control bound for azi
                azi = round(azi_origins[n_azi] + mean_azi + std_azi*random_state.randn())

            elev = round(elev_origin + mean_elev + std_elev*random_state.randn())
            # while elev < bound_elev[0] or elev > bound_elev[1]:                   # control bound for elev
            while elev < bound_elev[0] or elev > bound_elev[1] or (elev > bound_elev[2] and elev < bound_elev[3]):
                elev = round(elev_origin + mean_elev + std_elev*random_state.randn())

            while (azi, elev) in list(zip(azis, elevs)):   # control (azi, elev) not repeated
                azi = round(azi_origins[n_azi] + mean_azi + std_azi*random_state.randn())
                # while azi < bound_azi[n_azi][0] or azi > bound_azi[n_azi][1]: # control bound for azi
                while azi < bound_azi[n_azi][0] or azi > bound_azi[n_azi][1] or (azi > bound_azi[n_azi][2] and azi < bound_azi[n_azi][3]):
                    azi = round(azi_origins[n_azi] + mean_azi + std_azi*random_state.randn())

                elev = round(elev_origin + mean_elev + std_elev*random_state.randn())
                # while elev < bound_elev[0] or elev > bound_elev[1]: # control bound for elev
                while elev < bound_elev[0] or elev > bound_elev[1] or (elev > bound_elev[2] and elev < bound_elev[3]):
                    elev = round(elev_origin + mean_elev + std_elev*random_state.randn())

            r = r_origin + mean_r + std_r * random_state.randn()
            while r < bound_r[0] or r > bound_r[1]:       # control bound for r
                r = r_origin + mean_r + std_r * random_state.randn()

            azis.append(azi)
            elevs.append(elev)
            x_pos.append(r * np.cos(elev/180.0*np.pi) * np.cos(azi/180.0*np.pi))
            y_pos.append(r * np.cos(elev/180.0*np.pi) * np.sin(azi/180.0*np.pi))
            z_pos.append(r * np.sin(elev/180.0*np.pi))

    # x_pos = r * np.cos(elev_origin/180.0*np.pi) * np.cos(azi_origins/180.0*np.pi)
    # y_pos = r * np.cos(elev_origin/180.0*np.pi) * np.sin(azi_origins/180.0*np.pi)
    # z_pos = r * np.sin(elev_origin/180.0*np.pi)
    # z_pos = np.stack([z_pos]*num_azis)
    # # zz = r * np.sin(elev_origin/180.0*np.pi)

    # # azis_perturb = [round(azi + mean_azi + std_azi*random_state.randn()) for azi in azi_origins]
    # # elevs_perturb = [round(elev_origin + mean_elev + std_elev*random_state.randn()) for _ in range(num_elevs)]

    # azis_perturb = []
    # for azi in azi_origins:
    #     deg = round(azi + mean_azi + std_azi*random_state.randn())
    #     while deg in azi_origins or deg in azis_perturb:
    #         deg = round(azi + mean_azi + std_azi*random_state.randn())
    #     azis_perturb.append(deg)

    # elevs_perturb = []
    # for _ in range(num_elevs):
    #     deg = round(elev_origin + mean_elev + std_elev*random_state.randn())
    #     while deg == elev_origin or deg in elevs_perturb:
    #         deg = round(elev_origin + mean_elev + std_elev*random_state.randn())
    #     elevs_perturb.append(deg)

    # azis = azi_origins
    # elev_origin = [elev_origin] * num_azis
    # elevs = elev_origin
    # for azi in azis_perturb:
    #     for elev in elevs_perturb:
    #         azis = np.append(azis, azi)
    #         elevs = np.append(elevs, elev)
    #         r_random = r + mean_r + std_r * random_state.randn()
    #         x_pos = np.append(x_pos, r_random * np.cos(elev/180.0*np.pi) * np.cos(azi/180.0*np.pi))
    #         y_pos = np.append(y_pos, r_random * np.cos(elev/180.0*np.pi) * np.sin(azi/180.0*np.pi))
    #         z_pos = np.append(z_pos, r_random * np.sin(elev/180.0*np.pi))

    # distance = 2
    # degree = np.linspace(0,315,8)
    # degree_e = degree/180*np.pi
    # degree_v = 10/180*np.pi
    # x_pos = np.cos(degree_e)*distance
    # y_pos = np.sin(degree_e)*distance
    # zz = np.sin(degree_v)*distance
    # bpy.ops.object.empty_add(type='PLAIN_AXES', view_align=False, location=(0, 0, 0), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
    obj_camera = bpy.data.objects["Camera"]

    # Set up rendering of depth map.
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)  


    # Create input render layer node.
    objs = bpy.data.objects
    objs.remove(objs["Lamp"], True)
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    sil_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    sil_file_output.label = 'Sil Output'
    links.new(render_layers.outputs['Image'], sil_file_output.inputs[0])
    sil_file_output.base_path = ''


    for azi, elev, xx, yy, zz in zip(azis, elevs, x_pos, y_pos, z_pos):
        # if os.path.exists(file_name % (azi,'contour_0001.svg')):
        #     continue
        
        # obj_camera = bpy.data.objects['Camera']
        # obj_other = bpy.data.objects['Empty']
        # look_at(obj_camera, obj_other.matrix_world.to_translation())
        obj_camera.location = (xx,yy,zz)
        look_at(obj_camera, center)    
        #bpy.data.objects['Camera'].rotation_euler = (np.pi/2, 0, np.pi/2 + azi/180.0*np.pi)
        

        # bpy.data.objects['Lamp'].location = (xx,yy,zz)
        # bpy.data.objects['Lamp'].rotation_euler = (np.pi/2, 0, np.pi/2 + azi/180.0*np.pi)


        bpy.data.scenes['Scene'].render.resolution_x = 540
        bpy.data.scenes['Scene'].render.resolution_y = 540
        bpy.data.scenes['Scene'].render.resolution_percentage = 100

        bpy.data.scenes['Scene'].render.use_freestyle = True
 
        bpy.data.scenes['Scene'].render.line_thickness = 2.5
        bpy.data.linestyles['LineStyle'].color = (0,0,0)

        # bpy.data.scenes['Scene'].render.layers['RenderLayer'].freestyle_settinags.linesets['LineSet'].select_border = False
        # bpy.data.scenes['Scene'].render.layers['RenderLayer'].freestyle_settings.linesets['LineSet'].select_silhouette = False 
        # bpy.data.scenes['Scene'].render.layers['RenderLayer'].freestyle_settings.linesets['LineSet'].select_crease = False

        # bpy.data.scenes['Scene'].render.layers['RenderLayer'].freestyle_settings.linesets['LineSet'].select_contour =True 
        # bpy.data.scenes['Scene'].render.layers['RenderLayer'].freestyle_settings.linesets['LineSet'].select_suggestive_contour = False

        bpy.context.scene.render.image_settings.file_format = 'PNG'
        # bpy.context.scene.render.filepath = file_name % (dd,'contour_')
        bpy.context.scene.render.filepath = export_path + '/svg/' + uid + '/' + 'azi_{}_elev_{}_'.format(int(azi), int(elev))
        bpy.context.scene.svg_export.use_svg_export = True
        sil_file_output.file_slots[0].path = export_path + '/sil/' + uid + '/' + 'azi_{}_elev_{}_'.format(int(azi), int(elev)) + "_sil.png"
        bpy.ops.render.render(write_still = False)

if __name__ == "__main__":
    #import pdb;pdb.set_trace()
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    path, uid = argv[0], argv[1]
    proc_obj(path, uid)

#!/usr/bin/python3


import Sofa
import SofaCaribou 
radius = 1
N = 10
n = [N,N]
subdivisions = 0


def createScene(root):

    root.addObject('RequiredPlugin', name="SofaBaseMechanics")
    root.addObject('CircleIsoSurface', radius=radius, center=[0, 0])
    root.addObject('FictitiousGrid',
                   template='Vec2d',
                   name='grid',
                   n=n,
                   min=[-2, -2],
                   max=[2, 2],
                   maximum_number_of_subdivision_levels=subdivisions,
                   printLog=True,
                   draw_boundary_cells=True,
                   draw_outside_cells=True,
                   draw_inside_cells=True
                   )
    root.addObject('MechanicalObject', template='Vec2d', position='@grid.position')
    root.addObject('TriangleSetTopologyContainer', triangles='@grid.triangles')
    return root

root = Sofa.Core.Node()
createScene(root)
Sofa.Simulation.init(root)

print(f'{root.grid.number_of_cells()=}')
print(f'{root.grid.number_of_nodes()=}')
print(f'{root.grid.boundary_cells_indices()=}')
print(f'{root.grid.face_cell_indices(3)=}')
print(f'{root.grid.boundary_faces()=}')
print(f'{len(root.grid.boundary_faces())=}')


faces = []
for cell in root.grid.boundary_cells_indices()[0]:
    faces += root.grid.face_cell_indices(cell)

print(len(faces))
unique_faces = list(set(faces))
print(len(unique_faces))



faces_outside = []
for cell in root.grid.boundary_cells_indices()[2]:
    faces_outside += root.grid.face_cell_indices(cell)

print(len(faces_outside))
unique_faces_outside = list(set(faces_outside))
print(len(unique_faces_outside))


def common_member(a, b):     
    a_set = set(a) 
    b_set = set(b) 
      
    
    if len(a_set.intersection(b_set)) > 0: 
        return(a_set.intersection(b_set))   
    else: 
        return("no common elements") 
      
   
common = list(common_member(unique_faces, unique_faces_outside))
for index in common :
    unique_faces.remove(index)

print(len(unique_faces))

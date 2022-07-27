#!/usr/bin/python3

import Sofa
import SofaCaribou 


radius = 4.
N = 20
n = [N,N,N]
subdivisions = 0

def createScene(root):

	root.addObject('RequiredPlugin', name="SofaBaseMechanics")
	root.addObject('RequiredPlugin', name="SofaCaribou")
	root.addObject('LevelSet', mesh_type = 4)
	root.addObject('FictitiousGrid',
				template='Vec3d',
				name='grid',
				n=n,
				min=[-3,-3,-3],
				max=[3,3,3],
				maximum_number_of_subdivision_levels=subdivisions,
				printLog=True,
				draw_boundary_cells=True,
				draw_outside_cells=False,
				draw_inside_cells=True
				)
	root.addObject('MechanicalObject', template='Vec3d', position='@grid.position')
	root.addObject('QuadSetTopologyContainer', quads='@grid.quads')

		
root = Sofa.Core.Node()
createScene(root)
Sofa.Simulation.init(root)

print(f'{root.grid.number_of_cells()=}')
print(f'{root.grid.number_of_nodes()=}')
print(f'{root.grid.boundary_cells_indices()=}')
print(f'{root.grid.face_cell_indices(0)=}')
print(f'{len(root.grid.boundary_faces())=}')
print(f'{root.grid.phi(0)=}')

phi_values = root.grid.phi()
pos, neg, null = [], [], 0 
for val in phi_values :
	if val > 0 :
		pos.append(val)

	elif val < 0 :
		neg.append(val)

	elif val == 0 :
		null +=1
print(len(pos))
print(len(neg))


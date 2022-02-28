from pycuda import driver, compiler, gpuarray, tools, curandom
import pycuda.autoinit, time
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

import h5py
results_dir = 'wmhd_ip_180808_v2'#'wmhd_ne_183245'
shot_data_file = '180808_new.h5'#'183245_data.h5'
forest_file = 'forest_245_15_dan.h5'

TPB = 32
NDIM = 2
NGEN = 25#150
if len(sys.argv)-1 > 0:
	#set up grid and select instance based on argument
	NPTS_array = [100,300,500,700,900]
	NPOP_array = [50, 150, 250, 350, 400]
	NLMI_array = [3, 5, 7]
	NPTSv, NPOPv, NLMIv = np.meshgrid(NPTS_array, NPOP_array, NLMI_array)
	index = int(sys.argv[1])
	NPTS = int(NPTSv.flatten()[index])
	NLMI = int(NLMIv.flatten()[index])
	NPOP = int(NPOPv.flatten()[index])
	print(f"Index={index}, NPTS={NPTS}, NLMI={NLMI}, NPOP={NPOP}")
	print(f"Total jobs = {len(NPTSv.flatten())}")
	with h5py.File(f'./nvprof_logs/SORI_settings_{index}.h5',"w") as f:
		f.attrs['NPTS'] = NPTS
		f.attrs['NLMI'] = NLMI
		f.attrs['NPOP'] = NPOP	
else:
	NPTS = 900#300 #2000,4,400 used in original plots
	NLMI = 3
	NPOP = 400#40
NCON = NLMI * NPOP
NTOURN = 4
NELITE = 10
NFEATURES = 8
disruptivity_threshold = 0.4#0.15


import os
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
# load compiled cubin
strm1 = driver.Stream() # create a GPU stream
mod = driver.module_from_file("sori_kernel.cubin")
mod_dprf = driver.module_from_file("dprf_kernel.cubin")
init_kernel = mod.get_function("init")
generate_test_points_kernel = mod.get_function("generate_test_points")
random_points_kernel = mod.get_function("random_points")
evaluate_constraint_sos_kernel = mod.get_function("evaluate_constraint_sos")
gpu_mmul_ABT_kernel = mod.get_function("gpu_mmul_ABT")
evaluate_constraints_satisfied_kernel = mod.get_function("evaluate_constraints_satisfied")
evaluate_all_constraints_satisfied_kernel = mod.get_function("evaluate_all_constraints_satisfied")
GpuCopy_kernel = mod.get_function("GpuCopy")
genetic_select_kernel = mod.get_function("genetic_select")
genetic_mate_kernel = mod.get_function("genetic_mate")
genetic_mutate_kernel = mod.get_function("genetic_mutate")
carry_over_elite_kernel = mod.get_function("carry_over_elite")
# evaluate_disruptivity_kernel = mod.get_function("evaluate_disruptivity")
eval_forest_kernel = mod_dprf.get_function("eval_forest")

# create GPU arrays
d_disruptivity = gpuarray.zeros(NPTS, np.float32)
d_constraints = gpuarray.zeros(NCON * NDIM, np.float32)
d_constraints_prev = gpuarray.zeros(NCON * NDIM, np.float32)
d_constraint_sos = gpuarray.zeros(NCON, np.float32)
d_dot_products = gpuarray.zeros(NCON*NPTS, np.float32)
d_pt_satisfies_constraint = gpuarray.zeros(NCON*NPTS, np.bool_)
d_pt_satisfies_all_constraints = gpuarray.zeros(NPOP*NPTS, np.bool_)
d_n_safe_inside = gpuarray.zeros(NPOP, np.int32)
d_n_unsafe_inside = gpuarray.zeros(NPOP, np.int32)
d_cost = gpuarray.zeros(NPOP, np.float32)
d_tournament_members = gpuarray.zeros(NTOURN*NPOP, np.int32)
d_winners = gpuarray.zeros(NPOP, np.int32)
d_J_elite = gpuarray.zeros(NELITE, np.float32)
d_result = gpuarray.zeros(NLMI*NDIM, np.float32)

#initialize
init_kernel(np.uint32(1), np.int32(NPTS), np.int32(NCON), block = (TPB,1,1), grid = ((NPTS*NCON+TPB-1)//TPB,1,1), stream = strm1)

#scale range for input features
scale = np.array([
  [0.5e6, 6e5, 5e19, 0.05, 0.4, 0.02, 0.4, 0.5],
  [0.02e6, 2e4, 4e18, 0.005, 0.01, 0.0025, 0.005, 0.05]
],dtype=np.float32)
scale[1,:] = scale[1,:]
scale[0,:] = scale[0,:]

d_scale = gpuarray.zeros(2*NFEATURES, np.float32)
d_scale.set_async(scale.flatten())
	
important_features = np.array([1,1,0,0,0,0,0,0],dtype=np.uint32)
feature_names = ['$I_p$', '$W_{MHD}$', '${n}_e$', '$a$', 'Triang.', 'Square.', '$\kappa$', '$\ell_i$']
feature_plot_units = ['MA', 'MJ', '$10^{19}$', 'm', '', '', '', '']
plot_scales = [1.0e-6, 1.0e-6, 1.0e-19, 1, 1, 1, 1, 1] 
important_feature_indices = np.where(important_features)[0]
feature_names_used = []
feature_scales_used = []
for i in important_feature_indices:
	feature_names_used.append(feature_names[i])
	feature_scales_used.append(scale[0,i]*plot_scales[i])
d_important_features = gpuarray.zeros(NFEATURES, np.uint32)
d_important_features.set_async(important_features)
weights = 40.0

#DPRF inputs
with h5py.File(shot_data_file, 'r') as hf:
    shot_data = hf['X'][()]
    shot_time = hf['time'][()]
    print(shot_data.shape)
    print(shot_time.shape)
    current_operating_point = shot_data[0,:]

#DPRF parameters
with h5py.File(forest_file, 'r') as hf:
    feature = hf['feature'][()]
    children_left = hf['children_left'][()]
    children_right = hf['children_right'][()]
    tree_start = hf['tree_start'][()]
    threshold = hf['threshold'][()]
    value = hf['value'][()]
    n_trees = hf['n_trees'][()][0]
    n_nodes = hf['n_nodes'][()][0]
    n_classes = hf['n_classes'][()][0]
    
#DPRF gpu arrays
d_n_trees = gpuarray.zeros(1, np.int32)
d_n_trees.set_async(n_trees.flatten())

d_n_classes = gpuarray.zeros(1, np.int32)
d_n_classes.set_async(n_classes.flatten())

d_n_nodes = gpuarray.zeros(1, np.int32)
d_n_nodes.set_async(n_nodes.flatten())

d_feature = gpuarray.zeros(n_nodes, np.int32)
d_feature.set_async(feature)

d_children_left = gpuarray.zeros(n_nodes, np.int32)
d_children_left.set_async(children_left)

d_children_right = gpuarray.zeros(n_nodes, np.int32)
d_children_right.set_async(children_right)

d_tree_start = gpuarray.zeros(n_trees, np.int32)
d_tree_start.set_async(tree_start)

d_threshold = gpuarray.zeros(n_nodes, np.float32)
d_threshold.set_async(threshold)

d_value = gpuarray.zeros(n_nodes*n_classes, np.float32)
d_value.set_async(value.flatten())

d_npoints = gpuarray.zeros(1,np.int32)
d_npoints.set_async(np.int32(NPTS).flatten())

d_tree_result_scan = gpuarray.zeros(NPTS * n_trees, np.float32);
d_feature_contributions = gpuarray.zeros(NPTS * NFEATURES, np.float32)
d_feature_contributions_forest = gpuarray.zeros(NPTS * NFEATURES * n_trees, np.float32)
d_total_result_scan = gpuarray.zeros(NPTS, np.float32)
d_calculate_contributions = gpuarray.zeros(1, np.int32)
d_calculate_contributions.set_async(np.int32(0).flatten())


d_points = gpuarray.zeros(NPTS * NDIM, np.float32)
d_best = gpuarray.zeros(1,np.int32)

d_feature_points = gpuarray.zeros(NFEATURES * NPTS, np.float32)
	 
d_current_operating_point = gpuarray.zeros(NFEATURES, np.float32)

#evaluation_indices = np.arange(10000, 20300, 16)
evaluation_indices = np.arange(10000, len(shot_data), 10)
#evaluation_indices = np.arange(5,len(shot_data),1)
j_plot_points = 0
j_plot_result = 0
generate_plot_indices = np.array([18000,19040,19700])
#generate_plot_indices = np.array([50,100,220])

proximity_array = np.zeros((len(evaluation_indices),))
constraints_array = np.zeros((len(evaluation_indices),NLMI,NDIM))
points_array = np.zeros((len(evaluation_indices),NPTS,NDIM))
disruptivity_array = np.zeros((len(evaluation_indices),NPTS))
best = -1

#TODO: Update kernel inputs since some were modified
for i_data_index, data_index in enumerate(evaluation_indices):
	d_best.set_async(np.int32(best))
	print(data_index)
	current_operating_point = shot_data[data_index,:]
	d_current_operating_point.set_async(current_operating_point)

	generate_test_points_kernel(d_feature_points, d_points, np.int32(NPTS), np.int32(NFEATURES), np.int32(NDIM), d_scale, d_current_operating_point, d_important_features, 
		block = (TPB,1,1), grid = ((NPTS+TPB-1)//TPB,1,1), stream = strm1);

	#evaluate_disruptivity_kernel(d_points, d_disruptivity, block = (TPB,1,1), grid = ((NPTS+TPB-1)//TPB,1,1), stream = strm1)

	eval_forest_kernel(d_npoints, d_n_trees, d_feature_points, d_feature, d_children_left, d_children_right, d_tree_start, d_threshold, d_value, d_n_classes, d_tree_result_scan, d_feature_contributions, d_feature_contributions_forest, d_total_result_scan, d_calculate_contributions, block = (int(n_trees),1,1), grid = (NPTS,1,1), stream = strm1)


	cpu_points = d_points.get_async(stream = strm1).reshape((NPTS,NDIM))
	#cpu_disruptivity = d_disruptivity.get_async(stream = strm1).reshape((NPTS,1))
	cpu_disruptivity = d_total_result_scan.get_async(stream = strm1).reshape((NPTS,1))#/float(n_trees)

	#print(d_feature_points.get_async(stream = strm1))
	#print(cpu_disruptivity)
	#quit()
	if ((j_plot_points<len(generate_plot_indices)) and (data_index >= generate_plot_indices[j_plot_points])):
		j_plot_points = j_plot_points + 1
		fig,ax = plt.subplots()
		#scatter_plot = ax.scatter(cpu_points[:,0],cpu_points[:,1],c=cpu_disruptivity.reshape((NPTS,))<disruptivity_threshold)
		scatter_plot = ax.tricontour(cpu_points[:,0],cpu_points[:,1],cpu_disruptivity.reshape((NPTS,)),colors=['midnightblue','firebrick'],levels=[disruptivity_threshold*0.75,disruptivity_threshold])
		scatter_plot = ax.tricontourf(cpu_points[:,0],cpu_points[:,1],cpu_disruptivity.reshape((NPTS,)),cmap=cm.Spectral_r)
		scatter_plot = ax.scatter(cpu_points[:,0],cpu_points[:,1],c=cpu_disruptivity.reshape((NPTS,)),linewidths=1,edgecolors='k',cmap=cm.Spectral_r)
		fig.colorbar(scatter_plot,ax=ax)
		ax.set_xlim([-1.0,1.0])
		ax.set_ylim([-1.0,1.0])
		ticks_x = plt.xticks()[0]
		plt.xticks(ticks=ticks_x,labels= ["%.2f" % number for number in ticks_x*feature_scales_used[0]])
		ticks_y = plt.yticks()[0]
		plt.yticks(ticks=ticks_y,labels= ["%.2f" % number for number in ticks_y*feature_scales_used[1]])
		#ax.ticklabel_format(axis='both',scilimits=(-2,2))
		if len(feature_plot_units[important_feature_indices[0]])>0:
			ax.set_xlabel(f'$\Delta$ {feature_names_used[0]} [{feature_plot_units[important_feature_indices[0]]}]')
		else:
			ax.set_xlabel(f'$\Delta$ {feature_names_used[0]}')
		if len(feature_plot_units[important_feature_indices[1]])>0:
			ax.set_ylabel(f'$\Delta$ {feature_names_used[1]} [{feature_plot_units[important_feature_indices[1]]}]')
		else:
			ax.set_ylabel(f'$\Delta$ {feature_names_used[1]}')
		ax.set_aspect('equal')
		title = (
			f"DIII-D Shot = 180808, Time = {shot_time[data_index]:.3f}s\n"
			f"{feature_names_used[0]}={current_operating_point[important_feature_indices[0]]*plot_scales[important_feature_indices[0]]:.3f} {feature_plot_units[important_feature_indices[0]]}, "
			f"{feature_names_used[1]}={current_operating_point[important_feature_indices[1]]*plot_scales[important_feature_indices[1]]:.3f} {feature_plot_units[important_feature_indices[1]]}"
		)
		ax.set_title(title)
		#plt.savefig(f'./{results_dir}/{results_dir}_dprf_points_{data_index}.png')
	#quit()
	random_points_kernel(d_constraints, np.int32(NDIM), np.int32(NCON), np.float32(1.0), np.float32(0.5*1.0), d_best, block = (TPB,1,1), grid = ((NCON+TPB-1)//TPB,1,1), stream = strm1)
	constraints = d_constraints.get_async(stream=strm1).reshape(NCON,NDIM)
	#print(constraints)
	#for j in np.arange(NPOP):
	#	A = constraints[j*NLMI:(j+1)*NLMI,:]

	#	fig,ax = plt.subplots()
	#	ax.scatter(cpu_points[:,0],cpu_points[:,1],c=cpu_disruptivity.reshape((NPTS,))<0.5)
	#	ax.scatter(A[:,0],A[:,1])

	#	for i in np.arange(NLMI):
	#		ax.plot([0, A[i,0]],[0,A[i,1]],linewidth=3)
	#		ax.plot([A[i,0], A[i,0]-100*A[i,1]],[A[i,1],A[i,1]+100*A[i,0]],'k',linewidth=3)
	#		ax.plot([A[i,0], A[i,0]+100*A[i,1]],[A[i,1],A[i,1]-100*A[i,0]],'k',linewidth=3)
	#	ax.set_xlim([-1.0,1.0])
	#	ax.set_ylim([-1.0,1.0])
	#	ax.set_aspect('equal')
	#	#plt.show()
	#	plt.savefig(f"SORI_population_{j}.png")

	for gen in np.arange(NGEN):
		evaluate_constraint_sos_kernel(d_constraints, d_constraint_sos, np.int32(NDIM), np.int32(NCON), block = (TPB,1,1), grid = ((NCON+TPB-1)//TPB,1,1), stream = strm1)

		#print(d_constraint_sos.get_async(stream=strm1))

		gpu_mmul_ABT_kernel(d_constraints, d_points, d_dot_products, np.int32(NCON), np.int32(NDIM), np.int32(NPTS),block = (TPB,1,1), grid = ((NCON*NPTS+TPB-1)//TPB,1,1), stream = strm1)

		GpuCopy_kernel(d_constraints_prev, d_constraints, np.int32(NCON), np.int32(NDIM), block = (TPB,1,1), grid = ((NCON*NDIM+TPB-1)//TPB,1,1), stream = strm1)

		evaluate_constraints_satisfied_kernel(d_dot_products, d_constraint_sos, d_pt_satisfies_constraint, np.int32(NCON), np.int32(NPTS), block = (TPB,1,1), grid = ((NCON*NPTS+TPB-1)//TPB,1,1), stream = strm1)

		evaluate_all_constraints_satisfied_kernel(d_pt_satisfies_constraint, d_pt_satisfies_all_constraints, d_n_safe_inside, d_n_unsafe_inside, d_total_result_scan, np.float32(disruptivity_threshold), np.float32(weights), d_cost, np.int32(NPTS), np.int32(NPOP), np.int32(NLMI), np.int32(NCON), block = (TPB,1,1), grid = ((NPOP*NPTS+TPB-1)//TPB,1,1), stream = strm1)

		pt_satisfies_all_constraints = d_pt_satisfies_all_constraints.get_async(stream=strm1)
		#print(f"d_n_safe_inside: {d_n_safe_inside.get_async(stream=strm1)}")
		#print(f"d_n_unsafe_inside: {d_n_unsafe_inside.get_async(stream=strm1)}")
		#print(f"d_cost: {d_cost.get_async(stream=strm1)}")
		constraints = d_constraints.get_async(stream=strm1).reshape(NCON,NDIM)
		
		if False:
			for j in np.arange(NPOP):
				A = constraints[j*NLMI:(j+1)*NLMI,:]

				fig,ax = plt.subplots()
				ax.scatter(cpu_points[:,0],cpu_points[:,1],50,c=pt_satisfies_all_constraints[j*NPTS:(j+1)*NPTS])
				ax.scatter(cpu_points[:,0],cpu_points[:,1],20,c=cpu_disruptivity.reshape((NPTS,))<disruptivity_threshold)
				ax.scatter(A[:,0],A[:,1])

				for i in np.arange(NLMI):
					ax.plot([0, A[i,0]],[0,A[i,1]],linewidth=3)
					ax.plot([A[i,0], A[i,0]-100*A[i,1]],[A[i,1],A[i,1]+100*A[i,0]],'k',linewidth=3)
					ax.plot([A[i,0], A[i,0]+100*A[i,1]],[A[i,1],A[i,1]-100*A[i,0]],'k',linewidth=3)
				ax.set_xlim([-1.0,1.0])
				ax.set_ylim([-1.0,1.0])
				ax.set_aspect('equal')
				#plt.savefig(f"./{results_dir}/{results_dir}_SORI_population_{j}_{gen}.png")

		genetic_select_kernel(d_cost, d_tournament_members, d_winners, d_constraints, d_constraints_prev, np.int32(NPOP), np.int32(NTOURN), np.int32(NCON), np.int32(NLMI), np.int32(NDIM), block = (TPB,1,1), grid = ((NPOP*NTOURN+TPB-1)//TPB,1,1), stream = strm1)

		genetic_mate_kernel(d_constraints, np.float32(0.15), np.int32(NPOP), np.int32(NLMI), np.int32(NDIM), np.int32(NCON), block = (TPB,1,1), grid = ((NPOP//2+TPB-1)//TPB,1,1), stream = strm1)

		genetic_mutate_kernel(d_constraints, np.float32(0.3), np.float32(0.1), np.int32(NCON*NDIM), block = (TPB,1,1), grid = ((NCON*NDIM+TPB-1)//TPB,1,1), stream = strm1)

		carry_over_elite_kernel(d_cost, d_J_elite, d_constraints, d_constraints_prev, d_result, np.int32(NELITE), np.int32(NPOP%NELITE), np.int32(NPOP//NELITE),  np.int32(NLMI), np.int32(NDIM), np.int32(NCON), block = (NLMI*NDIM,1,1), grid = (NELITE,1,1), stream = strm1)

		constraints_new = d_constraints.get_async(stream=strm1)
		J_elite = d_J_elite.get_async(stream=strm1)

		#print(J_elite)
		best = np.argmax(J_elite)
		#print(best)

	constraints_new = constraints_new.reshape(NCON,NDIM)
	#A = constraints[0*NLMI:(0+1)*NLMI,:]
	#print(constraints_new)
	A = constraints_new[best*(NPOP//NELITE)*NLMI:best*(NPOP//NELITE)*NLMI+NLMI,:]

	proximities = np.zeros((NLMI,))
	print(A)
	for jlmi in np.arange(NLMI):
		proximities[jlmi] = np.sqrt((A[jlmi,0]*feature_scales_used[0]/(plot_scales[important_feature_indices[0]]*scale[1,important_feature_indices[0]]))**2 + (A[jlmi,1]*feature_scales_used[1]/(plot_scales[important_feature_indices[1]]*scale[1,important_feature_indices[1]]))**2)
	print(proximities)
	proximity_array[i_data_index] = np.min(proximities)
	constraints_array[i_data_index,:,:] = A
	points_array[i_data_index,:,:] = cpu_points
	disruptivity_array[i_data_index,:] = np.ravel(cpu_disruptivity)
	print(proximity_array[i_data_index])
				
	if ((j_plot_result<len(generate_plot_indices)) and (data_index >= generate_plot_indices[j_plot_result])):
		nxg, nyg = (100, 100)
		xg = np.linspace(-1, 1, nxg)
		yg = np.linspace(-1, 1, nyg)
		xvg, yvg = np.meshgrid(xg, yg)
		zvg = np.ones(xvg.shape)
		for ig in np.arange(nxg):
			for jg in np.arange(nyg):
				for i in np.arange(NLMI):
					zvg[ig,jg] = np.logical_and(zvg[ig,jg],(A[i,0]*xvg[ig,jg] + A[i,1]*yvg[ig,jg]) < A[i,0]**2 + A[i,1]**2)
		j_plot_result = j_plot_result + 1	
		fig,ax = plt.subplots()
		import map_colormaps
		color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
		RdBu_light = map_colormaps.cmap_map(lambda x: x/2+0.5, cm.coolwarm_r)
		ax.pcolor(xvg,yvg,zvg,cmap=RdBu_light)
		ax.scatter(cpu_points[:,0],cpu_points[:,1],c=cpu_disruptivity.reshape((NPTS,))<disruptivity_threshold,cmap=cm.coolwarm_r,linewidths=1,edgecolors='k')
		ax.scatter(A[:,0],A[:,1])
		
		for i in np.arange(NLMI):
			color2use = color_cycle[i]
			ax.plot([0, A[i,0]],[0,A[i,1]],linewidth=3,color=color2use)
			ax.plot([A[i,0], A[i,0]-100*A[i,1]],[A[i,1],A[i,1]+100*A[i,0]],color=color2use,linewidth=3)
			ax.plot([A[i,0], A[i,0]+100*A[i,1]],[A[i,1],A[i,1]-100*A[i,0]],color=color2use,linewidth=3)
		ax.set_xlim([-1.0,1.0])
		ax.set_ylim([-1.0,1.0])
		ticks_x = plt.xticks()[0]
		plt.xticks(ticks=ticks_x,labels= ["%.2f" % number for number in ticks_x*feature_scales_used[0]])
		ticks_y = plt.yticks()[0]
		plt.yticks(ticks=ticks_y,labels= ["%.2f" % number for number in ticks_y*feature_scales_used[1]])
		#ax.ticklabel_format(axis='both',scilimits=(-2,2))
		if len(feature_plot_units[important_feature_indices[0]])>0:
			ax.set_xlabel(f'$\Delta$ {feature_names_used[0]} [{feature_plot_units[important_feature_indices[0]]}]')
		else:
			ax.set_xlabel(f'$\Delta$ {feature_names_used[0]}')
		if len(feature_plot_units[important_feature_indices[1]])>0:
			ax.set_ylabel(f'$\Delta$ {feature_names_used[1]} [{feature_plot_units[important_feature_indices[1]]}]')
		else:
			ax.set_ylabel(f'$\Delta$ {feature_names_used[1]}')
		ax.set_aspect('equal')
		title = (
			f"DIII-D Shot = 180808, Time = {shot_time[data_index]:.3f}s\n"
			f"{feature_names_used[0]}={current_operating_point[important_feature_indices[0]]*plot_scales[important_feature_indices[0]]:.3f} {feature_plot_units[important_feature_indices[0]]}, "
			f"{feature_names_used[1]}={current_operating_point[important_feature_indices[1]]*plot_scales[important_feature_indices[1]]:.3f} {feature_plot_units[important_feature_indices[1]]}"
		)
		ax.set_title(title)
		#plt.show()
		plt.savefig(f'./{results_dir}/{results_dir}_SORI_result_{data_index}.png')

	feature_points = d_feature_points.get_async(stream=strm1)
	#print(f'Feature points shape: {feature_points.shape}')
	#print(feature_points[:8])

print(proximity_array)
fig,ax = plt.subplots()
plt.plot(shot_time[evaluation_indices],proximity_array)
#plt.show()
#plt.plot(np.diff(shot_time))
plt.savefig(f'./{results_dir}/{results_dir}_SORI_proximity.png')
#Archive data
archive_filename = f"./{results_dir}/{results_dir}_archive.h5"

if len(sys.argv)-1 == 0: 
	with h5py.File(archive_filename,"w") as f:
		f.create_dataset('results/proximity', data=proximity_array)
		f.create_dataset('results/disruptivity', data=disruptivity_array)
		f.create_dataset('results/constraints', data=constraints_array)
		f.create_dataset('results/points', data=points_array)
		f.attrs['shot_data_file'] = shot_data_file
		f.attrs['forest_file'] = forest_file
		for (var,val) in zip(['TPB','NDIM','NPTS','NLMI','NPOP','NGEN','NCON',
		'NTOURN','NELITE','NFEATURES','disruptivity_threshold','scale','important_features','weights','feature_names','feature_plot_units','plot_scales','important_feature_indices','feature_names_used','feature_scales_used','evaluation_indices','generate_plot_indices'],[TPB,NDIM,NPTS,NLMI,NPOP,NGEN,NCON,NTOURN,NELITE,NFEATURES,disruptivity_threshold,scale,important_features,weights,feature_names,feature_plot_units,plot_scales,important_feature_indices,feature_names_used,feature_scales_used,evaluation_indices,generate_plot_indices]):
			f.attrs[var] = val
		
	

# Free gpu memory
d_points.gpudata.free()
d_feature_points.gpudata.free()
d_scale.gpudata.free()
d_current_operating_point.gpudata.free()
d_important_features.gpudata.free()
d_constraints.gpudata.free()
d_constraints_prev.gpudata.free()
d_constraint_sos.gpudata.free()
d_dot_products.gpudata.free()
d_pt_satisfies_constraint.gpudata.free()
d_pt_satisfies_all_constraints.gpudata.free()
d_n_safe_inside.gpudata.free()
d_n_unsafe_inside.gpudata.free()
d_cost.gpudata.free()
d_tournament_members.gpudata.free()
d_winners.gpudata.free()
d_J_elite.gpudata.free()
d_result.gpudata.free()

d_n_trees.gpudata.free()
d_n_classes.gpudata.free()
d_n_nodes.gpudata.free()
d_feature.gpudata.free()
d_children_left.gpudata.free()
d_children_right.gpudata.free()
d_tree_start.gpudata.free()
d_threshold.gpudata.free()
d_value.gpudata.free()
d_npoints.gpudata.free()
d_feature_contributions.gpudata.free()
d_feature_contributions_forest.gpudata.free()
d_total_result_scan.gpudata.free()
d_tree_result_scan.gpudata.free()
d_calculate_contributions.gpudata.free()

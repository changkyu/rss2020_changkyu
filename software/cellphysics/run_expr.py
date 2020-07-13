import os, sys
import argparse
import pickle
import glob
import yaml

from DiffPhysics import DiffPhysics
#from lcp_physics_wrapper import MDiffPhysics
from OtherPhysics import *

from setups_sim import get_setups_tools_sim
idxes_set = [
[[[0], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
 [[1], [0, 2, 3, 4, 5, 6, 7, 8, 9]],
 [[2], [0, 1, 3, 4, 5, 6, 7, 8, 9]],
 [[3], [0, 1, 2, 4, 5, 6, 7, 8, 9]],
 [[4], [0, 1, 2, 3, 5, 6, 7, 8, 9]],
 [[5], [0, 1, 2, 3, 4, 6, 7, 8, 9]],
 [[6], [0, 1, 2, 3, 4, 5, 7, 8, 9]],
 [[7], [0, 1, 2, 3, 4, 5, 6, 8, 9]],
 [[8], [0, 1, 2, 3, 4, 5, 6, 7, 9]],
 [[9], [0, 1, 2, 3, 4, 5, 6, 7, 8]]],
[[[2, 6], [0, 1, 3, 4, 5, 7, 8, 9]],
 [[6, 5], [0, 1, 2, 3, 4, 7, 8, 9]],
 [[9, 6], [0, 1, 2, 3, 4, 5, 7, 8]],
 [[8, 9], [0, 1, 2, 3, 4, 5, 6, 7]],
 [[3, 4], [0, 1, 2, 5, 6, 7, 8, 9]],
 [[1, 0], [2, 3, 4, 5, 6, 7, 8, 9]],
 [[0, 7], [1, 2, 3, 4, 5, 6, 8, 9]],
 [[4, 9], [0, 1, 2, 3, 5, 6, 7, 8]],
 [[5, 6], [0, 1, 2, 3, 4, 7, 8, 9]],
 [[0, 3], [1, 2, 4, 5, 6, 7, 8, 9]]],
[[[6, 4, 9], [0, 1, 2, 3, 5, 7, 8]],
 [[5, 9, 3], [0, 1, 2, 4, 6, 7, 8]],
 [[6, 8, 5], [0, 1, 2, 3, 4, 7, 9]],
 [[4, 9, 8], [0, 1, 2, 3, 5, 6, 7]],
 [[3, 9, 6], [0, 1, 2, 4, 5, 7, 8]],
 [[3, 4, 6], [0, 1, 2, 5, 7, 8, 9]],
 [[7, 8, 1], [0, 2, 3, 4, 5, 6, 9]],
 [[7, 2, 3], [0, 1, 4, 5, 6, 8, 9]],
 [[4, 3, 0], [1, 2, 5, 6, 7, 8, 9]],
 [[0, 5, 8], [1, 2, 3, 4, 6, 7, 9]]],
[[[0, 6, 7, 2], [1, 3, 4, 5, 8, 9]],
 [[0, 4, 8, 6], [1, 2, 3, 5, 7, 9]],
 [[6, 7, 5, 8], [0, 1, 2, 3, 4, 9]],
 [[7, 4, 2, 8], [0, 1, 3, 5, 6, 9]],
 [[6, 8, 4, 5], [0, 1, 2, 3, 7, 9]],
 [[7, 5, 9, 2], [0, 1, 3, 4, 6, 8]],
 [[4, 1, 7, 8], [0, 2, 3, 5, 6, 9]],
 [[8, 1, 2, 4], [0, 3, 5, 6, 7, 9]],
 [[2, 4, 1, 7], [0, 3, 5, 6, 8, 9]],
 [[7, 0, 4, 3], [1, 2, 5, 6, 8, 9]]],
[[[1, 6, 8, 5, 9], [0, 2, 3, 4, 7]],
 [[9, 6, 1, 4, 2], [0, 3, 5, 7, 8]],
 [[6, 9, 8, 0, 7], [1, 2, 3, 4, 5]],
 [[6, 4, 8, 5, 0], [1, 2, 3, 7, 9]],
 [[1, 2, 3, 8, 6], [0, 4, 5, 7, 9]],
 [[8, 2, 1, 4, 5], [0, 3, 6, 7, 9]],
 [[8, 5, 9, 7, 3], [0, 1, 2, 4, 6]],
 [[1, 4, 2, 6, 0], [3, 5, 7, 8, 9]],
 [[2, 7, 6, 4, 3], [0, 1, 5, 8, 9]],
 [[2, 6, 9, 5, 7], [0, 1, 3, 4, 8]]],
]

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-i','--indir', type=str,  default='',     help='input expr file')
    parser.add_argument('-o','--outdir',type=str,  default='./res',help='output directory')
    parser.add_argument('-v','--video', type=str,  default='',    help='output video')
    parser.add_argument('-p','--param', type=int,  default=-1,    help='idx params')
    parser.add_argument('-m','--method',type=str,  required=True, help='method')
    parser.add_argument('--sets',       type=int,  nargs='+',     help='set idxes')
    parser.add_argument('--objects',    type=str,  nargs='+',     help='objects')
    parser.add_argument('--n_trains',   type=int,  nargs='+',     help='n_trains')
    parser.add_argument('--n_iters',    type=int,  default=50,    help='n_iters')
    parser.add_argument('--skip',       type=bool, default=False, help='skip')
    parser.add_argument('--debug',      type=bool, default=False, help='visualization')
    args = parser.parse_args()

    if len(args.video) > 0 and not os.path.isdir(os.path.dirname(args.video)):
        print("[Error] Invalid path: ", args.video)
        exit(0)

    if args.method not in ['mdiff_cell',  'rand_cell', 'grand_cell', 'bf_cell', 'grid_cell', 'fdiff_cell', 'diff_cell',  'cma_cell',  'nm_cell',
                           'mdiff_group', 'rand_group','grand_group','bf_group','grid_group','fdiff_group','diff_group', 'cma_group', 'nm_group']:
        print('[Error] Invalid method: ', args.method)
        exit(0)

    if args.indir=='' and args.param<=0:
        print('[Error] set param (for sim) or set indir directory (for real)')
        exit(0)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    def get_engine(method):
        if args.method.startswith('rand'):
            engine = RandomSearchPhysics(is_dbg=args.debug, fp_video=args.video)
        elif args.method.startswith('grand'):
            engine = GaussianRandomPhysics(is_dbg=args.debug, fp_video=args.video)
        elif args.method.startswith('bf'):
            engine = BruteForceSearchPhysics(is_dbg=args.debug, fp_video=args.video)
        elif args.method.startswith('grid'):
            engine = GridSearchPhysics(is_dbg=args.debug, fp_video=args.video)   
        elif args.method.startswith('fdiff'):
            engine = FiniteDiffPhysics(is_dbg=args.debug, fp_video=args.video)
        elif args.method.startswith('diff'):
            engine = DiffPhysics(is_dbg=args.debug, fp_video=args.video)
        elif args.method.startswith('mdiff'):
            engine = MDiffPhysics(is_dbg=args.debug, fp_video=args.video)
        elif args.method.startswith('cma'):
            engine = CMAPhysics(is_dbg=args.debug, fp_video=args.video)
        elif args.method.startswith('nm'):
            engine = NelderMeadPhysics(is_dbg=args.debug, fp_video=args.video)
        else:
            print('[Error] Invalid method: ', method)
            exit(0)

        return engine        

    obj2ngroups = {'book':28,'box':65,'cramp':86,'crimp':44,'hammer':88,'ranch':52,'snack':64,'toothpaste':74,'windex':84}

    if args.sets is None or len(args.sets)==0:
        args.sets = range(10)

    if args.n_trains is None or len(args.n_trains)==0:
        args.n_trains = range(5,0,-1)

    # simulation
    if args.param > 0:
        if args.objects is None or len(args.objects)==0:
            args.objects = ['book','box','cramp','crimp','hammer','ranch','snack','toothpaste','windex']
        
        if args.method.endswith('group'):
            n_groups = [3,10,20]
        else:
            n_groups = [-1]

        times_sims = []
        times_grads = []

        for n_trains in args.n_trains:
            for obj in args.objects:
                for n_group in n_groups:

                    print('%s - # of train: %d' % (obj,n_trains))

                    idx_param = args.param - 1

                    for idx in args.sets:

                        if args.skip:
                            fp_save = os.path.join(args.outdir,
                                                   '%s_%s_param%d_train%d_set%d_c%d.pkl'%\
                                                   (args.method,obj,idx_param,n_trains,idx,obj2ngroups[obj]))
                            if os.path.isfile(fp_save):
                                print('[SKIP] %s'%fp_save)
                                continue
                        
                        idxes_train, idxes_test = idxes_set[n_trains-1][idx]

                        setups = get_setups_tools_sim([obj],idxes_train,idxes_test,idx_param)
                        engine = get_engine(args.method)
                                   
                        if args.method.endswith('_cell'):                    
                            engine.setup(setups, n_group=n_group)
                            history = engine.infer(infer_type='cell_mass',nsimslimit=100)
                            n_group = len(engine.meta_targets[obj]['cells']['cells'])
                        else:                    
                            engine.setup(setups, n_group=n_group)
                            history = engine.infer(infer_type='group_mass')

                        print('Test Result: %.3f (m) %.3f (deg) %.3f (cell)' % 
                               (history[-1]['test_error_xy'],
                                history[-1]['test_error_yaw'],
                                history[-1]['test_error_cell']) )

                        summary = {'object':obj,
                                   'idxes_train':idxes_train,
                                   'idxes_test':idxes_test,
                                   'history':history,
                                  }

                        fp_save = os.path.join(args.outdir,
                                               '%s_%s_param%d_train%d_set%d_c%d.pkl'%\
                                               (args.method,obj,idx_param,n_trains,idx,n_group))

                        with open(fp_save,'wb') as f:
                            pickle.dump(summary, f, protocol=2)

                        times_sims.append( history[-1]['time_sims']/float(history[-1]['n_sims']) )
                        times_grads.append(history[-1]['time_grads']/float(history[-1]['n_grads']))
    
        print('average time / a simulation: {} (\pm{})'.format(np.mean(times_sims),  np.std(times_sims)))
        print('average time / a gradient: {} (\pm{})'.format(  np.mean(times_grads), np.std(times_grads)))

    # real experiment
    else:

        folds = [[[9, 5, 12, 1, 4, 3, 7], [0, 2, 6, 8, 10, 11, 13]],
                 [[7, 0, 10, 8, 12, 5, 9], [1, 2, 3, 4, 6, 11, 13]],
                 [[0, 4, 6, 8, 10, 1, 2], [3, 5, 7, 9, 11, 12, 13]],
                 [[9, 5, 12, 11, 13, 1, 0], [2, 3, 4, 6, 7, 8, 10]],
                 [[7, 0, 6, 4, 9, 3, 11], [1, 2, 5, 8, 10, 12, 13]],
                 [[4, 2, 3, 5, 12, 8, 9], [0, 1, 6, 7, 10, 11, 13]],
                 [[9, 10, 5, 0, 3, 8, 13], [1, 2, 4, 6, 7, 11, 12]],
                 [[1, 11, 9, 10, 8, 12, 0], [2, 3, 4, 5, 6, 7, 13]],
                 [[2, 0, 7, 4, 13, 6, 8], [1, 3, 5, 9, 10, 11, 12]],
                 [[2, 0, 13, 8, 7, 12, 6], [1, 3, 4, 5, 9, 10, 11]]]

        if args.objects is None or len(args.objects)==0:
            args.objects = ['book','box','hammer','snack','windex']

        for obj in args.objects:

            fp_label = os.path.join('../../dataset/objects',obj+'_label.pgm')
            setup = {'actor':{'name':'finger','type':'finger'},
                     'target_infos':[{'name':obj,'label':fp_label,'masses':[1,1,1],'meter_per_pixel':0.53/640.0}],
                     'mass_minmax':[0.1, 5.0],
                     'train':[],
                     'test':[],
                     'real':[]}

            files = glob.glob(os.path.join(args.indir, obj+'_*_traj.txt'))
            files = sorted(files)

            for file in files:

                f = open(file,'r')
                node = yaml.load(f, Loader=yaml.BaseLoader)
                f.close()

                vec_x = float(node['push']['vec']['x'])
                vec_x = vec_x/abs(vec_x)

                push_x = float(node['push']['x']) - float(node['begin']['center_of_pixels']['x']) - vec_x*0.10
                push_y = float(node['push']['y']) - float(node['begin']['center_of_pixels']['y'])

                x_res = float(node['transform']['x'])
                y_res = float(node['transform']['y'])                
                yaw_res = float(node['transform']['yaw'])
                
                action = {'finger':{'pos': [push_x,push_y], 'yaw':0,
                                    'velocity':[float(node['push']['vec']['x']) + vec_x*0.10,
                                                float(node['push']['vec']['y']),
                                                float(node['push']['vec']['z'])],
                                    'duration':1.0},
                          'targets':[obj], 
                          obj:{'pos':[0,0], 'yaw':0, 
                               'pos_res':[x_res,y_res], 'yaw_res':yaw_res}
                         }
                setup['real'].append(action)

            for idx in args.sets:

                idxes_train, idxes_test = folds[idx]
                setup['train'] = idxes_train
                setup['test'] = idxes_test

                engine = get_engine(args.method)
                engine.setup(setup, n_group=-1)
                n_group = len(engine.meta_targets[obj]['cells']['cells'])
    
                history = engine.infer(infer_type='cell_mass',nsimslimit=500)
    
                print('[Test Result] cell err: %.3f (m) pose err: %.3f (m) %.3f (deg)' % 
                       (history[-1]['test_error_cell'],
                        history[-1]['test_error_xy'],
                        history[-1]['test_error_yaw']) )
    
                summary = {'object':obj,
                           'idxes_train':idxes_train,
                           'idxes_test':idxes_test,
                           'history':history,
                          }
    
                fp_save = os.path.join(args.outdir,
                                       '%s_%s_real_set%d_c%d.pkl'%\
                                       (args.method,obj,idx,n_group))
                with open(fp_save,'wb') as f:
                    pickle.dump(summary, f, protocol=2)

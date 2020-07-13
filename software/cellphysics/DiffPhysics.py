import time
import scipy
import numpy as np
from PhysicsBase import CellPhysicsBase

class DiffPhysics(CellPhysicsBase):

    def __init__(self, is_dbg=False, fp_video=''):
        CellPhysicsBase.__init__(self,is_dbg=is_dbg,fp_video=fp_video)

        self.n_grads = 0
        self.time_grads = 0
    
    def M(self, obj, masses):

        cells = self.meta_targets[obj]['cells']['cells']
        cell_size = self.meta_targets[obj]['cells']['cell_size']
        
        n = len(cells)
        M = np.zeros([3*n,3*n])
        for c, m in enumerate(masses):
            m = 1.0
            I = m * (cell_size**2) * 2 / float(12)
            M_i = np.array([[I,0,0],[0,m,0],[0,0,m]])
            M[3*c:3*(c+1),3*c:3*(c+1)] = M_i

        return M

    def Je(self, obj, cellpos):

        c_ctr = self.meta_targets[obj]['cells']['idx_center']
        n = len(cellpos)

        x_ctr = cellpos[c_ctr][1]
        y_ctr = cellpos[c_ctr][2]

        Je = np.zeros([3*(n-1),3*n])
        i = 0
        for c, pos in enumerate(cellpos):
            x = pos[1]
            y = pos[2]
            if c != c_ctr:
                Je_1 = np.array([[-(y_ctr-y), 1, 0],\
                                 [ (x_ctr-x), 0, 1],
                                 [         1, 0, 0]])
                Je_c = np.array([[         0,-1, 0],\
                                 [         0, 0,-1],
                                 [         1, 0, 0]])
                Je[3*i:3*(i+1),3*c    :3*(c    +1)] = Je_1
                Je[3*i:3*(i+1),3*c_ctr:3*(c_ctr+1)] = Je_c
                i = i + 1

        return Je

    def Jf(self, obj, cellvel):

        cell_size = self.meta_targets[obj]['cells']['cell_size']

        n = len(cellvel)
        Jf = np.zeros([2*n,3*n])
        for c, vel in enumerate(cellvel):
            Jf[2*c,  3*c] = 1 if vel[0] < 0 else -1
            Jf[2*c+1,3*c+1:3*(c+1)] = -vel[1:2]/np.linalg.norm(vel[1:2])
        
        return Jf

    def Mu_c(self, obj):

        cells = self.meta_targets[obj]['cells']['cells']
        n = len(cells)
        C = np.zeros([2*n,3*n])
        for c in range(n):
            C[2*c,  3*c] = 1
            C[2*c+1,3*c+1:3*(c+1)] = 0.5
        return C

    def computeGrad(self, name, masses, cellpos_ref, traj_now):

        time_start = time.time()
        def lVlMu(M_inv, Je, Jf, Mu_c, Vtdt, Vt):
            JeT = Je.transpose()            
            return (M_inv + M_inv.dot(JeT).dot(np.linalg.inv(-Je.dot(M_inv).dot(JeT))).dot(Je).dot(M_inv)).dot(Jf.transpose().dot(Mu_c) - np.diag(Vtdt-Vt))

        M = self.M(name, masses)
        M_inv = np.diag(1.0/np.diag(M))
        Mu_c = self.Mu_c(name)

        dldV = (np.array(traj_now['cellpos'][-1])-np.array(cellpos_ref)).reshape(-1)

        X = np.zeros(M.shape[0])
        #n_iters = len(traj_now['cellpos'])
        #for i in range(1,n_iters):
        #    cellpos_pprv= traj_now['cellpos'][i-(2 if i>1 else 1)]
        #    cellpos_prv = traj_now['cellpos'][i-1]
        #    cellpos_cur = traj_now['cellpos'][i]
        #    cellvel_prv = np.array(cellpos_prv)-np.array(cellpos_pprv)
        #    cellvel_cur = np.array(cellpos_cur)-np.array(cellpos_prv)
        #    Je = self.Je(name, cellpos_cur)
        #    Jf = self.Jf(name, cellvel_cur)
        #    X = X + lVlMu(M_inv, Je, Jf, Mu_c, cellvel_cur.reshape(-1), cellvel_prv.reshape(-1))
        #X = X / float(n_iters-1)

        cellvel_prv = np.array(traj_now['cellpos'][-1])-np.array(traj_now['cellpos'][0])
        cellvel_cur = np.array(traj_now['cellpos'][-1])-np.array(traj_now['cellpos'][0])
        Je = self.Je(name, traj_now['cellpos'][-1])
        Jf = self.Jf(name, cellvel_cur)
        X = lVlMu(M_inv, Je, Jf, Mu_c, cellvel_cur.reshape(-1), cellvel_prv.reshape(-1))

        grad_masses = dldV.dot(X)
        ret = grad_masses.reshape(-1,3)[:,1].squeeze()

        self.time_grads = self.time_grads + (time.time() - time_start)
        self.n_grads = self.n_grads + 1
        
        return ret

    def infer(self, n_iters=100, infer_type='cell_mass', timelimit=np.inf,
                    nsimslimit=np.inf, cellinfos_init=None, errlimit=0 ):
        # infer
        history = []
        self.n_sims = 0
        self.n_grads = 0
        self.time_sims = 0
        self.time_grads = 0
        time_start = time.time()
        err_min = [np.inf]
        cache = {}
        def simulate(masses):

            m = 0
            for name in self.meta_targets:
                cellinfo = self.sim.getCellBodyProperties(name)
                for info in cellinfo:
                    info['mass'] = masses[m]
                    m = m + 1                            
                self.sim.setCellBodyProperties(name,cellinfo)

            errs_xy = []
            errs_yaw = []
            errs_cell = []

            grad_all = []
            for i, action in enumerate(self.actions_train):

                traj = self.simulate(action,traj_names=action['targets'])

                grad_objs = []
                for name in action['targets']:
                    x,y,yaw = self.sim.getObjectPose(name)
                    cellpos = self.sim.getCellBodyPose(name)

                    x_ref   = self.poses_refs[i][name]['x']
                    y_ref   = self.poses_refs[i][name]['y']
                    yaw_ref = self.poses_refs[i][name]['yaw']
                    cellpos_ref = self.poses_refs[i][name]['cellpos']

                    errs_xy.append(np.linalg.norm(np.array([x_ref-x,y_ref-y])))
                    errs_yaw.append((36000+abs(yaw_ref-yaw)/np.pi*18000)%36000*0.01)
                    errs_cell.append(self.distanceCellPos(cellpos_ref,cellpos))

                    grad_obj = self.computeGrad(name,masses,cellpos_ref,traj[name])
                    grad_objs = grad_objs + list(grad_obj)

                grad_all.append(grad_objs)

            grad = np.mean(np.array(grad_all),axis=0)
            
            err_xy = np.mean(np.array(errs_xy))
            err_yaw = np.mean(np.array(errs_yaw))
            err_cell = np.mean(np.array(errs_cell))

            cache[tuple(masses)] = {'loss':err_cell,'grad':grad}

            if err_min[0] > err_cell:
                err_min[0] = err_cell
                cellinfos = {}
                for name in self.meta_targets:
                    cellinfos[name] = self.sim.getCellBodyProperties(name)
                cellinfos_best = cellinfos
                masses_min = masses
                history.append({'n_sims':self.n_sims,
                                'time':time.time()-time_start,
                                'time_sims':self.time_sims,
                                'n_grads':self.n_grads,
                                'time_grads':self.time_grads,
                                'error_xy'  :err_xy,
                                'error_yaw' :err_yaw,
                                'error_cell':err_cell,
                                'cellinfos' :cellinfos,
                               })

            print('nsim:%d\t cell err: %.3f pose err: %.3f (m) %.3f (deg)'%\
              (self.n_sims,err_cell, err_xy, err_yaw))

        def loss(masses):
            if tuple(masses) not in cache:
                simulate(masses)
            return cache[tuple(masses)]['loss']

        def grad(masses):
            if tuple(masses) not in cache:
                simulate(masses)
            return cache[tuple(masses)]['grad']

        # initialize
        masses = []
        if cellinfos_init is None:
            for name in self.meta_targets:
                cellinfo = self.sim.getCellBodyProperties(name)
                for c, info in enumerate(cellinfo):
                    masses.append((self.mass_minmax[1]-self.mass_minmax[0])*0.5)

                if self.is_vis:
                    for c, info in enumerate(cellinfo):
                        if c%3==0:
                            info['color'] = [0.4,0.4,0.4,1]
                        elif c%3==1:
                            info['color'] = [0.6,0.6,0.6,1]
                        else:
                            info['color'] = [0.5,0.5,0.5,1]
                    self.sim.setCellBodyProperties(name, cellinfo)
        else:
            for info in cellinfos_init:
                masses.append(info['mass'])

        it = 0
        masses_cur = list(masses)
        while it <= n_iters:

            g = grad(masses_cur)
            masses_cur = masses_cur - g*0.9**it

            for m, mass in enumerate(masses_cur):
                if self.mass_minmax[0] > info['mass']:
                    masses_cur[m] = self.mass_minmax[0]
                elif self.mass_minmax[1] < info['mass']:
                    masses_cur[m] = self.mass_minmax[1]

            it = it + 1

            if time.time()-time_start > timelimit:
                break

            if self.n_sims > nsimslimit:
                break

            if err_min[0] <= errlimit:
                break

        simulate(masses_cur)

        self.cellinfos_infer = history[-1]['cellinfos']
        time_end = time.time()
        self.time_infer = time_end - time_start

        # evaluation
        for i, h in enumerate(history):
            summary = self.evaluate(h['cellinfos'])
            print('eval:%d\t cell err: %.3f pose err: %.3f (m) %.3f (deg)'%\
                  (i,summary['error_cell'], summary['error_xy'], summary['error_yaw']))
            for key in summary:
                h['test_' + key] = summary[key]
        return history

if __name__=='__main__':

    obj = 'hammer'
    idxes_train = [4,1,7,8]
    idxes_test =  [0,2,3,5,6,9]
    #idxes_train = [1]
    #idxes_test = [1]
    idx_param = 9

    from setups_sim import get_setups_tools_sim
    setups = get_setups_tools_sim([obj],idxes_train,idxes_test,idx_param)

    engine = DiffPhysics(is_dbg=True, fp_video='')
    engine.setup(setups, n_group=-1)
    history = engine.infer(infer_type='cell_mass',nsimslimit=500)
    print('[Test Result] cell err: %.3f (m) pose err: %.3f (m) %.3f (deg)' % 
           (history[-1]['test_error_cell'],
            history[-1]['test_error_xy'],
            history[-1]['test_error_yaw']) )
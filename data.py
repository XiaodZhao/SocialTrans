from typing import Optional, Sequence, List

import os, sys
import torch
import numpy as np

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

# !as_completed 是 Python 的 concurrent.futures 模块中提供的一个函数，用于在一组 Future 对象中迭代出完成的 Future。
# !在你提供的代码片段中，as_completed函数被用来追踪一组并发执行的任务（通过 ProcessPoolExecutor 提交的任务），
# !并在它们完成时进行处理。

# !具体来说，as_completed 函数会接受一个 Future 对象的可迭代集合作为输入，并返回一个生成器。
# !在迭代这个生成器时，它会实时地 yield 出已经完成的 Future 对象，而不会阻塞等待所有任务都完成。
# !这样可以让你在任务完成时立即处理结果，而不需要等待所有任务都执行完毕。


class Dataloader(torch.utils.data.Dataset):

    class FixedNumberBatchSampler(torch.utils.data.sampler.BatchSampler):
        def __init__(self, n_batches, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_batches = n_batches
            # print(n_batches,'222222222222')
            self.sampler_iter = None #iter(self.sampler)
        def __iter__(self):
            # !特殊方法，使得对象成为可迭代的，允许在对象上进行迭代操作。当你在类中实现了这个方法后，实例对象可以在for循环中使用
            # !实现该方法可以提供一个迭代器，该迭代器必须实现两个方法__iter__&__next__，前者返回迭代器对象本身，后者返回容器的下一个项目，
            # !如果全部返回，后者应该抛出一个'StopIteration'异常来通知迭代的中止
            # same with BatchSampler, but StopIteration every n batches
            # !与BatchSampler相同，但每n个批次就会抛出StopIteration错误。
            counter = 0
            batch = []
            while True:
                if counter >= self.n_batches:
                    break
                if self.sampler_iter is None: 
                    self.sampler_iter = iter(self.sampler)
                try:
                    idx = next(self.sampler_iter)
                except StopIteration:
                    self.sampler_iter = None
                    if self.drop_last: batch = []
                    continue
                batch.append(idx)
                if len(batch) == self.batch_size:
                    counter += 1
                    yield batch
                    batch = []
    # !这个在配置文件中设定 batch_size: int
    def __init__(self,
        files: List[str], 
        ob_horizon: int, # !
        pred_horizon: int, # !
        batch_size: int, # !
        drop_last: bool=False, # ?
        shuffle: bool=False,    # !
        batches_per_epoch=None, # ?
        frameskip: int=1, # !
        inclusive_groups: Optional[Sequence]=None,  # !
        batch_first: bool=False,    # !
        seed: Optional[int]=None,   # !
        device: Optional[torch.device]=None,  # !
        flip: bool=False,   # ?
        rotate: bool=False,   # ?
        scale: bool=False   # ?
    ):
        super().__init__()
        self.ob_horizon = ob_horizon
        self.pred_horizon = pred_horizon
        self.horizon = self.ob_horizon+self.pred_horizon
        self.frameskip = int(frameskip) if frameskip and int(frameskip) > 1 else 1
        self.batch_first = batch_first
        self.flip = flip
        self.rotate = rotate
        self.scale = scale
        # print(self.flip,self.rotate,self.rotate,'flip,rotate,scale')
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu") 
        else:
            self.device = device
        # print('This is files',files)
        if inclusive_groups is None:
            inclusive_groups = [[] for _ in range(len(files))]
        assert(len(inclusive_groups) == len(files))
        # print('This is files inclusive_groups',inclusive_groups)
        print(" Scanning files...")
        files_ = []
        # print('start jiansuo')
        # print(files)
        for path, incl_g in zip(files, inclusive_groups):
            if os.path.isdir(path):
                files_.extend([(os.path.join(root, f), incl_g) \
                    for root, _, fs in os.walk(path) \
                    for f in fs if f.endswith(".txt")])
            elif os.path.exists(path):
                files_.append((path, incl_g))
        data_files = sorted(files_, key=lambda _: _[0])       #对被排列对象按照第多少个元素进行排序
        # print(data_files,'This is data_files')
        data = []

        done = 0
        # too large of max_workers will cause the problem of memory usage
        max_workers = min(len(data_files), torch.get_num_threads(), 20)
        # print(max_workers)

        with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"), max_workers=max_workers) as p:
            futures = [p.submit(self.__class__.load, self, f, incl_g) for f, incl_g in data_files]      
            # !在静态方法中
            # !@staticmethod
            # !def load(self, filename, inclusive_groups):
            # !as_completed用于追踪
            for fut in as_completed(futures):
                done += 1
                sys.stdout.write("\r\033[K Loading data files...{}/{}".format(done, len(data_files)))
            itemzxd = fut.result()
            # print(itemzxd[0][0].shape,'11111111111111111')
            # print(itemzxd[0][1].shape,'11111111111111111')
            # print(itemzxd[0][2].shape,'11111111111111111')
            for fut in futures:
                item = fut.result()
                if item is not None:
                    data.extend(item)
                sys.stdout.write("\r\033[K Loading data files...{}/{} ".format(done, len(data_files)))

        self.data = np.array(data, dtype=object)
        del data
        # print('loadloadloadloadloadload')
        # !显示轨迹加载完成
        print("\n   {} trajectories loaded.".format(len(self.data)))

        self.rng = np.random.RandomState()
        if seed: self.rng.seed(seed)
        # !在这里定义了一个数据采样器sampler
        # !如果torch.utils.data.sampler.RandomSampler()在一个类(这个类表示一个数据集)的方法内被调用，则输入参数为self
        # !在这种情况下，self指的是数据集实例本身
        # !这意味着，正在将当前的数据集实例传递给'RandomSampler'，它会使用这个数据集的长度即'__len__'方法返回的数值
        # !来随机选择索引，从而在每次迭代时抽取数据集
        if shuffle:   #   如果为真，则说明打乱数据，那么将会进行随机采样
            sampler = torch.utils.data.sampler.RandomSampler(self)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(self)  #   这里是顺序采样
        if batches_per_epoch is None:
            self.batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
            self.batches_per_epoch = len(self.batch_sampler)
            # print(self.batch_sampler,len(self.batch_sampler),'11111111111111')
        else:
            self.batch_sampler = self.__class__.FixedNumberBatchSampler(batches_per_epoch, sampler, batch_size, drop_last)
            self.batches_per_epoch = batches_per_epoch

    def collate_fn(self, batch): 
        # print('-------------看看batch-----------------------------') 
        # print(len(batch))
        # !数据加载器'DataLoader'加载数据，从'Dataset'中采集样本。前者根据batch_size一次收集多个样本
        # !这些样本自然形成了一个列表，其中包含了该批次所有的数据，以便模型进行并行处理
        # !传入的batch是一个列表数据，len(batch)的数值就是配置文件和中的BATCH_SIZE = 128   
        # print('\n','this is batch')
        # print(len(batch))
        X, Y, NEIGHBOR = [], [], []
        # print('33333333')
        # print(batch[0],'33333333')
        jishu = 0   # !他的大小其实就是batch_size的大小
        for item in batch:
            jishu += 1
            hist, future, neighbor = item[0], item[1], item[2]        # 具体的需要参加load函数中的处理方式

            hist_shape = hist.shape
            neighbor_shape = neighbor.shape
            hist = np.reshape(hist, (-1, 2))
            neighbor = np.reshape(neighbor, (-1, 2))
            # !load返回的item加载回来，使用(-1, 2)是为了把[x,y,vx,vy,ax,ay]
            # !变为三列，分别是[x,y],[vx,vy],[ax,ay]
            # 
            # print('1111')
            # print(hist_shape,neighbor_shape)
            # !下面三个判断都不会进入
            # !flip: bool=False, 
            # !rotate: bool=False, 
            # !scale: bool=False
            if self.flip:       
                if self.rng.randint(2):    #  随即地返回0或者1
                    # 返回1的话，对最后一个维度的第1个索引的所有数值取相反数
                    hist[..., 1] *= -1
                    future[..., 1] *= -1
                    neighbor[..., 1] *= -1
                if self.rng.randint(2):
                    # 返回0的话，对最后一个维度的第0个索引的所有数值取相反数
                    hist[..., 0] *= -1
                    future[..., 0] *= -1
                    neighbor[..., 0] *= -1      
            if self.rotate:
                rot = self.rng.random() * (np.pi+np.pi)     # 生成0-1之间的随机浮点数与两个2pi相乘
                s, c = np.sin(rot), np.cos(rot)
                r = np.asarray([
                    [c, -s],
                    [s,  c]
                ])
                hist = (r @ np.expand_dims(hist, -1)).squeeze(-1)   # 在最后一个维度加中括号，升维度，矩阵作乘之后再去处维度
                future = (r @ np.expand_dims(future, -1)).squeeze(-1)
                neighbor = (r @ np.expand_dims(neighbor, -1)).squeeze(-1)
            if self.scale:
                s = self.rng.randn()*0.05 + 1 # N(1, 0.05)
                hist = s * hist
                future = s * future
                neighbor = s * neighbor
            hist = np.reshape(hist, hist_shape)
            neighbor = np.reshape(neighbor, neighbor_shape)
            #!这个地方又把hist和neighbor的shape转换回去了，
            # print(jishu,'jishu')
            # print(hist.shape)
            # print(future.shape)
            # print(neighbor.shape)
            X.append(hist)
            Y.append(future)
            NEIGHBOR.append(neighbor)
            # if jishu == 2:
            #     print('\n',len(X),'hhhhhhhhhhhhhhhhhh')
            #     print(X)
        # print('\n')
        # print(X[1],'X')
        # !n.shape[1]就是N - 1 = len(idx) + len(neighbor_idx) - 1
        n_neighbors = [n.shape[1] for n in NEIGHBOR]   # 每一批次中，把邻居的数量作为，找最大的邻居列表
        # print(n_neighbors,'nnnnnnnnnnnnnnnnnnn')
        max_neighbors = max(n_neighbors) 
        if max_neighbors != min(n_neighbors):
            NEIGHBOR = [
                np.pad(neighbor, ((0, 0), (0, max_neighbors-n), (0, 0)),
                "constant", constant_values=1e9)
                for neighbor, n in zip(NEIGHBOR, n_neighbors)
            ]

            # !(0, 0)：表示在第一个维度上不进行填充。
            # !(0, max_neighbors-n)：表示在第二个维度上向右填充 max_neighbors-n 个元素。
            # !(0, 0)：表示在第三个维度上不进行填充。

            # 填充只在索引1的维度进行操作,让邻居的shape一样，用1e9无穷远处代表没有邻居
        
            
        stack_dim = 0 if self.batch_first else 1     
        # !self.batch_first始终为false，数据全部堆叠在数值为1的维度上
        # !N = len(idx) + len(neighbor_idx)
        x = np.stack(X, stack_dim)    # ! L_ob x jishu x 6
        y = np.stack(Y, stack_dim)    # ! L_pred x jishu x 6
        neighbor = np.stack(NEIGHBOR, stack_dim)    # ! L_ob x jishu x (N-1) x 6
        # print('---------------转化为torch类型之前----------------------------')
        # print(jishu,'jishu')
        # print(x.shape)
        # print(y.shape)
        # print(neighbor.shape)
        
        # 再将数据转化为torch类型的
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        neighbor = torch.tensor(neighbor, dtype=torch.float32, device=self.device)
        # print('this x')
        # print('hello')
        # print('\nThis collate_fn-----------------')
        # print(x.shape)
        # print(y.shape)
        # print(neighbor.shape)
        # print('-------------------------------')
        return x, y, neighbor

    def __len__(self):
        # !允许一个对象支持内置的len()函数，从而能够返回集合中元素的数量。
        # !当定义了自己的类时，并希望能够使用len()获得该实例的长度或大小时，就需要实现这个方法
        # print('this is data')
        # print(self.data)
        return len(self.data)

    def __getitem__(self, idx):
        # !实现对象的索引操作，使得对象支持下标访问，允许类的实例表现得像序列或映射，可以使用索引或键来获取数值
        # print('this is idx')
        # print(self.data[idx])
        return self.data[idx]

    # !是一个静态方法，通过类直接调用，而不需要一个类的实例
    @staticmethod
    def load(self, filename, inclusive_groups):        # [p.submit(self.__class__.load, self, f, incl_g) for f, incl_g in data_files]
        if os.path.isdir(filename): return None

        horizon = (self.horizon-1)*self.frameskip   # !self.horizon = self.ob_horizon+self.pred_horizon
        # print(horizon)
        with open(filename, "r") as record:
            data = self.load_traj(record)   # !数据读出的形式是字典的嵌套data[t][idx] = [x, y, group]
        data = self.extend(data, self.frameskip)   # !self.frameskip为1
        time = np.sort(list(data.keys()))
        if len(time) < horizon+1: return None
        valid_horizon = self.ob_horizon + self.pred_horizon

        traj = []
        e = len(time)   # !用来获取总的时间量程
        # print(e)
        tid0 = 0     # 从第零帧开始
        while tid0 < e-horizon:   # 必须得留出一个horizon的
            tid1 = tid0+horizon
            t0 = time[tid0]    # 取得tid0对应的时间

            idx = [aid for aid, d in data[t0].items() if not inclusive_groups or any(g in inclusive_groups for g in d[-1])]
            # 满足后面的条件，列表中才会存储他们的元素
            #    inclusive_groups在上面的data_files中，是空的列表
            # d in data[t0].items()    不可用来单独判断，都会是false
            # !根据下面的结构可以了解，data[t0].items()里面嵌套的是字典，aid获取idx，d获取对应的数值
            # {1: [8.46, 3.59, 1.1099999999999994, 0.20000000000000018, 0.0, 0.0, None]}
            # idx 是数据集的第二列 ,用来存储每一帧里面人的标签



            if idx:    # !取出当前之后horizon帧的idx，他们在当前帧出现过
                idx_all = list(data[t0].keys())
                for tid in range(tid0+self.frameskip, tid1+1, self.frameskip):   # 在horizon的帧数以内
                    t = time[tid]
                    idx_cur = [aid for aid, d in data[t].items() if not inclusive_groups or any(g in inclusive_groups for g in d[-1])]
                    if not idx_cur: # ignore empty frames     idx直接归零，重新开始
                        tid0 = tid
                        idx = []
                        break
                    idx = np.intersect1d(idx, idx_cur)   # 对他们俩取交集,在horizon帧数范围内的交集
                    # 没有交集的话，idx变成[]，主动从循环中推出，
                    # 可以理解为在没有到预定帧数的范围内，就与idx没有了交集，idx中的某些id没有全部出现在循化的frame里
                    if len(idx) == 0: break     # 没有交集的话主动本次当前循环
                    idx_all.extend(data[t].keys())    # 为寻找idx的邻居做准备
            # print(idx,'\t',tid0)
            if len(idx):
                # print(tid0,time[tid0],'tid0')
                data_dim = 6  
                neighbor_idx = np.setdiff1d(idx_all, idx)         # !求两个数组的差集，并且进行去重操作
                # print(neighbor_idx)
                if len(idx) == 1 and len(neighbor_idx) == 0:
                    # print(idx,'idx')
                    # print(neighbor_idx)
                    agents = np.array([[data[time[tid]][idx[0]][:data_dim]] + [[1e9]*data_dim] for tid in range(tid0, tid1+1, self.frameskip) ]) # L x 2 x 6
                    # print(agents)
                else:
                    agents = np.array([
                        [data[time[tid]][i][:data_dim] for i in idx] +
                        [data[time[tid]][j][:data_dim] if j in data[time[tid]] else [1e9]*data_dim for j in neighbor_idx]
                        for tid in range(tid0, tid1+1, self.frameskip)
                    ])  
                    r"""
                    # L X N x 6        N = len(idx) + len(neighbor_idx)
                    # if len(idx)==3 and len(neighbor_idx)==10:
                    #     print('-----------------------agents------------------------------')
                    #     # print(len(idx),len(neighbor_idx))
                    #     # np.savez('agents',agents)
                    #     print(idx)

                    #  for tid in range(tid0, tid1+1, self.frameskip) 是第一列的时间序列
                    #  有了tid从而有  for i in idx   从而对第二列进行遍历，再用[:data_dim]进行索引
                    # print(len(idx),len(neighbor_idx))
                    # print(agents.shape,'aaaaaaaaaaaaaaaaaaaaaa')
                    """
                    
                for i in range(len(idx)):
                    
                    hist = agents[:self.ob_horizon,i]  # !L_ob x 6       8
                    future = agents[self.ob_horizon:valid_horizon,i,:2]  # !L_pred x 2      12
                    neighbor = agents[:valid_horizon, [d for d in range(agents.shape[1]) if d != i]] # !L x (N-1) x 6     20   7     6
                    # 不取当前i索引的数据，其他的都是邻居，有的邻居1e9在很远处，其实也就是没有
                    traj.append((hist, future, neighbor))
            tid0 += 1

        # print(len(traj),'tratratratratratratratratratratra')
        items = []
        for hist, future, neighbor in traj:
            hist = np.float32(hist)
            future = np.float32(future)
            neighbor = np.float32(neighbor)
            items.append((hist, future, neighbor))
            # print(neighbor.shape)
            # 所以每一个item里的每一个是历史，未来和邻居
        # print('\nThis is load--------------------------------')
        # print('len(items)',len(items))
        # print(items[0][0].shape)
        # print(items[0][1].shape)
        # print(items[0][2].shape)
        return items
    # !load引用def extend以及def load_traj处理数据，最终得到items.append((hist, future, neighbor))

    def extend(self, data, frameskip):
        time = np.sort(list(data.keys()))
        dts = np.unique(time[1:] - time[:-1])       
        # !让后一个时间步减去前一个时间步，1->end&0->(end-1),就是对数据的第一列进行处理
        dt = dts.min()   # 找到最小的时间差
        if np.any(dts % dt != 0):
            # !dts % dt != 0这里面是否至少有一个元素为True
            raise ValueError("Inconsistent frame interval:", dts)         # 帧间隔不一致
        i = 0
        while i < len(time)-1:
            if time[i+1] - time[i] != dt:
                time = np.insert(time, i+1, time[i]+dt)
            i += 1
        # !这个地方是把第一列数据中的时间差统一为最小的时间差
        # !ignore those only appearing at one frame    忽略那些只出现一帧的
        for tid, t in enumerate(time):
            removed = []
            if t not in data: data[t] = {}
            for idx in data[t].keys():          # data[t].keys()指的是data字典中t键对应的字典中的键，也就是数据集的第二列
                t0 = time[tid-frameskip] if tid >= frameskip else None
                t1 = time[tid+frameskip] if tid+frameskip < len(time) else None
                if (t0 is None or t0 not in data or idx not in data[t0]) and (t1 is None or t1 not in data or idx not in data[t1]):
                    removed.append(idx)
            for idx in removed:
                data[t].pop(idx)           #删除t键值下对应的字典的idx键的信息。并且返回删除的信息（这个在jupyter中会显示）
        # extend v      速度
        for tid in range(len(time)-frameskip):
            t0 = time[tid]
            t1 = time[tid+frameskip]
            if t1 not in data or t0 not in data: continue      # !这里判断的是t0和t1是不是data 的最外层的键里
            for i, item in data[t1].items():     # 把t1键下嵌套的字典进行键和元素遍历
                if i not in data[t0]: continue     # 如果
                x0 = data[t0][i][0]
                y0 = data[t0][i][1]
                x1 = data[t1][i][0]
                y1 = data[t1][i][1]
                vx, vy = x1-x0, y1-y0
                data[t1][i].insert(2, vx)        #在 data[t1][i]插入vx
                data[t1][i].insert(3, vy)
                if tid < frameskip or i not in data[time[tid-1]]:
                    data[t0][i].insert(2, vx)
                    data[t0][i].insert(3, vy)
        # extend a      加速度
        for tid in range(len(time)-frameskip):
            t_1 = None if tid < frameskip else time[tid-frameskip]
            t0 = time[tid]
            t1 = time[tid+frameskip]
            if t1 not in data or t0 not in data: continue
            for i, item in data[t1].items():
                if i not in data[t0]: continue
                vx0 = data[t0][i][2]
                vy0 = data[t0][i][3]
                vx1 = data[t1][i][2]
                vy1 = data[t1][i][3]
                ax, ay = vx1-vx0, vy1-vy0
                data[t1][i].insert(4, ax)
                data[t1][i].insert(5, ay)
                if t_1 is None or i not in data[t_1]:
                    # first appearing frame, pick value from the next frame      # 第一个出现的帧，从下一帧中拾取值
                    data[t0][i].insert(4, ax)
                    data[t0][i].insert(5, ay)
        return data         #  至此，数据集被整合为[t,idx,x,y,vx,vy,ax,ay]

    def load_traj(self, file):
        data = {}
        for row in file.readlines():         # file.readlines()最终得到的是一个列表
            item = row.split()
            if not item: continue
            t = int(float(item[0]))
            idx = int(float(item[1]))
            x = float(item[2])
            y = float(item[3])
            group = item[4].split("/") if len(item) > 4 else None
            if t not in data:   # !检查是否存在t这个键值了，不在的话创建一个空的再赋值
                data[t] = {}
            data[t][idx] = [x, y, group]        
        # 在这里数据读出的形式是字典的嵌套
        # print('This is data')
        # print(data)
        return data

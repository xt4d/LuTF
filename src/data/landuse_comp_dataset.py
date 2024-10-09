import torch
import pickle
import numpy as np
import os
from skimage import measure
from scipy import ndimage


class LanduseCompDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        data: list, 
        class_names: list, 
        seq_order: list = None, 
        use_geo_measures: bool = False, 
        use_progress: bool = False, 
        use_plan: bool = False, 
        comp_type: str = 'masked', 
        vis_type: str = None, 
        use_random_comb: bool = False,
        replace_mask: bool = False,
        width: int = 32, 
        height: int = 32, 
        device: str = 'cpu'
    ):

        max_length = width * height

        self.data = data
        self.class_map = { name:idx for idx, name in enumerate(class_names) }
        self.device = device
        self.width = width
        self.height = height
        self.max_length = max_length
        self.token_dim = len(class_names)

        self.class_names = class_names

        self.use_geo_measures = use_geo_measures
        self.use_progress = use_progress
        self.use_plan = use_plan

        self.seq_order = seq_order

        self.comp_type = comp_type
        self.vis_type = vis_type
        self.use_random_comb = use_random_comb

        self.replace_mask = replace_mask


    @staticmethod
    def load_data_list(data_root, data_list_path):
        data_list = []
        with open(data_list_path, 'r') as fin:
            for fname in fin:
                data_list.append(os.path.join(data_root, fname.strip()))
        
        return data_list


    @staticmethod
    def load_data_pkl(fpath_list):
        dataset = []
        for fpath in fpath_list:
            with open(fpath, 'rb') as fin:
                data = pickle.load(fin)
                dataset += data
        return dataset


    def get_class_names(self):
        return self.class_names


    def get_class_name(self, idx):
        return self.class_names[idx]


    def get_class_num(self):
        return len(self.get_class_names())


    def get_class_id(self, name):
        if name not in self.class_map:
            return -1
        return self.class_map[name]


    def get_name_from_gtype(self, gtype):
        vals = gtype.split(';')
        v = np.random.choice(vals, size=1)[0]
        return v


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        try:
            with open(self.data[index], 'rb') as fin:
                x_raw = pickle.load(fin)
        except:
            print(self.data[index], 'not found')
            return {
                'x': torch.zeros(self.max_length, dtype=torch.long, device=self.device)
            }

        x_seq = np.array(x_raw)

        if self.seq_order is not None:
            x_seq = x_seq[self.seq_order]

        data_dict = {
            'x': torch.tensor(x_seq, dtype=torch.long, device=self.device)
        }

        if self.use_geo_measures:

            x_dist, x_direct, x_degree = self.get_geo_measures(
                x_seq.reshape((self.height, self.width))
            )

            x_dist = np.array(x_dist).reshape(self.max_length)
            x_dist = x_dist / np.sqrt(self.height**2 + self.width**2)

            x_direct = np.array(x_direct).reshape(self.max_length)
            x_direct = (x_direct + np.pi) / (2 * np.pi)

            x_degree = np.array(x_degree).reshape(self.max_length)
            x_degree = x_degree / 4.0

            data_dict['dist'] = torch.tensor(x_dist, dtype=torch.float, device=self.device)
            data_dict['direct'] = torch.tensor(x_direct, dtype=torch.float, device=self.device)
            data_dict['degree'] = torch.tensor(x_degree, dtype=torch.float, device=self.device)

        if self.use_progress:
            comp_progress = [[ np.sum(x_seq[idx:] == ci) / self.max_length for ci in range(0, len(self.class_names)) ] for idx in range(0, len(x_seq))]
            data_dict['progress'] = torch.tensor(comp_progress, dtype=torch.float, device=self.device)
        
        if self.use_plan:
            comp_plan = [[ np.sum(x_seq[idx:] == ci) / (self.max_length - idx) for ci in range(0, len(self.class_names)) ] for idx in range(0, len(x_seq))]
            data_dict['plan'] = torch.tensor(comp_plan, dtype=torch.float, device=self.device)

        vis_type = self.vis_type
        comp_type = self.comp_type

        if self.use_random_comb:
            options = [(t1, t2) for t1 in [None, 'grid', 'stroke'] for t2 in [None, 'masked'] ]
            vis_type, comp_type = options[ np.random.randint(len(options)) ]
        
        '''none visible: fully masked'''
        mask = np.zeros(self.max_length, dtype=np.int32)

        if vis_type == 'grid':
            mask = self.generate_grid_mask(4, 4).reshape(self.max_length)
        elif vis_type == 'stroke':
            mask = self.generate_stroke_mask(self.width, self.height).reshape(self.max_length)
        elif vis_type == 'edge':
            mask = self.generate_edge_mask(self.width, self.height).reshape(self.max_length)

        mx_seq = mask * x_seq

        if self.replace_mask:
            mx_seq = mask * x_seq + (1 - mask) * np.random.randint(1, len(self.class_names), size=self.max_length)

        data_dict['mx'] = torch.tensor(mx_seq, dtype=torch.long, device=self.device)

        '''none comp'''
        comp_name = np.array([ i for i in range(len(self.class_names))])
        comp_ratio = np.zeros(len(self.class_names), dtype=np.float32)
        comp_ratio[0] = 1.0

        if comp_type is not None:

            xc = [ 0 for i in range(len(self.class_names)) ]
            x_hidden = x_seq[mx_seq == 0]

            for cid in x_hidden:
                if cid >= 0 and cid < len(self.class_names):
                    xc[cid] += 1

            sum_xc = np.sum(xc)
            if sum_xc == 0:
                sum_xc = 1

            comp_ratio = np.array(xc) / sum_xc

        comp = np.concatenate((np.expand_dims(comp_name, -1), np.expand_dims(comp_ratio, -1)), axis=-1)

        data_dict['comp'] = torch.tensor(comp, dtype=torch.float, device=self.device)

        return data_dict


    def get_geo_measures(self, img):

        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        def rotation(p1, p2):
            dy = p2[0] - p1[0]
            dx = p2[1] - p1[1]
            return np.arctan2(dy, dx)

        def degree4(img, y, x):
            d = 0
            if y >= 1:
                d += (img[y][x] != img[y-1][x])
            if y < img.shape[0] - 1:
                d += (img[y][x] != img[y+1][x])
            if x >= 1:
                d += (img[y][x] != img[y][x-1])
            if x < img.shape[1] - 1:
                d += (img[y][x] != img[y][x+1])
            return d
    
        assert len(img.shape) == 2 and img.shape[0] == self.height and img.shape[1] == self.width
        
        labels = measure.label(img)

        mass_centers = [(0., 0.)] + ndimage.center_of_mass(labels, labels, [ i for i in range(1, labels.max() + 1) ])
        
        img_dist = [ [ distance( mass_centers[labels[y][x]], (y, x) ) for x in range(img.shape[1]) ] for y in range(img.shape[0]) ]
        img_direct = [ [ rotation( mass_centers[labels[y][x]], (y, x) ) for x in range(img.shape[1]) ] for y in range(img.shape[0]) ]
        img_degree = [ [ degree4(img, y, x) for x in range(img.shape[1]) ] for y in range(img.shape[0]) ]
        
        return img_dist, img_direct, img_degree


    def generate_grid_mask(self, gw, gh):

        assert (self.width % gw) == 0 and (self.height % gh) == 0

        sw = self.width // gw
        sh = self.height // gh

        mask = np.random.randint(2, size=(gh, gw))

        mask = mask.reshape((gh, gw, 1))
        mask = np.repeat(mask, sh, axis=0)
        mask = np.repeat(mask, sw, axis=1)
        mask = mask.reshape((self.height, self.width))

        return mask


    def generate_stroke_mask(self, width, height, thick=1, max_parts=10, max_vertex=100):

        def np_free_form_region(h, w, max_vertex):
            vis = np.zeros((h, w), np.int32)
            num_vertex = np.random.randint(max_vertex + 1)

            startY = np.random.randint(h)
            startX = np.random.randint(w)

            dX = [ 0,  1,  0,  -1]
            dY = [-1,  0,  1,   0]

            for i in range(num_vertex):

                vis[startY-thick:startY+thick+1, startX-thick:startX+thick+1] = 1
                
                cands = []

                for d in range(0, len(dX)):
                    nextY = startY + dX[d]
                    nextX = startX + dY[d]
                    nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int32)
                    nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int32)
                    #if vis[nextY, nextX] == 0:
                    cands.append((nextY, nextX))

                if len(cands) < 1:
                    break

                startY, startX = cands[np.random.randint(len(cands))]

            return vis

        vis = np.zeros((height, width), dtype=np.int32)
        parts = np.random.randint(1, max_parts + 1)
        # print(parts)
        for i in range(parts):
            vis = vis + np_free_form_region(height, width, max_vertex)
        vis = np.minimum(vis, 1)
        return vis


    def generate_edge_mask(self, width, height):
    
        vis = np.zeros((height, width), dtype=np.int32)
        cx = np.random.randint(1, width)
        cy = np.random.randint(1, height)
        
        v0 = np.random.randint(2)
        v1 = np.random.randint(2)
        v2 = np.random.randint(2)
        v3 = np.random.randint(2)
        
        if v0 == v1 and v1 == v2:
            v3 = 1 - v0
        
        vis[:cx, :cy] = v0
        vis[cx:, :cy] = v1
        vis[:cx, cy:] = v2
        vis[cx:, cy:] = v3
        
        return vis

from PIL import Image
import numpy as np
import torch

class_colors = {
    "mask": [0, 0, 0],
    "none": [50, 50, 50],
    "residential": [255, 0, 0],
    "farmland": [255, 255, 0],
    "retail": [255, 150, 0],
    "industrial": [200, 200, 200],
    "meadow": [0, 200, 200],
    "grass": [0, 255, 200],
    "forest": [0, 255, 0],
    "recreation_ground": [0, 0, 255],
    "commercial": [159, 43, 104],
    "railway": [30, 30, 192],
    "cemetery": [100, 100, 100]
}


def comp_vector_using_name(x, class_names):

    return [ (name, x[ci][1]) for ci, name in enumerate(class_names)]


def x_to_comp(x, class_names):
    
    x = np.array(x)
    vec = [ [ci, np.sum(x == ci) / len(x)] for ci in range(0, len(class_names)) ]

    return np.array(vec)


def get_class_color(name):
    assert name in class_colors
    return class_colors[name];


def find_largest_square(mat, v, lsize):
    
    def find_square(mat, y_indcs, x_indcs, size):
        for y, x in zip(y_indcs, x_indcs):
            if np.sum(mat[y:y+size, x:x+size]) == size * size:
                return y, x
        return -1, -1
    
    y_indcs, x_indcs = np.where(mat == v)
    y_indcs = y_indcs.tolist()
    x_indcs = x_indcs.tolist()
    
    sy = -1
    sx = -1
    
    size = mat.shape[0]
    while size >= lsize:
        sy, sx = find_square(mat, y_indcs, x_indcs, size)
        if sy >= 0 and sx >= 0:
            break
        size -= 1
    return sy, sx, size
    
    
def paste_icons(img, mat, class_names, icons):
    
    for ci, name in enumerate(class_names):
        if name not in icons:
            continue
        sy, sx, size = find_largest_square(mat == ci, 1, 3)
        if sy < 0 or sx < 0:
            continue
        print(sy, sx, size, name)
        ico = Image.open(icons[name], 'r')
        ico = ico.resize((size*16, size*16))
        img.paste(ico, (sy*16, sx*16), ico)
        
    return img


def draw_icon_scene(inp, class_names, icon_paths, out_path = None, ext='png'):

    data = np.array(inp)
    if data.ndim == 1:
        assert data.size == 1024
        height = 32
        width = 32
        data = data.reshape(height, width)
    else:
        height, width = data.shape

    comp = [ (name, np.sum(data == cid)) for cid, name in enumerate(class_names) ]

    scale = 16
    sh = height * scale
    sw = width * scale

    fg = create_layout_colormat(data, class_names, add_alpha=False, scale=scale)
    bg = Image.fromarray(np.zeros_like(fg), mode='RGB')

    icon_imgs = { class_names.index(name): Image.open(icon_paths[name]).resize(((width//32)*scale, (height//32)*scale)) for name in icon_paths }

    for y in range(0, 32):
        for x in range(0, 32):
            ci = data[y*(width//32), x*(height//32)]
            if ci not in icon_imgs:
                ci = 0
            icon = icon_imgs[ ci ]
            bg.paste(icon, (x*(width//32)*scale, y*(height//32)*scale), icon)

    fmat = np.array(fg)
    bmat = np.array(bg)
    mask = np.expand_dims((fmat[:, :, 0]== 0) * (fmat[:, :, 1] == 0) * (fmat[:, :, 2] == 0), axis=-1)

    img = Image.fromarray((fmat * (1-mask) + bmat * mask).astype(np.uint8), mode='RGB')

    if out_path:
        img.save(out_path, ext)
    else:
        img.show()

    return comp


def create_layout_colormat(inp, class_names, add_alpha=True, scale=16):

    height, width = inp.shape

    mat = inp.reshape(height, width, 1)
    mat = np.repeat(mat, scale, axis=0)
    mat = np.repeat(mat, scale, axis=1)
    mat = mat.reshape(height*scale, width*scale)

    cmat = np.array([ class_colors[class_names[z]] for z in mat.reshape(mat.size).tolist() ])
    cmat = cmat.reshape(height*scale, width*scale, 3)

    if add_alpha:
        alpha = np.ones((height*16, width*16, 1), dtype=np.uint8) * 255
        cmat = np.concatenate((cmat, alpha), axis=-1)

    return cmat
    

def draw_scene(inp, class_names, out_path=None, ext='png'):

    data = np.array(inp)
    if data.ndim == 1:
        assert data.size == 1024
        height = 32
        width = 32
        data = data.reshape(height, width)
    else:
        height, width = data.shape

    comp = [ (name, np.sum(data == cid) / (height*width)) for cid, name in enumerate(class_names) ]

    cmat = create_layout_colormat(data, class_names, add_alpha=False)

    img = Image.fromarray(cmat.astype(np.uint8), mode='RGB')

    if out_path:
        img.save(out_path, ext)
    else:
        img.show()

    return comp


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def composition_metric(samples, comp_vectors, class_names, max_length=1024):
    
    ds = []

    for sample, comp_v in zip(samples, comp_vectors):
        vec = [ 0.0 for name in class_names ]
        for k in sample:
            vec[k] += 1
        
        d = np.sum(np.abs(np.array(vec) - np.array(comp_v)[:, 1] * max_length)) / (2 * max_length)
        ds.append(d)

    return np.mean(ds), np.std(ds), ds


def random_comp_vector(k, class_names):
    
    choice = np.random.choice([ ci for ci in range(1, len(class_names)) ], size=k, replace=False)
    
    vec = [ np.random.rand() if ci in choice else 0.0 for ci in range(0, len(class_names)) ]
    proba = np.array(vec) / np.sum(vec)
    
    comp_v = [ [ci, proba[ci]] for ci in range(len(class_names)) ]
    
    elements = [ '%s=%.2f' % (class_names[ci], proba[ci]) for ci in range(len(class_names)) if proba[ci] > 0.01 ]
    comp_name = '+'.join(elements)
    
    return comp_v, comp_name

import bpy
from common.interp import linear_interp
from blender.print import print

def set_bits(*args : int):
    value = 0
    for a in args:
        value |= 1 << a
    return value

def encode_symbol(symbol):
    match symbol:
        case 0 | '0': return set_bits(0, 1, 2, 4, 5, 6)
        case 1 | '1': return set_bits(2, 5)
        case 2 | '2': return set_bits(0, 2, 3, 4, 6)
        case 3 | '3': return set_bits(0, 2, 3, 5, 6)
        case 4 | '4': return set_bits(1, 3, 2, 5)
        case 5 | '5': return set_bits(0, 1, 3, 5, 6)
        case 6 | '6': return set_bits(0, 1, 3, 4, 5, 6)
        case 7 | '7': return set_bits(0, 2, 5)
        case 8 | '8': return set_bits(0, 1, 2, 3, 4, 5, 6)
        case 9 | '9': return set_bits(0, 1, 2, 3, 5)
        case '.': return set_bits(7)
        case '0': return set_bits(3, 4, 5, 6)
        case 'E': return set_bits(0, 1, 3, 4, 6)
        case 'r': return set_bits(3, 4)
        case 'd': return set_bits(2, 3, 4, 5, 6)
        case 'L': return set_bits(1, 4, 6, 5)
        case '-': return set_bits(3)
        case None: return 0
    return 0

def encode_symbols(*args):
    return [encode_symbol(a) for a in args]

def encode_float(value, ndigits=6, prec=3):
    s = f'{value:.{prec}f}'
    n = len(s)

    if n > ndigits + prec + 1:
        return encode_symbols(*'Error')

    if n > ndigits + 1:
        nfrac = 10 - n
        s = f'{value:.{nfrac}f}'
        n = len(s)

    l = list(s)
    if '.' in l:
        j = l.index('.')
        l.remove('.')
        encoded = encode_symbols(*l)
        encoded[j-1] |= encode_symbol('.')
    else:
        encoded = encode_symbols(*l)

    npadding = ndigits - len(encoded)
    assert npadding >= 0
    encoded = [0,] * npadding + encoded
    return encoded

def set_display_number(display, value):
    masks = encode_float(value)
    for i in range(6):
        display['Values'][i] = masks[i]

def animate_display(display, tarr, xarr):
  scene = bpy.data.scenes['Scene']
  fps = scene.render.fps
  tstart = tarr[0]
  tend = tarr[-1]
  duration = tend - tstart
  nframes = int(duration * fps + 0.5)

  for i in range(1, nframes + 1):
    t = (i - 1) / fps
    x = linear_interp(tarr, xarr, t)
    set_display_number(display, x)
    display.keyframe_insert(data_path='["Values"]', frame=i)

  fcurves = display.animation_data.action.fcurves
  for fc in fcurves:
    kpts = fc.keyframe_points        
    for kpt in kpts:
      kpt.interpolation = 'CONSTANT'

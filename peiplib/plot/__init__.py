import math

from mpl_toolkits.axes_grid1 import make_axes_locatable


deep_hex = {
    'butter':     '#CCB974',
    'chameleon':  '#55A868',
    'plum':       '#8172B2',
    'scarletred': '#C44E52',
    'skyblue1':   '#64B5CD',
    'skyblue2':   '#4C72B0', }

tango_hex = {
    'butter1':     '#fce94f',
    'butter2':     '#edd400',
    'butter3':     '#c4a000',
    'chameleon1':  '#8ae234',
    'chameleon2':  '#73d216',
    'chameleon3':  '#4e9a06',
    'orange1':     '#fcaf3e',
    'orange2':     '#f57900',
    'orange3':     '#ce5c00',
    'skyblue1':    '#729fcf',
    'skyblue2':    '#3465a4',
    'skyblue3':    '#204a87',
    'plum1':       '#ad7fa8',
    'plum2':       '#75507b',
    'plum3':       '#5c3566',
    'chocolate1':  '#e9b96e',
    'chocolate2':  '#c17d11',
    'chocolate3':  '#8f5902',
    'scarletred1': '#ef2929',
    'scarletred2': '#cc0000',
    'scarletred3': '#a40000',
    'aluminium1':  '#eeeeec',
    'aluminium2':  '#d3d7cf',
    'aluminium3':  '#babdb6',
    'aluminium4':  '#888a85',
    'aluminium5':  '#555753',
    'aluminium6':  '#2e3436', }


tango_RGB = {
    'butter1':     (252, 233,  79),
    'butter2':     (237, 212,   0),
    'butter3':     (196, 160,   0),
    'chameleon1':  (138, 226,  52),
    'chameleon2':  (115, 210,  22),
    'chameleon3':  (78,  154,   6),
    'orange1':     (252, 175,  62),
    'orange2':     (245, 121,   0),
    'orange3':     (206,  92,   0),
    'skyblue1':    (114, 159, 207),
    'skyblue2':    (52,  101, 164),
    'skyblue3':    (32,   74, 135),
    'plum1':       (173, 127, 168),
    'plum2':       (117,  80, 123),
    'plum3':       (92,  53, 102),
    'chocolate1':  (233, 185, 110),
    'chocolate2':  (193, 125,  17),
    'chocolate3':  (143,  89,   2),
    'scarletred1': (239,  41,  41),
    'scarletred2': (204,   0,   0),
    'scarletred3': (164,   0,   0),
    'aluminium1':  (238, 238, 236),
    'aluminium2':  (211, 215, 207),
    'aluminium3':  (186, 189, 182),
    'aluminium4':  (136, 138, 133),
    'aluminium5':  (85,   87,  83),
    'aluminium6':  (46,   52,  54), }


def to01(c):
    return tuple(x/255. for x in c)


def tohex(c):
    return '%02x%02x%02x' % c


def nice_sci_notation(x, ndecimals=2, precision=None, exponent=None):
    """
    https://stackoverflow.com/a/18313780/3202380
    """
    if not exponent:
        exponent = int(math.floor(math.log10(abs(x))))

    coeff = round(x/float(10**exponent), ndecimals)

    precision = precision or ndecimals

    return r"${0:.{1}f}\times10^{{{2:d}}}$".format(coeff, precision, exponent)


def get_cbar_axes(ax, position='right', size='5%', pad='3%'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size, pad=pad)
    return cax

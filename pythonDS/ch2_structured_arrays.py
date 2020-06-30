import numpy as np

name = ['Tom', 'jerry']
body = ['catlike', 'tiny']
wit = [4, 12]


data = np.dtype({'names':('name', 'body', 'wit'),
                            'formats:':('U10', 'U10', 'i8')})

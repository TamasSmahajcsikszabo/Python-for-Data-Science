## importing
# find ot
# compile it
# run it

# the module search path
import sys
sys.path

# mutables can be changed in modules
## example
import small
print(small.y) # originally it's [1,2]

small.y[0] = 12 # it changes

import small
small.x = 123

# to access module's namespace
small.__dict__
dir(small)

# to see only the names generated by the module
list(name for name in small.__dict__ if not name.startswith('__'))

## attribute name qualification
X.Y.Z# look up X in the current scope, and Z within its attribute Y

## no upward visibility: an imported file does not see names in the importer
## it works in the other way (nested namespace): the importer can fetch object names and attributes from a second level onto an imported module within an impored module, i.e.
## the importer has access to three global scopes

## reload forces to reload and rerun already loaded module objects
## it's a function, not a statement, e.g. reload()
## it overwrites the existing namespace

import changer
changer.printer()

from imp import reload
reload(changer)
changer.printer()
# use reload with import not with from

# reload() is useful in GUI, as the GUI can run while the widget's callback action is reloaded
from utils import my_summary
from architecture.MST import MST
my_summary(MST(), 256, 256, 28, 1)

from architecture.mod_mst import MST1
my_summary(MST1(), 256, 256, 28, 1)

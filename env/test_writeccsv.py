import csv
import csv
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

class WCSV:
    def __init__(self, parentdir):
        self.file = csv.writer(open(os.path.join(parentdir, 'neobotix_schunk_pybullet/results/successdada.csv'), "w"))

if __name__ == '__main__':
    float_list = [0.13, 0.20, 3.28]
    ws=WCSV(parentdir)
    ws.file.writerow(float_list)
    ws.file.writerow(float_list)
    ws.file.writerow(float_list)
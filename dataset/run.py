import os
from multiprocessing import Pool
import pandas as pd
blender_path = '/vol/research/ycres/blender-2.79b-linux-glibc219-x86_64/'
os.chdir(blender_path)
def work(arg):
    import os
    path, uid = arg
    os.system('./blender --background --python /vol/research/ycres/syntheticSketch/new_data/code/render_chair.py -- %s %s' % (path, uid))

df = pd.read_csv('/vol/research/ycres/syntheticSketch/new_data/csvs/backup/03001627.csv')

#work_info = [(path, is_test, uid) for path, is_test, uid in df.iloc[:,1:].values]
work_info = [(path, uid) for path, uid in df.iloc[:,1:].values[:, 0:3:1]]
work_info.reverse()
if __name__ == '__main__':
    with Pool(32) as p:
        p.map(work, work_info)
    #for work_in in work_info:

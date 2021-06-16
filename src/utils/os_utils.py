import os
import glob

def delete_dir_and_contents(path):
    try:
        files = glob.glob(path + "/*")
        for f in files:
            os.remove(f)
        os.rmdir(path)
    except:
        pass
import os
rootdir = "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Beat Saber\\CustomSongs"


for filename in os.listdir(rootdir):
    print(os.path.join(rootdir, filename))
"""
    "_lineLayer": 0,
    "_lineIndex": 3,
    "_type": 1,
    "_time": 4.125,
    "_cutDirection": 1

So, our possibilities are:

x,y,t,d = 192 possibilities.

noteRef = {0-32:"empty",
33-224:{"_lineLayer": 0,
    "_lineIndex": 0,
    "_type": 0-12,
    "_time": x,
    "_cutDirection": 0-8}}


"""


import sys, os, json, configparser

def getNoteIndex(lineIndex,lineLayer,colIndex,direction):
    if lineIndex < 0 or lineIndex > 4 or lineLayer < 0 or lineLayer > 3 or type < 0 or type > 1 or direction < 0 or direction > 8:
        return None
    
    return (lineIndex * 54) + (lineLayer * 18) + (colIndex * 9) + direction + 20

noteRef = {}

xRange = 4
yRange = 3
tRange = 2
dRange = 9

for a in range(20):
    noteRef[a] = None

for b in range(236,256):
    noteRef[b] = None

for x in range(xRange):
    for y in range(yRange):
        for t in range(tRange):
            for d in range(dRange):
                i = getNoteIndex(x,y,t,d)
                noteRef[i] =  { "_lineLayer": y,
                            "_lineIndex": x,
                            "_type": t,
                            #"_time": 0, # we can set time afterwards.
                            "_cutDirection": d}









with open(os.getcwd() + '\\noteRef.json', 'w') as outfile:
    json.dump(noteRef, outfile)

print("Wrote noteRef file to: " + os.getcwd())
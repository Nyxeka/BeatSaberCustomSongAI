"""
Song Directory:
---

SongData
>song
>fourier
>easy
>normal
>hard
>expert
>beatfreq
songsList.json
stats.txt

---

Each folder contains the appropriate file, formatted as 00000000.[json,ogg,etc]
this goes on as 00000001, 00000002, 00001234 (for the first, second, and one thousand thirty-fourth song in the list)

We'll output some human-readable data to stats.txt, including: number of songs, number of easy, medium, hard, and expert modes, lowest and highest BPM, etc....

For now, we can skip stat generation, and simply format the folders with what we have.

Steps:
1. move and rename songs etc files.
2. perform fourier on songs
3. generate beat freq file using a neural network.

"""
import sys, os, json, configparser, csv
from time import gmtime, strftime
from shutil import copy2,which
from subprocess import Popen
import librosa as lb 
from multiprocessing.dummy import Pool as ThreadPool 

config = configparser.ConfigParser()

config.read('config.ini')

customSongsFolder = "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Beat Saber\\CustomSongs"

songsList = [["00000000","song-name"]]

songDataDir = "C:\\Users\\njhyl\\Code\\ML\\songMaker\\songData"

def copyFormatSongs(_origSongFolder,_songDataDir):
    numEasy=0
    numNormal=0
    numHard=0
    numExpert=0
    curSongNum=0

    # str(integer).zfill(num_digits)

    statInfo = {"numSongs":0,
    "numEasy":0,
    "numNormal":0,
    "numHard":0,
    "numExpert":0,
    "SongsList" : {"00000000":"song-name"}}

    # make sure the directorys exist

    os.makedirs(_songDataDir + "\\song")
    os.makedirs(_songDataDir + "\\easy")
    os.makedirs(_songDataDir + "\\normal")
    os.makedirs(_songDataDir + "\\hard")
    os.makedirs(_songDataDir + "\\expert")
    os.makedirs(_songDataDir + "\\info")

    for _dir in os.listdir(_origSongFolder):
        
        if not _dir.startswith(".") and os.path.isdir( os.path.join(_origSongFolder, _dir) ):
            curSongNum += 1
            for _file in os.listdir(os.path.join(_origSongFolder,_dir)):
                
                # Find the files we need
                if _file.endswith(".ogg"):
                    songNumStr = str(curSongNum).zfill(8)
                    newSongDir = os.path.join(_songDataDir,"song\\" + songNumStr + ".ogg") # new dir
                    statInfo["SongsList"][songNumStr] = _dir
                    oldSongDir = os.path.join(_origSongFolder,_dir,_file)
                    copy2(oldSongDir,newSongDir)
                    continue
                elif _file.endswith(".json"):
                    if (_file.startswith("Easy")):
                        numEasy += 1
                        songNumStr = str(curSongNum).zfill(8)
                        newSongDir = os.path.join(_songDataDir,"easy\\" + songNumStr + ".json") # new dir
                        oldSongDir = os.path.join(_origSongFolder,_dir,_file)
                        copy2(oldSongDir,newSongDir)
                        continue
                    elif (_file.startswith("Normal")):
                        numNormal += 1
                        songNumStr = str(curSongNum).zfill(8)
                        newSongDir = os.path.join(_songDataDir,"normal\\" + songNumStr + ".json") # new dir
                        oldSongDir = os.path.join(_origSongFolder,_dir,_file)
                        copy2(oldSongDir,newSongDir)
                        continue
                    elif (_file.startswith("Hard")):
                        numHard += 1
                        songNumStr = str(curSongNum).zfill(8)
                        newSongDir = os.path.join(_songDataDir,"hard\\" + songNumStr + ".json") # new dir
                        oldSongDir = os.path.join(_origSongFolder,_dir,_file)
                        copy2(oldSongDir,newSongDir)
                        continue
                    elif (_file.startswith("Expert")):
                        numExpert += 1
                        songNumStr = str(curSongNum).zfill(8)
                        newSongDir = os.path.join(_songDataDir,"expert\\" + songNumStr + ".json") # new dir
                        oldSongDir = os.path.join(_origSongFolder,_dir,_file)
                        copy2(oldSongDir,newSongDir)
                        continue
                    elif (_file.startswith("info")):
                        songNumStr = str(curSongNum).zfill(8)
                        newSongDir = os.path.join(_songDataDir,"info\\" + songNumStr + ".json") # new dir
                        oldSongDir = os.path.join(_origSongFolder,_dir,_file)
                        copy2(oldSongDir,newSongDir)
                        continue
    statInfo["numEasy"] = numEasy
    statInfo["numNormal"] = numNormal
    statInfo["numHard"] = numHard
    statInfo["numExpert"] = numExpert
    statInfo["numSongs"] = curSongNum
    
    return statInfo
    
"""# Now execute the copy
songStats = copyFormatSongs(customSongsFolder,songDataDir)

# Write the stats to the stats file
with open(songDataDir + '\\songsList.json', 'w') as outfile:
    json.dump(songStats, outfile)

"""

def convertToWav(file):
    """
    We want to take all the .ogg files in /songs and convert them to .wav in songData\\wav
    """
    y, sr = lb.load("songData/song/" + file)

    tempo, beat_frames = lb.beat.beat_track(y=y, sr=sr)
    beat_times = lb.frames_to_time(beat_frames, sr=sr)
    lb.output.times_csv("songData/beatTimes/" + file[:-4] + '.csv', beat_times)
    lb.output.write_wav("songData/wav/" + file[:-4] + ".wav" ,y,sr)
    print("Finished processing file: " + file)
    return file[:-4] + ": " + str(len(beat_times))


def startThreadedTaskA(func, array):
    songListArray = []
    for _file in os.listdir("songData/song"):
        songListArray.append(_file)

    infoFile = "songData/beatTimesInfo.txt"
    pool_size = 16

    pool = ThreadPool(pool_size)

    os.makedirs("songData/wav/",mode=0o777,exist_ok=True)
    os.makedirs("songData/beatTimes/",mode=0o777,exist_ok=True)

    results = pool.map(convertToWav,songListArray)

    infoDump = ""

    for a in results:
        infoDump = infoDump + "\n" + a

    with open(infoFile, "w") as outfile:
        outfile.write(infoDump)
 


"""
now we want to grab the note data, extract, compare to closest beat
we want to load the csv and song inf(expert,easy,normal,hard,whatever) into an array.

grab the songData["_note"][x]["_time"] and check its value, 
- then we convert that value into real-time in seconds, by calculating 60/bpm, then multiplying that by the value. This gives us beat-in-seconds.

then we run through the .csv array and look for the closest value to that beat-in-seconds. We'll iterate through until the value of the data is greater than
that of the current csv iterated line's data, compare to the value behind and in front of that line,  and whichever is closest to the time, will be the "new" time.

Then, we convert the xydt data to a single uint8, and throw it next to the current selected time in the .csv

if the value of that time is further away than, say, 1/20th of a second, then we will delete that note, and we will not record it!
We don't want out-of-sync beatmaps anyways.



"""

def getNoteIndex(lineIndex,lineLayer,colIndex,direction):
    if lineIndex < 0 or lineIndex > 4 or lineLayer < 0 or lineLayer > 3 or colIndex < 0 or colIndex > 1 or direction < 0 or direction > 8:
        return None
    
    return (lineIndex * 54) + (lineLayer * 18) + (colIndex * 9) + direction + 20

def build_csv_array(fileName):
    # load up CSV file to our list.

    csvResults = []
    beatMapData = {}
    i = 0
    blankNotes = [  0,0,0,0,
                    0,0,0,0,
                    0,0,0,0  ] # 12 zeroes. Note type reference can be found in noteRef.json info file.
    with open('songData/beatTimes/'+ fileName[:-5] + '.csv', 'r') as csvFile:
        _r = csv.reader(csvFile)
        for row in _r:
            if not row == []:
                csvResults.append(row)
                csvResults[i].extend(blankNotes)
                csvResults[i][0] = float(csvResults[i][0])
                i += 1
    with open('songData/expert/'+ fileName[:-5] + '.json') as beatmap_file:
        beatMapData=json.load(beatmap_file)

    beatTimeSeconds = (60 / beatMapData["_beatsPerMinute"])

    numNotes = 0
    for _a in beatMapData["_notes"]:
        numNotes += 1
        curTime = float(float(_a["_time"]) * beatTimeSeconds)
        noteRef = getNoteIndex(_a["_lineIndex"],_a["_lineLayer"],_a["_type"],_a["_cutDirection"])
        # now interate through the beats to find the closest one to the time listed to the current note.

        for i in range(len(csvResults)):

                dist = abs(csvResults[i][0] - curTime)
                
                if dist < 0.1:
                    # apply the note to the row in behind (i).
                    for _x in range(len(csvResults[i])):

                        if csvResults[i][_x] != 0:
                            continue
                        # print(dist)
                        csvResults[i][_x] = noteRef
                        break # placed the note, we can stop looking.
        

    # now write our new csv file:
    print("Number of notes: " + str(numNotes))
    # print(csvResults)
    newCSV = 'Training-Data/'+ fileName[:-5] + '.csv'
    with open(newCSV, 'w', newline='') as csvFile:
        _w = csv.writer(csvFile,delimiter = ',')
        for _b in csvResults:
            _w.writerow(_b)

def startThreadedTaskB(func):
    songListArray = []
    for _file in os.listdir("songData/expert"):
        songListArray.append(_file)
        
    infoFile = "songData/beatTimesInfo.txt"
    pool_size = 16

    pool = ThreadPool(pool_size)

    os.makedirs("songData/wav/",mode=0o777,exist_ok=True)
    os.makedirs("songData/beatTimes/",mode=0o777,exist_ok=True)

    pool.map(func,songListArray)




# infoDump = convertToWav()

# build_csv_array("00000001.ogg")

# startThreadedTaskB(build_csv_array)

"""
we just want to run through all the note combos and see how many there are. 
so, open CSV's.
read row into array.
sort array.
add array to set if not exists, set it to 1.
if it does exist, add 1 to it in the dictionary.

finally, get total different note-combinations.
"""

import csv, os

noteDict = {"":0}

noteSet = set([])

uniqueNotes = 0

maxNotesSize = 0
total=0

for _file in os.listdir("Training-Data"):
    if (_file != 'eval'):
        with open('Training-Data/'+ _file, 'r') as csvFile:
            _r = csv.reader(csvFile)
            i=0
            for row in _r:
                if row[-12:] != [0,0,0,0,0,0,0,0,0,0,0,0]:
                    total += 1
                    id = str(row[-12:])
                    if id in noteSet:
                        noteDict[id] += 1
                        if noteDict[id] > maxNotesSize:
                            maxNotesSize = noteDict[id]
                    else:
                        uniqueNotes += 1
                        noteSet.add(id)
                        noteDict[id] = 1

# now write the unique note combinations to a .csv file:

newCSV = 'uniqueNotes.csv'

print("Number of note combos: ",uniqueNotes)
print("Maximum note size: ",maxNotesSize)
print("Total Notes Processed: ", total)

sorted_by_value = sorted(noteDict.items(), key=lambda kv: kv[1])

with open(newCSV, 'w', newline='') as csvFile:
    _w = csv.writer(csvFile,delimiter = ',')
    for a, b in sorted_by_value:
        _w.writerow([a,b])



#makes the item file for fastABX

#https://abxpy.readthedocs.io/en/latest/FilesFormat.html
# what we're going for

#source	onset	offset	#label 1	label 2	label 3
# file 1	start 1	stop 1	value 1	value 1	value 1
# file 2	start 2	stop 2	value 2	value 1	value 1
# file 3	start 3	stop 3	value 3	value 1	value 1

import os
from pathlib import Path
from praatio import textgrid

#returns list of (phone, start, end) for a textGrid
def extract(fileName, fullPath):
    
    phones = []

    tg = textgrid.openTextgrid(fullPath, False)
    tier = tg.getTier("phonemes")
    for interval in tier.entries:
        phones.append([interval.label, interval.start, interval.end])
        
    return phones

#returns triphone pairs, with onset + offset for target (center) phone
def makeTriPhones(fileName, phones):

    triPhones = []

    #phones are in format label, intervalStart, intervalEnd

    for i in range(1, len(phones) - 1):

        targetPhone = phones[i]
        prevPhone = phones[i-1]
        nextPhone = phones[i+1]

        #don't take tripples with invalide characters
        if any(p in {"'", "sp", "sil"} for p in (targetPhone[0], prevPhone[0], nextPhone[0])):
            continue
        
        #only target relevant phones
        # if targetPhone[0] not in ['a', 'e', 'i', 'o', 'u', 'ɛ', 'ɔ']:
        #     continue
        
        #if the phone isn't long enough to work with ABX, don't include it.
        if targetPhone[2] - targetPhone[1] < 1/25:
            continue
        
        #returns targetPhone onset, targetPhone offset, targetPhone, previousPhone, nextPhone
        triPhones.append((str(targetPhone[1]), str(targetPhone[2]), targetPhone[0], prevPhone[0], nextPhone[0]))

    return triPhones

if __name__ == "__main__":
    workingDir = os.getcwd()
    evalTextGrids = os.path.join(workingDir, "catalanAlignments", "grids")
    output = os.path.join(workingDir, "fastABX_materials")

    with open(os.path.join(output, "triPhones.item"), "w+", encoding='utf-8') as f:

        f.write("#file onset offset #phone prev-phone next-phone speaker\n")

        for root, dirs, files in os.walk(evalTextGrids):
            for file in files:
                
                phones = extract(file, os.path.join(evalTextGrids, file))
                triPhones = makeTriPhones(file, phones)

                fileName = file.strip(".TextGrid")
                speaker = fileName.split("_")[-1]

                for triple in triPhones:
                    f.write(fileName + " " + (" ").join(triple) + " " + speaker + "\n" )

            f.flush()
    f.close()
                




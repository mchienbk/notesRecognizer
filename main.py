import argparse

from blob_detector import detect_blobs
from getting_lines import get_staffs
from note import *
from photo_adjuster import adjust_photo
from midiutil import MIDIFile


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',default='input/good/medium1.jpg')    
    return parser.parse_args()


def main():
    args = parse()
    image = cv2.imread(args.input)
    adjusted_photo = adjust_photo(image)
    staffs = get_staffs(adjusted_photo)
    blobs = detect_blobs(adjusted_photo, staffs)
    notes = extract_notes(blobs, staffs, adjusted_photo)
    draw_notes_pitch(adjusted_photo, notes)



def recognizer():
    #image link
    image = cv2.imread('samples/fire.jpg')


    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, adjusted_photo = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    staffs = get_staffs(adjusted_photo)
    blobs = detect_blobs(adjusted_photo, staffs)
    notes = extract_notes(blobs, staffs, adjusted_photo)
    image_with_notes = draw_notes_pitch(adjusted_photo, notes)

    # Print notes
    for note in notes: 
        print(note.pitch, "-")

    # Midi-file
    track    = 0
    channel  = 0
    time     = 0    # In beats
    duration = 1    # In beats
    tempo    = 60   # In BPM
    volume   = 100  # 0-127, as per the MIDI standard

    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                        # automatically)
    MyMIDI.addTempo(track, time, tempo)

    i = 0
    for note in notes:
    # for i, pitch in enumerate(notes):
        i += 1
        MyMIDI.addNote(track, channel, note.pitch_rate, time + i, duration, volume)

    with open("output.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)

    cv2.imshow("output file",image_with_notes)
    cv2.waitKey(0)

if __name__ == "__main__":
    # main()
    recognizer()
    cv2.destroyAllWindows()

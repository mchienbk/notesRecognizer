import cv2

from config import NOTE_PITCH_DETECTION_MIDDLE_SNAPPING, VERBOSE
from hu import classify_clef
from util import distance

key = {
    -6: 'MI',
    -5: 'RE',
    -4: 'DO',
    -3: 'SI',
    -2: 'LA',
    -1: 'SO',
    0:  'FA',
    1:  'MI',
    2:  'RE',
    3:  'DO',
    4:  'SI',
    5:  'LA',
    6:  'SO',
    7:  'FA',
    8:  'MI',
    9:  'RE',
    10: 'DO',
    11: 'SI',
    12: 'LA',
    13: 'SO',
    14: 'FA',
}

def extract_notes(blobs, staffs, image):
    clef = classify_clef(image, staffs[0])
    notes = []
    if VERBOSE:
        print('Detected clef: ' + clef)
        print('Extracting notes from blobs.')
    for blob in blobs:
        if blob[1] % 2 == 1:
            staff_no = int((blob[1] - 1) / 2)
            notes.append(Note(staff_no, staffs, blob[0], clef))
    if VERBOSE:
        print('Extracted ' + str(len(notes)) + ' notes.')
    return notes


def draw_notes_pitch(image, notes):
    im_with_pitch = image.copy()
    im_with_pitch = cv2.cvtColor(im_with_pitch, cv2.COLOR_GRAY2BGR)
    for note in notes:
        cv2.putText(im_with_pitch, note.pitch, (int(note.center[0])-20, int(note.center[1]) + 45),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1, color=(255, 0, 0))
    cv2.imwrite('output/9_with_pitch.png', im_with_pitch)


# noinspection PyMethodMayBeStatic
class Note:
    """
    Represents a single note
    """
    def __init__(self, staff_no, staffs, blob, clef):
        self.position_on_staff = self.detect_position_on_staff(staffs[staff_no], blob)
        self.staff_no = staff_no
        self.center = blob.pt
        self.clef = clef
        self.pitch = self.detect_pitch(self.position_on_staff)

    def detect_position_on_staff(self, staff, blob):
        distances_from_lines = []
        x, y = blob.pt
        for line_no, line in enumerate(staff.lines_location):
            distances_from_lines.append((2 * line_no, distance((x, y), (x, line))))
        # Generate three upper lines
        for line_no in range(5, 8):
            distances_from_lines.append((2 * line_no, distance((x, y), (x, staff.min_range + line_no * staff.lines_distance))))
        # Generate three lower lines
        for line_no in range(-3, 0):
            distances_from_lines.append((2 * line_no, distance((x, y), (x, staff.min_range + line_no * staff.lines_distance))))

        distances_from_lines = sorted(distances_from_lines, key=lambda tup: tup[1])
        # Check whether difference between two closest distances is within MIDDLE_SNAPPING value specified in config.py
        if distances_from_lines[1][1] - distances_from_lines[0][1] <= NOTE_PITCH_DETECTION_MIDDLE_SNAPPING:
            # Place the note between these two lines
            return int((distances_from_lines[0][0] + distances_from_lines[1][0]) / 2)
        else:
            # Place the note on the line closest to blob's center
            return distances_from_lines[0][0]

    def detect_pitch(self, position_on_staff):

        return key[position_on_staff]

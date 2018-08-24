import xml.etree.ElementTree as ET
import sys
import shutil

# Move to argparse
first_index = int(sys.argv[1])
last_index = int(sys.argv[2])
input_bb = sys.argv[3]
name = sys.argv[4]

for bb_index in range(first_index, last_index+1):
    file_name = '%s%04d' % (name, bb_index)
    with open(input_bb, 'r') as f1, open(file_name + '.xml', 'w') as f2:
        shutil.copyfileobj(f1, f2)
    
    tree = ET.parse(file_name + '.xml')
    root = tree.getroot()
    root.find('filename').text = file_name + '.jpg'
    print(root.find('filename').text)
    root.find('path').text = '/'.join(root.find('path').text.split('/')[:-1]) + '/' + file_name + '.jpg'
    tree.write(file_name + '.xml')
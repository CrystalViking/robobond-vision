import os

output_file = open('blue_from_pi_neg.txt', 'w')

for filename in os.listdir('blue_from_pi_negative'):  # replace with your directory
    if filename.endswith(".jpg") or filename.endswith(".png"):  # add any file type you need
        output_file.write(os.path.join('blue_from_pi_negative', filename) + '\n')

output_file.close()
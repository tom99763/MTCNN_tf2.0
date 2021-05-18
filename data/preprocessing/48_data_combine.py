import sys
import os

save_dir = "48"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f1 = open(os.path.join(save_dir, 'pos_%s.txt'%(save_dir)), 'r')
f2 = open(os.path.join(save_dir, 'neg_%s.txt'%(save_dir)), 'r')
f3 = open(os.path.join(save_dir, 'part_%s.txt'%(save_dir)), 'r')
f4 = open(os.path.join(save_dir, 'landmark_%s.txt'%(save_dir)), 'r')

pos = f1.readlines()
neg = f2.readlines()
part = f3.readlines()
landmark = f4.readlines()
f = open(os.path.join(save_dir, 'label-train%s.txt'%(save_dir)), 'w')

for i in range(int(len(pos))):
    p = pos[i].find(" ") + 1
    pos[i] = pos[i][:p-1] + ".jpg " + pos[i][p:-1] + "\n"
    f.write(pos[i])

for i in range(int(len(neg))):
    p = neg[i].find(" ") + 1
    neg[i] = neg[i][:p-1] + ".jpg " + neg[i][p:-1] + " -1 -1 -1 -1\n"
    f.write(neg[i])

for i in range(int(len(part))):
    p = part[i].find(" ") + 1
    part[i] = part[i][:p-1] + ".jpg " + part[i][p:-1] + "\n"
    f.write(part[i])

for i in range(int(len(landmark))):
    p = landmark[i].find(" ") + 1
    landmark[i] = landmark[i][:p-1] + ".jpg " + landmark[i][p:-1] + "\n"
    f.write(landmark[i])

f1.close()
f2.close()
f3.close()
f4.close()
